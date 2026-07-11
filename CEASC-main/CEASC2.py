import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
import numpy as np

# ===================== 集成 sparseconv_utils 模块（带可配置参数） =====================
class Mask():
    def __init__(self, hard):
        self.hard = hard
        if hard is None:
            self.n_keep = 0
        else:
            self.n_keep = int(torch.sum(hard).item())
        if self.n_keep == 0:
            self.nonzero_hard = None
        else:
            # 可修改：batch_size支持（当前仅支持1，可扩展）
            self.nonzero_hard = torch.nonzero(hard[0][0], as_tuple=True)


class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    可配置参数：
    - eps: 数值稳定性epsilon
    - gumbel_temp: Gumbel温度系数（控制分布平滑度）
    - gumbel_noise: 是否启用Gumbel噪声
    - threshold: 推理阶段的二值化阈值
    '''
    def __init__(self, eps=1e-8, threshold=0.0):
        super(Gumbel, self).__init__()
        self.eps = eps          # 可修改：数值稳定性参数，默认1e-8
        self.threshold = threshold  # 可修改：推理阶段二值化阈值，默认0.0

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
            gumbel_temp: 可修改：温度系数，默认1.0（值越小越接近离散）
            gumbel_noise: 可修改：是否添加Gumbel噪声，默认True
        Returns:
            Mask对象：包含hard掩码、有效像素数、非零坐标
        """
        if not self.training:  # 推理阶段无噪声
            hard = (x >= self.threshold).float() 
            ans = Mask(hard)
            return ans

        if gumbel_noise:
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1 = -torch.log(-torch.log(U1 + self.eps) + self.eps)
            g2 = -torch.log(-torch.log(U2 + self.eps) + self.eps)
            x = x + g1 - g2

        # Gumbel-Softmax核心计算
        soft = torch.sigmoid(x / gumbel_temp)
        # 直通估计（Straight-Through Estimator）
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard)), "Gumbel模块出现NaN值"
        
        ans = Mask(hard)
        return ans

# ===================== 基础工具函数 =====================
def xyxy2xywh(boxes):
    """将 [x1,y1,x2,y2] 转换为 [x,y,w,h]"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)

def xywh2xyxy(boxes):
    """将 [x,y,w,h] 转换为 [x1,y1,x2,y2]"""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=-1)

def generate_anchors_strides(feature_maps, strides, img_size):
    """生成 FCOS 锚点（每个特征图像素对应原图的坐标）"""
    anchors = []
    for fm, stride in zip(feature_maps, strides):
        h, w = fm.shape[-2:]
        # 生成网格坐标
        x = torch.arange(0, w*stride, stride, dtype=torch.float32) + stride/2
        y = torch.arange(0, h*stride, stride, dtype=torch.float32) + stride/2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        # 拼接为 [x,y] 并限制在图像尺寸内
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        grid = torch.clamp(grid, 0, img_size[0]-1)
        anchors.append(grid)
    return anchors

# ===================== 核心模块：AMM（集成Gumbel，带可配置参数） =====================
class AMM(nn.Module):
    """自适应多层掩码模块（Adaptive Multi-Layer Masking）
    可配置参数：
    - in_channels: 输入通道数
    - feat_channels: 特征通道数
    - gumbel_eps: Gumbel模块数值稳定性参数
    - gumbel_temp: Gumbel温度系数
    - gumbel_noise: 是否启用Gumbel噪声
    - threshold: 推理阶段二值化阈值
    - train_threshold: 训练阶段硬掩码阈值
    - ars_loss_weight: 激活率监督损失权重
    """
    def __init__(self, 
                 in_channels, 
                 feat_channels=256, 
                 gumbel_eps=1e-8,
                 gumbel_temp=1.0,
                 gumbel_noise=True,
                 threshold=0.5,
                 train_threshold=0.5,
                 ars_loss_weight=1.0):
        super().__init__()
        # Gumbel模块参数（可修改）
        self.gumbel = Gumbel(eps=gumbel_eps, threshold=threshold)
        self.gumbel_temp = gumbel_temp        # 可修改：Gumbel温度系数
        self.gumbel_noise = gumbel_noise      # 可修改：是否启用Gumbel噪声
        self.train_threshold = train_threshold  # 可修改：训练阶段阈值
        
        # 掩码生成卷积（可修改）
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        
        # 激活率监督损失（可修改）
        self.ars_loss_weight = ars_loss_weight  # 可修改：损失权重
        self.ars_loss = nn.L1Loss(reduction='mean')

    def forward(self, x, mask_gt=None):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
            mask_gt: 掩码GT (B, 1, H, W) [训练时传入]
        Returns:
            hard_mask: 二值掩码 (B, 1, H, W)
            loss_ars: 激活率监督损失（训练时返回）
        """
        # 1. 生成掩码特征
        mask_feat = self.mask_conv(x)
        
        # 2. 使用Gumbel模块生成硬掩码（可调整参数）
        mask_obj = self.gumbel(mask_feat, 
                              gumbel_temp=self.gumbel_temp, 
                              gumbel_noise=self.gumbel_noise)
        hard_mask = mask_obj.hard
        
        # 3. 激活率监督损失计算
        loss_ars = torch.tensor(0.0, device=x.device)
        if self.training and mask_gt is not None:
            soft_mask = torch.sigmoid(mask_feat)  # 软掩码用于计算激活率
            ar_pred = torch.mean(soft_mask, dim=[2,3])  # (B,1) 激活率
            ar_gt = torch.mean(mask_gt, dim=[2,3])      # (B,1) GT激活率
            loss_ars = self.ars_loss_weight * self.ars_loss(ar_pred, ar_gt)
        
        return hard_mask, loss_ars

# ===================== 核心模块：CE-GN =====================
class CEGN(nn.Module):
    """上下文增强组归一化（Context-Enhanced Group Normalization）
    可配置参数：
    - num_channels: 输入通道数
    - num_groups: 分组数
    - eps: 数值稳定性参数
    - global_fusion_kernel: 全局融合卷积核大小
    """
    def __init__(self, 
                 num_channels, 
                 num_groups=32, 
                 eps=1e-5,
                 global_fusion_kernel=1):
        super().__init__()
        # 可修改参数
        self.num_groups = num_groups    # 可修改：GN分组数
        self.eps = eps                  # 可修改：GNepsilon
        
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)
        # 全局上下文融合（可修改卷积核大小）
        self.global_fusion = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 
                      kernel_size=global_fusion_kernel, 
                      padding=global_fusion_kernel//2,
                      bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, global_feat):
        """
        Args:
            x: 稀疏卷积输出 (B,C,H,W)
            global_feat: 全局上下文特征 G_i (B,C,1,1)
        Returns:
            out: 上下文增强特征 (B,C,H,W)
        """
        # 1. 基础 GroupNorm
        x_gn = self.gn(x)
        # 2. 全局特征广播 + 融合
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_fused = self.global_fusion(x_gn + global_feat)
        return x_fused

# ===================== 独立模块：SparseConv（单独抽离的稀疏卷积） =====================
class SparseConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3,
                 dilation=1,
                 padding=None,
                 bias=False,
                 stride=1):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size // 2)
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
    
    def forward(self, x, hard_mask):
        mask_broadcast = hard_mask.expand(-1, x.shape[1], -1, -1)
        x_masked = x * mask_broadcast
        x_sparse = self.conv(x_masked)
        return x_sparse

# ===================== 核心模块：CESC（原CESCModule，类名简化） =====================
class CESC(nn.Module):
    """上下文增强稀疏卷积（Context-Enhanced Sparse Conv）
    可配置参数：
    - in_channels: 输入通道数
    - out_channels: 输出通道数
    - num_groups: CE-GN分组数
    - dilation: 膨胀率
    - kernel_size: 稀疏卷积核大小
    - residual: 是否启用残差连接
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_groups=32, 
                 dilation=1,
                 kernel_size=3,
                 residual=True):
        super().__init__()
        # 可修改参数
        self.in_channels = in_channels    # 可修改：输入通道
        self.out_channels = out_channels  # 可修改：输出通道
        self.residual = residual          # 可修改：是否启用残差
        
        # 1. 逐点卷积生成全局上下文特征 G_i（可修改）
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 2. 使用独立的SparseConv模块
        self.sparse_conv = SparseConv(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # 3. CE-GN 模块（可传递参数）
        self.ce_gn = CEGN(out_channels, num_groups=num_groups)
        
        # 残差连接（可配置是否启用）
        if self.residual:
            self.residual_layer = nn.Identity() if in_channels == out_channels else \
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x, hard_mask, global_feat=None):
        """
        Args:
            x: 输入特征 (B,C,H,W)
            hard_mask: AMM 生成的二值掩码 (B,1,H,W)
            global_feat: 全局特征 G_i（None 时自动生成）
        Returns:
            out: 上下文增强稀疏卷积输出
        """
        # 1. 生成全局上下文特征 G_i
        if global_feat is None:
            global_feat = self.point_conv(x)
            global_feat = F.adaptive_avg_pool2d(global_feat, 1)  # (B,C,1,1)
        
        # 2. 稀疏卷积：调用独立的SparseConv模块
        x_sparse = self.sparse_conv(x, hard_mask)
        
        # 3. CE-GN 增强
        x_cegn = self.ce_gn(x_sparse, global_feat)
        
        # 4. 残差连接
        out = x_cegn + self.residual_layer(x)
        return out

# ===================== 核心：CEASCHead（基于 FCOS） =====================
class CEASCHead(nn.Module):
    """CEASC 检测头（无 MMDetection 依赖，基于 FCOS 底座）
    可配置核心参数：
    - num_classes: 类别数
    - in_channels: 输入通道数
    - feat_channels: 特征通道数
    - num_cls_convs/num_reg_convs: 分类/回归分支卷积层数
    - strides: 多尺度步长
    - center_sampling_radius: 中心采样半径
    - amm_cfg: AMM模块配置字典
    - cesc_cfg: CESC模块配置字典
    - loss_weights: 各损失权重
    - score_thr/nms_thr: 推理阈值
    """
    def __init__(self, 
                 num_classes=80,
                 in_channels=256,
                 feat_channels=256,
                 num_cls_convs=4,
                 num_reg_convs=4,
                 strides=[8,16,32,64,128],
                 center_sampling_radius=1.5,
                 # AMM 配置（可深度自定义）
                 amm_cfg=dict(
                     feat_channels=256,
                     gumbel_eps=1e-8,
                     gumbel_temp=1.0,
                     gumbel_noise=True,
                     threshold=0.5,
                     train_threshold=0.5,
                     ars_loss_weight=1.0
                 ),
                 # CESC 配置（可深度自定义）
                 cesc_cfg=dict(
                     num_groups=32,
                     dilation=1,
                     kernel_size=3,
                     residual=True
                 ),
                 # 损失权重（可修改）
                 loss_weights=dict(
                     cls_loss=1.0,
                     reg_loss=1.0,
                     centerness_loss=1.0,
                     amm_loss=0.1
                 ),
                 # 推理参数（可修改）
                 score_thr=0.3,
                 nms_thr=0.5):
        super().__init__()
        # 基础配置（可修改）
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = strides
        self.center_sampling_radius = center_sampling_radius
        self.loss_weights = loss_weights
        self.score_thr = score_thr  # 可修改：推理得分阈值
        self.nms_thr = nms_thr      # 可修改：NMS阈值
        
        # 1. 初始化 AMM 模块（分类/回归分支各一个）
        self.amm_cls = AMM(in_channels,** amm_cfg)
        self.amm_reg = AMM(in_channels, **amm_cfg)
        
        # 2. 初始化 CESC 模块（分类/回归分支第一层卷积）
        self.cesc_cls = CESC(in_channels, feat_channels,** cesc_cfg)
        self.cesc_reg = CESC(in_channels, feat_channels, **cesc_cfg)
        
        # 3. 分类分支剩余卷积（可修改层数/卷积参数）
        self.cls_convs = nn.ModuleList()
        for i in range(1, num_cls_convs):
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, feat_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # 分类输出层
        self.cls_out = nn.Conv2d(feat_channels, num_classes, kernel_size=3, padding=1)
        
        # 4. 回归分支剩余卷积（可修改层数/卷积参数）
        self.reg_convs = nn.ModuleList()
        for i in range(1, num_reg_convs):
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, feat_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # 回归输出层（预测 l/t/r/b）
        self.reg_out = nn.Conv2d(feat_channels, 4, kernel_size=3, padding=1)
        # 中心度输出层
        self.centerness_out = nn.Conv2d(feat_channels, 1, kernel_size=3, padding=1)
        
        # 5. 损失函数（可替换为Focal Loss等）
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.centerness_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化卷积层权重（可修改初始化方式）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)  # 可修改：初始化均值/方差
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_single(self, x, stride, mask_gt=None):
        """单尺度特征前向传播"""
        B, C, H, W = x.shape
        
        # 1. AMM 生成掩码
        hard_mask_cls, loss_amm_cls = self.amm_cls(x, mask_gt)
        hard_mask_reg, loss_amm_reg = self.amm_reg(x, mask_gt)
        loss_amm = (loss_amm_cls + loss_amm_reg) / 2
        
        # 2. 全局上下文特征（共享）
        global_feat = self.cesc_cls.point_conv(x)
        global_feat = F.adaptive_avg_pool2d(global_feat, 1)
        
        # 3. 分类分支
        cls_feat = self.cesc_cls(x, hard_mask_cls, global_feat)
        for conv in self.cls_convs:
            cls_feat = conv(cls_feat)
        cls_score = self.cls_out(cls_feat)  # (B, num_classes, H, W)
        
        # 4. 回归分支
        reg_feat = self.cesc_reg(x, hard_mask_reg, global_feat)
        for conv in self.reg_convs:
            reg_feat = conv(reg_feat)
        bbox_pred = self.reg_out(reg_feat)  # (B,4,H,W) → l/t/r/b
        centerness = self.centerness_out(reg_feat)  # (B,1,H,W)
        
        # 5. 回归值尺度缩放（FCOS 核心）
        bbox_pred = bbox_pred * stride
        
        return cls_score, bbox_pred, centerness, loss_amm

    def forward(self, feats, mask_gts=None):
        """多尺度特征前向传播"""
        assert len(feats) == len(self.strides), "特征数与步长数不匹配"
        cls_scores, bbox_preds, centernesses, loss_amms = [], [], [], []
        
        for i, (x, stride) in enumerate(zip(feats, self.strides)):
            mask_gt = mask_gts[i] if (mask_gts is not None and i < len(mask_gts)) else None
            cls_score, bbox_pred, centerness, loss_amm = self.forward_single(x, stride, mask_gt)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
            loss_amms.append(loss_amm)
        
        # 训练阶段返回损失，推理阶段返回预测
        if self.training:
            return cls_scores, bbox_preds, centernesses, torch.stack(loss_amms).mean()
        else:
            return cls_scores, bbox_preds, centernesses

    def compute_loss(self, cls_scores, bbox_preds, centernesses, 
                     gt_bboxes, gt_labels, mask_gts=None, img_size=(640, 640)):
        """计算总损失（分类+回归+中心度+AMM）"""
        # 1. 生成锚点（每个特征图像素对应原图坐标）
        anchors = generate_anchors_strides(cls_scores, self.strides, img_size=img_size)
        
        # 2. 分配正负样本 + 生成目标（简化版 FCOS 目标分配）
        cls_targets, reg_targets, centerness_targets, fg_masks = [], [], [], []
        for i, (cls_score, bbox_pred, stride) in enumerate(zip(cls_scores, bbox_preds, self.strides)):
            B, _, H, W = cls_score.shape
            anchor = anchors[i].to(cls_score.device)  # (H*W, 2)
            
            # 遍历每张图
            cls_target = torch.zeros(B, self.num_classes, H*W, device=cls_score.device)
            reg_target = torch.zeros(B, 4, H*W, device=cls_score.device)
            centerness_target = torch.zeros(B, 1, H*W, device=cls_score.device)
            fg_mask = torch.zeros(B, 1, H*W, device=cls_score.device, dtype=bool)
            
            for b in range(B):
                gt_bbox = gt_bboxes[b]  # (N,4) [x1,y1,x2,y2]
                gt_label = gt_labels[b]  # (N,)
                
                if len(gt_bbox) == 0:
                    continue
                
                # 计算锚点到 GT 边界的距离（l/t/r/b）
                gt_xyxy = gt_bbox.unsqueeze(1)  # (N,1,4)
                anchor_xy = anchor.unsqueeze(0)  # (1,H*W,2)
                l = anchor_xy[...,0] - gt_xyxy[...,0]
                t = anchor_xy[...,1] - gt_xyxy[...,1]
                r = gt_xyxy[...,2] - anchor_xy[...,0]
                b_dist = gt_xyxy[...,3] - anchor_xy[...,1]
                dists = torch.stack([l,t,r,b_dist], dim=-1)  # (N,H*W,4)
                
                # 筛选有效锚点（距离>0 + 中心采样）
                valid = (dists > 0).all(-1)
                # 中心采样：锚点需在 GT 中心区域内
                gt_xywh = xyxy2xywh(gt_bbox)
                center_x, center_y = gt_xywh[:,0], gt_xywh[:,1]
                center_radius = self.center_sampling_radius * stride
                center_dist = torch.sqrt(
                    (anchor_xy[...,0] - center_x.unsqueeze(1))**2 + 
                    (anchor_xy[...,1] - center_y.unsqueeze(1))**2
                )
                center_valid = center_dist < center_radius
                valid = valid & center_valid
                
                # 为每个锚点分配最佳 GT
                min_dists = dists.min(-1)[0]  # (N,H*W)
                min_dists[~valid] = float('inf')
                gt_idx = min_dists.argmin(0)  # (H*W,) → long类型
                valid_idx = min_dists.min(0)[0] < float('inf')  # (H*W,) → bool类型
                valid_anchor_idx = torch.nonzero(valid_idx, as_tuple=True)[0]
                if len(valid_anchor_idx) == 0:
                    continue
                
                # 获取有效锚点对应的GT索引
                valid_gt_idx = gt_idx[valid_anchor_idx]  # (M,)
                
                # 赋值目标
                fg_mask[b, 0, valid_anchor_idx] = True
                cls_target[b, gt_label[valid_gt_idx], valid_anchor_idx] = 1.0
                reg_target_vals = dists[valid_gt_idx, valid_anchor_idx, :]
                reg_target[b, :, valid_anchor_idx] = reg_target_vals.t()
                
                # 计算中心度（添加除零保护）
                l_val, t_val, r_val, b_val = reg_target_vals.unbind(-1)
                centerness_vals = torch.sqrt(
                    (torch.min(l_val, r_val) / (torch.max(l_val, r_val) + 1e-6)) * 
                    (torch.min(t_val, b_val) / (torch.max(t_val, b_val) + 1e-6))
                )
                centerness_target[b, 0, valid_anchor_idx] = centerness_vals
            
            # 展平特征维度
            cls_targets.append(cls_target.reshape(B, self.num_classes, H, W))
            reg_targets.append(reg_target.reshape(B, 4, H, W))
            centerness_targets.append(centerness_target.reshape(B, 1, H, W))
            fg_masks.append(fg_mask.reshape(B, 1, H, W))
        
        # 3. 计算损失
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_centerness_loss = 0.0
        total_amm_loss = 0.0
        
        # AMM 损失（前向时已计算）
        _, _, _, loss_amm = self.forward(feats, mask_gts)
        total_amm_loss = loss_amm
        
        # 分类/回归/中心度损失
        num_fg = 0
        for cls_score, bbox_pred, centerness, cls_target, reg_target, centerness_target, fg_mask in zip(
            cls_scores, bbox_preds, centernesses, cls_targets, reg_targets, centerness_targets, fg_masks):
            
            B = cls_score.shape[0]
            # 分类损失（仅正样本）
            cls_loss = self.cls_loss(cls_score, cls_target)
            cls_loss = cls_loss * fg_mask.expand(-1, self.num_classes, -1, -1)
            cls_loss = cls_loss.sum() / max(B, 1)
            total_cls_loss += self.loss_weights['cls_loss'] * cls_loss
            
            # 回归损失（仅正样本）
            reg_loss = self.reg_loss(bbox_pred, reg_target)
            reg_loss = reg_loss * fg_mask.expand(-1, 4, -1, -1)
            reg_loss = reg_loss.sum() / max(B, 1)
            total_reg_loss += self.loss_weights['reg_loss'] * reg_loss
            
            # 中心度损失（仅正样本）
            centerness_loss = self.centerness_loss(centerness, centerness_target)
            centerness_loss = centerness_loss * fg_mask
            centerness_loss = centerness_loss.sum() / max(B, 1)
            total_centerness_loss += self.loss_weights['centerness_loss'] * centerness_loss
            
            # 统计正样本数
            num_fg += fg_mask.sum()
        
        # 归一化损失
        num_fg = max(num_fg, 1)
        total_loss = (
            total_cls_loss + 
            total_reg_loss + 
            total_centerness_loss + 
            self.loss_weights['amm_loss'] * total_amm_loss
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': total_cls_loss,
            'reg_loss': total_reg_loss,
            'centerness_loss': total_centerness_loss,
            'amm_loss': total_amm_loss
        }

    def inference(self, feats, img_size=(640,640), score_thr=None, nms_thr=None):
        """推理阶段：生成最终检测框
        可动态修改：score_thr/nms_thr（优先使用传入值，否则用类初始化值）
        """
        # 优先使用传入的阈值，否则用类默认值
        score_thr = score_thr if score_thr is not None else self.score_thr
        nms_thr = nms_thr if nms_thr is not None else self.nms_thr
        
        cls_scores, bbox_preds, centernesses = self.forward(feats)
        batch_boxes = []
        batch_scores = []
        batch_labels = []
        
        # 遍历多尺度特征
        for i, (cls_score, bbox_pred, centerness, stride) in enumerate(zip(
            cls_scores, bbox_preds, centernesses, self.strides)):
            
            B, _, H, W = cls_score.shape
            # 生成锚点
            anchor = generate_anchors_strides([cls_score], [stride], img_size)[0].to(cls_score.device)
            
            # 展平特征
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(B, H*W, self.num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, H*W, 4)
            centerness = centerness.permute(0, 2, 3, 1).reshape(B, H*W, 1)
            
            # 计算置信度（分类得分 * 中心度）
            cls_score_sigmoid = torch.sigmoid(cls_score)
            centerness_sigmoid = torch.sigmoid(centerness)
            scores = cls_score_sigmoid * centerness_sigmoid
            
            # 遍历每张图
            for b in range(B):
                img_boxes = []
                img_scores = []
                img_labels = []
                
                # 遍历每个类别
                for cls_idx in range(self.num_classes):
                    cls_scores_single = scores[b, :, cls_idx]
                    keep_idx = cls_scores_single > score_thr
                    if not keep_idx.any():
                        continue
                    
                    # 提取有效数据
                    valid_scores = cls_scores_single[keep_idx]
                    valid_bbox_pred = bbox_pred[b, keep_idx, :]
                    valid_anchor = anchor[keep_idx, :]
                    
                    # 转换为x1y1x2y2
                    l, t, r, b_dist = valid_bbox_pred.unbind(-1)
                    x1 = valid_anchor[:, 0] - l
                    y1 = valid_anchor[:, 1] - t
                    x2 = valid_anchor[:, 0] + r
                    y2 = valid_anchor[:, 1] + b_dist
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    
                    # 限制框在图像内
                    boxes = torch.clamp(boxes, 0, img_size[0]-1)
                    
                    # 收集结果
                    img_boxes.append(boxes)
                    img_scores.append(valid_scores)
                    img_labels.append(torch.full_like(valid_scores, cls_idx, dtype=torch.long))
                
                # 合并当前图结果
                if img_boxes:
                    img_boxes = torch.cat(img_boxes)
                    img_scores = torch.cat(img_scores)
                    img_labels = torch.cat(img_labels)
                    
                    # 类内NMS
                    keep = nms(img_boxes, img_scores, nms_thr)
                    batch_boxes.append(img_boxes[keep])
                    batch_scores.append(img_scores[keep])
                    batch_labels.append(img_labels[keep])
        
        # 合并所有批次结果
        if not batch_boxes:
            return torch.empty(0,4), torch.empty(0), torch.empty(0)
        all_boxes = torch.cat(batch_boxes)
        all_scores = torch.cat(batch_scores)
        all_labels = torch.cat(batch_labels)
        
        # 最终跨类NMS
        keep = nms(all_boxes, all_scores, nms_thr)
        return all_boxes[keep], all_scores[keep], all_labels[keep]

# ===================== 测试代码（可直接运行） =====================
if __name__ == "__main__":
    # 可修改：自定义配置参数
    custom_amm_cfg = dict(
        feat_channels=256,
        gumbel_eps=1e-8,
        gumbel_temp=0.8,       # 调整Gumbel温度
        gumbel_noise=True,
        threshold=0.5,
        train_threshold=0.5,
        ars_loss_weight=0.5    # 调整AMM损失权重
    )
    
    custom_cesc_cfg = dict(
        num_groups=16,         # 调整GN分组数
        dilation=2,            # 调整膨胀率
        kernel_size=3,
        residual=True
    )
    
    custom_loss_weights = dict(
        cls_loss=1.0,
        reg_loss=1.5,          # 增大回归损失权重
        centerness_loss=1.0,
        amm_loss=0.1
    )
    
    # 1. 创建 CEASCHead 实例（使用自定义参数）
    head = CEASCHead(
        num_classes=10,        # 可修改：数据集类别数
        in_channels=256,
        feat_channels=256,
        num_cls_convs=3,       # 可修改：分类分支卷积层数
        num_reg_convs=3,       # 可修改：回归分支卷积层数
        strides=[8,16,32],     # 可修改：多尺度步长
        center_sampling_radius=2.0,  # 可修改：中心采样半径
        amm_cfg=custom_amm_cfg,
        cesc_cfg=custom_cesc_cfg,
        loss_weights=custom_loss_weights,
        score_thr=0.2,         # 可修改：推理得分阈值
        nms_thr=0.4            # 可修改：NMS阈值
    )
    
    # 2. 模拟输入（FPN 多尺度特征）
    feats = [
        torch.randn(2, 256, 80, 80),  # stride=8
        torch.randn(2, 256, 40, 40),  # stride=16
        torch.randn(2, 256, 20, 20)   # stride=32
    ]
    
    # 3. 训练模式测试
    head.train()
    # 模拟 GT
    gt_bboxes = [
        torch.tensor([[50,50,100,100], [200,200,300,300]], dtype=torch.float32),
        torch.tensor([[100,100,150,150]], dtype=torch.float32)
    ]
    gt_labels = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2], dtype=torch.long)
    ]
    # 前向传播
    cls_scores, bbox_preds, centernesses, _ = head.forward(feats)
    # 计算损失
    losses = head.compute_loss(cls_scores, bbox_preds, centernesses, 
                              gt_bboxes, gt_labels, img_size=(640, 640))
    print("训练损失：")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # 4. 推理模式测试（可动态调整阈值）
    head.eval()
    with torch.no_grad():
        boxes, scores, labels = head.inference(
            feats, 
            img_size=(640,640),
            score_thr=0.15,  # 动态修改推理阈值
            nms_thr=0.45     # 动态修改NMS阈值
        )
    print("\n推理结果：")
    print(f"检测框数量：{len(boxes)}")
    if len(boxes) > 0:
        print(f"第一个框：{boxes[0].numpy()}")
        print(f"第一个框得分：{scores[0].item():.4f}")
        print(f"第一个框类别：{labels[0].item()}")
