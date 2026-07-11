import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
import numpy as np

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
        grid = torch.clamp(grid, 0, img_size[0]-1)  # img_size现在是可下标对象
        anchors.append(grid)
    return anchors

# ===================== 核心模块：AMM =====================
class AMM(nn.Module):
    """自适应多层掩码模块（Adaptive Multi-Layer Masking）"""
    def __init__(self, in_channels, feat_channels=256, gumbel_noise=True, threshold=0.5):
        super().__init__()
        self.gumbel_noise = gumbel_noise
        self.threshold = threshold
        
        # 3x3卷积生成软掩码
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        # 激活率监督损失（L1 Loss）
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
        # 1. 生成软掩码 S_i (B,1,H,W)
        soft_mask = self.mask_conv(x)
        soft_mask = torch.sigmoid(soft_mask)  # 归一化到 [0,1]
        
        # 2. 注入 Gumbel 噪声（仅训练阶段）
        if self.training and self.gumbel_noise:
            gumbel = torch.distributions.Gumbel(0, 1).sample(soft_mask.shape).to(x.device)
            soft_mask = soft_mask + gumbel
        
        # 3. 硬阈值生成二值掩码 H_i
        hard_mask = (soft_mask > self.threshold).float()
        
        # 4. 激活率监督损失计算
        loss_ars = torch.tensor(0.0, device=x.device)
        if self.training and mask_gt is not None:
            ar_pred = torch.mean(soft_mask, dim=[2,3])  # (B,1) 激活率
            ar_gt = torch.mean(mask_gt, dim=[2,3])      # (B,1) GT激活率
            loss_ars = self.ars_loss(ar_pred, ar_gt)
        
        return hard_mask, loss_ars

# ===================== 核心模块：CE-GN =====================
class CEGN(nn.Module):
    """上下文增强组归一化（Context-Enhanced Group Normalization）"""
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)
        # 1x1卷积融合全局上下文特征
        self.global_fusion = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
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

# ===================== 核心模块：CESC =====================
class CESC(nn.Module):
    """上下文增强稀疏卷积（Context-Enhanced Sparse Conv）"""
    def __init__(self, in_channels, out_channels, num_groups=32, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. 逐点卷积生成全局上下文特征 G_i
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 2. 3x3 稀疏卷积（掩码引导）
        self.sparse_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False
        )
        
        # 3. CE-GN 模块
        self.ce_gn = CEGN(out_channels, num_groups=num_groups)
        
        # 残差连接（通道不一致时适配）
        self.residual = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

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
        
        # 2. 稀疏卷积：仅掩码为 1 的区域参与计算
        mask_broadcast = hard_mask.expand(-1, self.in_channels, -1, -1)  # 广播到所有通道
        x_sparse = x * mask_broadcast  # 掩码过滤
        x_sparse = self.sparse_conv(x_sparse)
        
        # 3. CE-GN 增强
        x_cegn = self.ce_gn(x_sparse, global_feat)
        
        # 4. 残差连接
        out = x_cegn + self.residual(x)
        return out

# ===================== 核心：CEASCHead（基于 FCOS） =====================
class CEASCHead(nn.Module):
    """CEASC 检测头（无 MMDetection 依赖，基于 FCOS 底座）"""
    def __init__(self, 
                 num_classes=80,
                 in_channels=256,
                 feat_channels=256,
                 num_cls_convs=4,
                 num_reg_convs=4,
                 strides=[8,16,32,64,128],  # FCOS 多尺度步长
                 center_sampling_radius=1.5,
                 # AMM 配置
                 amm_cfg=dict(feat_channels=256, gumbel_noise=True, threshold=0.5),
                 # CESC 配置
                 cesc_cfg=dict(num_groups=32, dilation=1)):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = strides
        self.center_sampling_radius = center_sampling_radius
        
        # 1. 初始化 AMM 模块（分类/回归分支各一个）
        self.amm_cls = AMM(in_channels, **amm_cfg)
        self.amm_reg = AMM(in_channels, **amm_cfg)
        
        # 2. 初始化 CESC 模块（分类/回归分支第一层卷积）
        self.cesc_cls = CESC(in_channels, feat_channels, **cesc_cfg)
        self.cesc_reg = CESC(in_channels, feat_channels, **cesc_cfg)
        
        # 3. 分类分支剩余卷积（普通卷积）
        self.cls_convs = nn.ModuleList()
        for i in range(1, num_cls_convs):  # 第一层是 CESC，从第2层开始
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, feat_channels),
                    nn.ReLU(inplace=True)
                )
            )
        # 分类输出层
        self.cls_out = nn.Conv2d(feat_channels, num_classes, kernel_size=3, padding=1)
        
        # 4. 回归分支剩余卷积（普通卷积）
        self.reg_convs = nn.ModuleList()
        for i in range(1, num_reg_convs):  # 第一层是 CESC，从第2层开始
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
        
        # 5. 损失函数
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')  # Focal Loss 可替换
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.centerness_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
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
                     gt_bboxes, gt_labels, mask_gts=None, img_size=(640, 640)):  # 新增img_size参数
        """计算总损失（分类+回归+中心度+AMM）"""
        # 1. 生成锚点（每个特征图像素对应原图坐标）
        anchors = generate_anchors_strides(cls_scores, self.strides, img_size=img_size)  # 传入正确的img_size
        
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
                b_dist = gt_xyxy[...,3] - anchor_xy[...,1]  # 避免与循环变量b冲突
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
                # 转换为long类型的有效锚点索引（避免bool索引的复合问题）
                valid_anchor_idx = torch.nonzero(valid_idx, as_tuple=True)[0]  # (M,) 有效锚点的位置
                if len(valid_anchor_idx) == 0:
                    continue  # 无有效锚点，跳过
                
                # 获取有效锚点对应的GT索引
                valid_gt_idx = gt_idx[valid_anchor_idx]  # (M,)
                
                # 赋值目标（使用long类型索引）
                fg_mask[b, 0, valid_anchor_idx] = True
                cls_target[b, gt_label[valid_gt_idx], valid_anchor_idx] = 1.0
                # 正确提取每个有效锚点对应的距离值
                reg_target_vals = dists[valid_gt_idx, valid_anchor_idx, :]  # (M,4)
                reg_target[b, :, valid_anchor_idx] = reg_target_vals.t()  # (4,M) → 对应reg_target的(4,H*W)维度
                
                # 计算中心度（添加除零保护）
                l_val, t_val, r_val, b_val = reg_target_vals.unbind(-1)  # 各(M,)
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
            total_cls_loss += cls_loss
            
            # 回归损失（仅正样本）
            reg_loss = self.reg_loss(bbox_pred, reg_target)
            reg_loss = reg_loss * fg_mask.expand(-1, 4, -1, -1)
            reg_loss = reg_loss.sum() / max(B, 1)
            total_reg_loss += reg_loss
            
            # 中心度损失（仅正样本）
            centerness_loss = self.centerness_loss(centerness, centerness_target)
            centerness_loss = centerness_loss * fg_mask
            centerness_loss = centerness_loss.sum() / max(B, 1)
            total_centerness_loss += centerness_loss
            
            # 统计正样本数
            num_fg += fg_mask.sum()
        
        # 归一化损失
        num_fg = max(num_fg, 1)
        total_loss = (
            total_cls_loss + 
            total_reg_loss + 
            total_centerness_loss + 
            0.1 * total_amm_loss  # AMM 损失权重
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': total_cls_loss,
            'reg_loss': total_reg_loss,
            'centerness_loss': total_centerness_loss,
            'amm_loss': total_amm_loss
        }

    def inference(self, feats, img_size=(640,640), score_thr=0.3, nms_thr=0.5):
        """推理阶段：生成最终检测框"""
        cls_scores, bbox_preds, centernesses = self.forward(feats)
        batch_boxes = []
        batch_scores = []
        batch_labels = []
        
        # 遍历多尺度特征
        for i, (cls_score, bbox_pred, centerness, stride) in enumerate(zip(
            cls_scores, bbox_preds, centernesses, self.strides)):
            
            B, _, H, W = cls_score.shape
            # 生成锚点
            anchor = generate_anchors_strides([cls_score], [stride], img_size)[0].to(cls_score.device)  # (H*W, 2)
            
            # 展平特征：(B, C, H, W) → (B, H*W, C)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(B, H*W, self.num_classes)  # (B, H*W, num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, H*W, 4)  # (B, H*W, 4)
            centerness = centerness.permute(0, 2, 3, 1).reshape(B, H*W, 1)  # (B, H*W, 1)
            
            # 计算置信度（分类得分 * 中心度）
            cls_score_sigmoid = torch.sigmoid(cls_score)
            centerness_sigmoid = torch.sigmoid(centerness)
            scores = cls_score_sigmoid * centerness_sigmoid  # (B, H*W, num_classes)
            
            # 遍历每张图
            for b in range(B):
                img_boxes = []
                img_scores = []
                img_labels = []
                
                # 遍历每个类别，单独筛选
                for cls_idx in range(self.num_classes):
                    # 提取当前类别的得分
                    cls_scores_single = scores[b, :, cls_idx]  # (H*W,)
                    # 筛选高分锚点
                    keep_idx = cls_scores_single > score_thr  # (H*W,)
                    if not keep_idx.any():
                        continue
                    
                    # 提取有效数据（保证boxes和scores维度匹配）
                    valid_scores = cls_scores_single[keep_idx]  # (M,)
                    valid_bbox_pred = bbox_pred[b, keep_idx, :]  # (M, 4)
                    valid_anchor = anchor[keep_idx, :]  # (M, 2)
                    
                    # 从 l/t/r/b 转换为 x1/y1/x2/y2
                    l, t, r, b_dist = valid_bbox_pred.unbind(-1)
                    x1 = valid_anchor[:, 0] - l
                    y1 = valid_anchor[:, 1] - t
                    x2 = valid_anchor[:, 0] + r
                    y2 = valid_anchor[:, 1] + b_dist
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (M, 4)
                    
                    # 限制框在图像内
                    boxes = torch.clamp(boxes, 0, img_size[0]-1)
                    
                    # 收集当前类别的结果
                    img_boxes.append(boxes)
                    img_scores.append(valid_scores)
                    img_labels.append(torch.full_like(valid_scores, cls_idx, dtype=torch.long))
                
                # 合并当前图所有类别的结果
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
        
        # 最终跨类NMS（可选，根据需求调整）
        keep = nms(all_boxes, all_scores, nms_thr)
        return all_boxes[keep], all_scores[keep], all_labels[keep]

# ===================== 测试代码（可直接运行） =====================
if __name__ == "__main__":
    # 1. 创建 CEASCHead 实例
    head = CEASCHead(
        num_classes=10,  # VisDrone 数据集 10 类
        in_channels=256,
        feat_channels=256,
        strides=[8,16,32]  # 简化为 3 个尺度
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
    # 先调用forward获取cls_scores、bbox_preds、centernesses
    cls_scores, bbox_preds, centernesses, _ = head.forward(feats)
    # 计算损失（传入img_size参数）
    losses = head.compute_loss(cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_size=(640, 640))
    print("训练损失：")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # 4. 推理模式测试
    head.eval()
    with torch.no_grad():
        boxes, scores, labels = head.inference(feats, img_size=(640,640))
    print("\n推理结果：")
    print(f"检测框数量：{len(boxes)}")
    if len(boxes) > 0:
        print(f"第一个框：{boxes[0].numpy()}")
        print(f"第一个框得分：{scores[0].item():.4f}")
        print(f"第一个框类别：{labels[0].item()}")
