import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
from typing import Optional, List, Tuple, Union
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

# ===================== 引入稀疏卷积相关模块 =====================
# 从 sparseconv_utils.py 引入 Mask 和 Gumbel 类
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
            # this code only support batchsize=1 while inference, 
            # nonzero calculation need to be updated in the future
            self.nonzero_hard = torch.nonzero(hard[0][0], as_tuple=True)

class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, eps=1e-8):  # 可修改参数：eps（数值稳定性）
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):  # 可修改参数：gumbel_temp（温度系数）、gumbel_noise（是否加噪声）
        if not self.training:  # no Gumbel noise during inference
            hard = (x >= 0).float() 
            ans = Mask(hard)
            return ans

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        ans = Mask(hard)
        return ans

# 从 sparse_conv_net.py 引入 sparse_gn 和 SparseConv2d 类
def sparse_gn(x, gn, pw_x):
    """
    稀疏组归一化函数
    Args:
        x: 输入特征 (N, C, H, W)
        gn: GroupNorm层
        pw_x: 逐点统计特征，当为None时使用输入x
    Returns:
        归一化后的特征
    """
    N, C, H, W = x.size()
    G = gn.num_groups
    
    # 如果pw_x为None，则使用输入x作为统计特征
    if pw_x is None:
        pw_x = x
    
    x = x.view(N, G, -1)
    pw_x = pw_x.view(N, G, -1)
    mean_part = pw_x.mean(-1, keepdim=True)
    var_part = pw_x.var(-1, keepdim=True)

    x_part = (x - mean_part) / (var_part + gn.eps).sqrt()
    x_part = x_part.view(N, C, H, W)
    x_part = x_part * gn.weight.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3) + gn.bias.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)

    return x_part

class Sparse_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hard, weights, bias, stride, padding, isbias, base, gn=None, pw=None, nonzero_hard=None):
        groups = -999
        gnweight = bias
        gnbias = bias
        eps = 1e-5
        if pw == None:
            pw_mean = bias
            pw_rstd = bias
        else:
            pw_mean = pw[0].type_as(input)
            pw_rstd = pw[1].type_as(input)
        if gn != None:
            groups = gn.num_groups
            gnweight = gn.weight.type_as(input)
            gnbias = gn.bias.type_as(input)
            eps = gn.eps
        
        if str(input.device) == 'cpu':
            assert not str(input.device) == 'cpu', 'we do not support CPU inference, you can try codes in sparse_conv_cpu folder, but we cannot ensure the correctness'
            #import my_sparse_conv_cpu
            #output = my_sparse_conv_cpu.forward(input, hard.type_as(input), weights, bias, stride[0], padding[0], isbias, base, groups, gnweight, gnbias, pw_mean, pw_rstd, eps, nonzero_hard[0], nonzero_hard[1])[0]
        else:
            try:
                import sparse_conv
                output = sparse_conv.forward(input, hard, weights, bias, stride[0], padding[0], isbias, base, groups, gnweight, gnbias, pw_mean, pw_rstd, eps, nonzero_hard[0], nonzero_hard[1])[0]
            except ImportError:
                raise ImportError("请确保 sparse_conv 扩展已正确编译")
        
        variables = [input, hard, weights, bias, None, None, None, None, None, None, None]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "Warning: using sparse conv2d's backward, it should not happen as we do not provide its backward function"
        return None, None, None, None, None, None, None, None, None, None, None

class SparseConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        base=0,  # 可修改参数：base（稀疏区域的基准值）
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        self.isbias = bias
        self.base = base
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        
        super(SparseConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def _slow_forward(self, input, hard, pw, weight, bias, gn):
        hard = hard.hard
        x = F.conv2d(input, weight, bias, self.stride,
                    self.padding, self.dilation, self.groups)   
        if gn != None:
            with torch.no_grad():
                x_total = gn(x)
                x_total = hard * x_total

            assert self.base == 0
            x_part = sparse_gn(x, gn, pw)
            x_part = hard * x_part
            if not self.training:
                return x_part
            return x_part, F.mse_loss(x_part, x_total.detach())

        return hard * x + self.base * (1.0 - hard)

    def _fast_forward(self, input, hard, pw, weight, bias, isbias, base, gn):
        if hard.n_keep == 0:
            one_ = torch.ones((input.shape[0], self.out_channels, input.shape[2], input.shape[3]), dtype=input.dtype, device=input.device)
            return one_ * self.base
        
        nonzero_hard = hard.nonzero_hard
        hard = hard.hard
        
        if not isbias:
            bias_ = torch.ones((0)).to(input.device)
            x = Sparse_conv2d.apply(input, hard, weight, bias_, self.stride,
                        self.padding, isbias, base, gn, pw, nonzero_hard)
        else:
            x = Sparse_conv2d.apply(input, hard, weight, bias, self.stride,
                        self.padding, isbias, base, gn, pw, nonzero_hard) 

        if gn != None:
            x_part = hard * x
            return x_part.type_as(input)
        return (hard * x + self.base * (1.0 - hard)).type_as(input)

    def forward(self, input, hard, pw=None, gn=None):
        if self.training:
            return self._slow_forward(input, hard, pw, self.weight, self.bias, gn)
        else:
            fast_ans = self._fast_forward(input, hard, pw, self.weight, self.bias, self.isbias, self.base, gn)
            return fast_ans

# ===================== 核心模块：AMM =====================
class AMM(nn.Module):
    """自适应多层掩码模块（Adaptive Multi-Layer Masking）"""
    def __init__(self, in_channels, feat_channels=256, gumbel_noise=True, threshold=0.5, target_ar=0.3):
        super().__init__()
        self.gumbel_noise = gumbel_noise
        self.threshold = threshold
        self.target_ar = target_ar
        self.gumbel_module = Gumbel(eps=1e-8)
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.ars_loss = nn.L1Loss(reduction='mean')

    def forward(self, x, mask_gt=None):
        soft_mask = self.mask_conv(x)
        hard_mask = self.gumbel_module(
            soft_mask,
            gumbel_temp=1.0,
            gumbel_noise=self.gumbel_noise and self.training
        )
        loss_ars = torch.tensor(0.0, device=x.device)
        if self.training:
            ar_pred = torch.mean(torch.sigmoid(soft_mask), dim=[2,3])
            loss_ars = self.ars_loss(ar_pred, torch.full_like(ar_pred, self.target_ar))
        return hard_mask, loss_ars

# ===================== 核心模块：CE-GN =====================
class CEGN(nn.Module):
    """上下文增强组归一化（Context-Enhanced Group Normalization）"""
    def __init__(self, num_channels, num_groups=32, eps=1e-5):  # 可修改参数：num_groups（分组数）、eps（数值稳定性）
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
    def __init__(self, in_channels, out_channels, num_groups=32, dilation=1, base=0.0):
        # 添加base参数支持，用于稀疏卷积的基准值配置
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base = base  # 存储base参数供后续使用
        
        # 1. 逐点卷积生成全局上下文特征 G_i
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # 2. 3x3 稀疏卷积（替换原普通卷积，使用SparseConv2d）
        self.sparse_conv = SparseConv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False,
            base=self.base  # 使用传入的base参数
        )
        
        # 3. CE-GN 模块
        self.ce_gn = CEGN(out_channels, num_groups=num_groups)
        
        # 残差连接（通道不一致时适配）
        self.residual = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, hard_mask, global_feat=None, pw=None):
        if global_feat is None:
            global_feat = self.point_conv(x)
            global_feat = F.adaptive_avg_pool2d(global_feat, 1)

        sparse_result = self.sparse_conv(x, hard_mask, pw=pw, gn=self.ce_gn.gn)
        mse_loss = torch.tensor(0.0, device=x.device)
        if isinstance(sparse_result, tuple):
            x_sparse, mse_loss = sparse_result
        else:
            x_sparse = sparse_result

        x_cegn = self.ce_gn(x_sparse, global_feat)
        out = x_cegn + self.residual(x)
        return out, mse_loss

# ===================== 核心：CEASCHead（基于 FCOS） =====================
class CEASC(nn.Module):
    """CEASC 检测头（无 MMDetection 依赖，基于 FCOS 底座）"""
    def __init__(self, 
                 num_classes=80,      # 第一个参数：类别数 (nc)
                 in_channels=256,     # 第二个参数：输入通道数 (512) - 这也输出通道数
                 feat_channels=256,
                 num_cls_convs=4,
                 num_reg_convs=4,
                 strides=[8,16,32],  # 修改为3个尺度以适应3个输入
                 center_sampling_radius=1.5,  # 可修改参数：中心采样半径
                 # AMM 配置
                 amm_cfg=dict(feat_channels=256, gumbel_noise=True, threshold=0.5),
                 # CESC 配置
                 cesc_cfg=dict(num_groups=32, dilation=1, base=0.0)):  # 新增base参数
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = strides
        # Ultralytics 期望检测头暴露若干属性（stride, nl 等）。
        # 为兼容 Ultralytics loss/推理逻辑，补齐这些属性。
        try:
            self.stride = torch.tensor(self.strides, dtype=torch.float32)
        except Exception:
            # fallback to a tensor of floats
            self.stride = torch.tensor([float(x) for x in self.strides], dtype=torch.float32)
        self.nl = len(self.strides)
        # 标记是否为端到端检测头（默认否）
        self.end2end = False
        # 方便外部通过 head.nc 访问类别数
        self.nc = self.num_classes
        # 与 YOLO v8/v11 损失兼容的回归量化参数（默认为16）
        # ultralytics 中的 loss 会访问 `m.reg_max`，因此需要提供该属性
        self.reg_max = 16
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
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')  # 可替换为Focal Loss
        self.reg_loss = nn.SmoothL1Loss(reduction='none')       # 可修改参数：beta（SmoothL1的平滑系数）
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
        cls_feat, mse_cls = self.cesc_cls(x, hard_mask_cls, global_feat)
        for conv in self.cls_convs:
            cls_feat = conv(cls_feat)
        cls_score = self.cls_out(cls_feat)

        # 4. 回归分支
        reg_feat, mse_reg = self.cesc_reg(x, hard_mask_reg, global_feat)
        for conv in self.reg_convs:
            reg_feat = conv(reg_feat)
        bbox_pred = self.reg_out(reg_feat)
        centerness = self.centerness_out(reg_feat)

        # 5. 回归值尺度缩放
        bbox_pred = bbox_pred * stride

        total_aux_loss = loss_amm + mse_cls + mse_reg
        return cls_score, bbox_pred, centerness, total_aux_loss

    def forward(self, feats, mask_gts=None):
        """多尺度特征前向传播 - 接收3个张量的元组输入"""
        if isinstance(feats, (tuple, list)):
            assert len(feats) == 3, f"Expected 3 feature maps, got {len(feats)}"
            feats = list(feats)
        else:
            feats = [feats] * 3

        assert len(feats) == len(self.strides), f"特征数 {len(feats)} 与步长数 {len(self.strides)} 不匹配"
        cls_scores, bbox_preds, centernesses, total_aux = [], [], [], []

        for i, (x, stride) in enumerate(zip(feats, self.strides)):
            mask_gt = mask_gts[i] if (mask_gts is not None and i < len(mask_gts)) else None
            cls_score, bbox_pred, centerness, aux = self.forward_single(x, stride, mask_gt)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
            total_aux.append(aux)

        aux_loss_mean = torch.stack(total_aux).mean()

        if self.training:
            return cls_scores, bbox_preds, centernesses, aux_loss_mean
        else:
            return cls_scores, bbox_preds, centernesses

    def compute_loss(self, cls_scores, bbox_preds, centernesses, aux_loss, gt_bboxes, gt_labels, img_size):
        """FCOS loss 计算
        Args:
            cls_scores: list of [B, num_classes, Hi, Wi]
            bbox_preds: list of [B, 4, Hi, Wi] — (l, t, r, b) 距离，已乘 stride
            centernesses: list of [B, 1, Hi, Wi]
            aux_loss: 辅助损失 (AMM + CESC mse)
            gt_bboxes: list[Tensor] — 每张图的 GT box [Mi, 4] xywh 归一化格式
            gt_labels: list[Tensor] — 每张图的 GT 类别 [Mi]
            img_size: (H, W) 图像尺寸
        Returns:
            dict: {"loss_cls": ..., "loss_reg": ..., "loss_ctr": ..., "loss_aux": ...}
        """
        B = cls_scores[0].shape[0]
        INF = 1e8
        center_radius = self.center_sampling_radius

        loss_cls_total = torch.tensor(0.0, device=cls_scores[0].device)
        loss_reg_total = torch.tensor(0.0, device=cls_scores[0].device)
        loss_ctr_total = torch.tensor(0.0, device=cls_scores[0].device)
        num_pos_total = 0

        for level_idx, (cls_score, bbox_pred, centerness, stride) in enumerate(
            zip(cls_scores, bbox_preds, centernesses, self.strides)):
            _, _, H, W = cls_score.shape
            # 生成锚点网格
            xs = torch.arange(0, W * stride, stride, dtype=torch.float32, device=cls_score.device) + stride / 2
            ys = torch.arange(0, H * stride, stride, dtype=torch.float32, device=cls_score.device) + stride / 2
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            anchor_pts = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (N_anchors, 2)

            cls_score_flat = cls_score.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            centerness_flat = centerness.permute(0, 2, 3, 1).reshape(B, H * W, 1)

            for b in range(B):
                gt_box = gt_bboxes[b]  # [M, 4] xywh normalized
                gt_cls = gt_labels[b]  # [M]
                if gt_box.numel() == 0:
                    continue
                M = gt_box.shape[0]
                # convert normalized xywh → absolute xyxy
                gw, gh = img_size[1], img_size[0]
                gt_xyxy = xywh2xyxy(gt_box)
                gt_xyxy[:, [0, 2]] *= gw
                gt_xyxy[:, [1, 3]] *= gh

                # 每个 anchor 与所有 GT 的 ltrb 距离
                anchors = anchor_pts  # (N, 2)
                l = anchors[:, 0:1] - gt_xyxy[:, 0][None, :]  # (N, M)
                t = anchors[:, 1:2] - gt_xyxy[:, 1][None, :]
                r = gt_xyxy[:, 2][None, :] - anchors[:, 0:1]
                b_dist = gt_xyxy[:, 3][None, :] - anchors[:, 1:2]
                ltrb = torch.stack([l, t, r, b_dist], dim=-1)  # (N, M, 4)

                # 正样本：anchor 在 GT 框内
                in_box = (ltrb.min(dim=-1)[0] > 0)  # (N, M)
                # Center-sampling: anchor 在 GT 中心区域内
                gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2  # (M,)
                gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
                c_l = anchors[:, 0:1] - (gt_cx[None, :] - center_radius * stride)
                c_t = anchors[:, 1:2] - (gt_cy[None, :] - center_radius * stride)
                c_r = (gt_cx[None, :] + center_radius * stride) - anchors[:, 0:1]
                c_b = (gt_cy[None, :] + center_radius * stride) - anchors[:, 1:2]
                in_center = (torch.stack([c_l, c_t, c_r, c_b], dim=-1).min(dim=-1)[0] > 0)  # (N, M)

                positive = in_box & in_center  # (N, M)

                # 每个 anchor 最多匹配一个 GT（面积最小的）
                areas = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])  # (M,)
                areas_expand = areas[None, :].expand(H * W, M).clone()
                areas_expand[~positive] = INF
                min_area, matched_gt = areas_expand.min(dim=1)  # (N,)

                pos_mask = min_area < INF  # (N,)
                num_pos = pos_mask.sum().item()
                if num_pos == 0:
                    continue
                num_pos_total += num_pos

                pos_idx = pos_mask.nonzero(as_tuple=True)[0]  # (num_pos,)
                matched = matched_gt[pos_idx]  # (num_pos,)

                # 分类 target：one-hot
                cls_target = torch.zeros(num_pos, self.num_classes, device=cls_score.device)
                cls_target[torch.arange(num_pos), gt_cls[matched]] = 1
                cls_pred = cls_score_flat[b][pos_idx]  # (num_pos, num_classes)
                loss_cls_total += self.cls_loss(cls_pred, cls_target).sum() / num_pos

                # 回归 target：l*, t*, r*, b*
                reg_target = ltrb[pos_idx, matched, :]  # (num_pos, 4)
                reg_pred = bbox_pred_flat[b][pos_idx]  # (num_pos, 4)
                loss_reg_total += self.reg_loss(reg_pred, reg_target).sum() / num_pos

                # 中心度 target
                lt = reg_target[:, 0:1]
                tt = reg_target[:, 1:2]
                rt = reg_target[:, 2:3]
                bt = reg_target[:, 3:4]
                ctr_target = torch.sqrt(
                    (lt.min(rt) / lt.max(rt).clamp(min=1e-6)) * (tt.min(bt) / tt.max(bt).clamp(min=1e-6))
                )
                ctr_pred = centerness_flat[b][pos_idx]
                loss_ctr_total += self.centerness_loss(ctr_pred, ctr_target).sum() / num_pos

        if num_pos_total > 0:
            loss_cls_total = loss_cls_total / len(self.strides)
            loss_reg_total = loss_reg_total / len(self.strides)
            loss_ctr_total = loss_ctr_total / len(self.strides)
        else:
            # 无正样本时只计算分类 loss 的负样本部分
            for cls_score in cls_scores:
                B, _, H, W = cls_score.shape
                cls_flat = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                loss_cls_total += self.cls_loss(cls_flat, torch.zeros_like(cls_flat)).mean()
            loss_cls_total = loss_cls_total / len(self.strides)

        return {
            "loss_cls": loss_cls_total,
            "loss_reg": loss_reg_total,
            "loss_ctr": loss_ctr_total,
            "loss_aux": aux_loss,
        }

    def inference_batch(self, cls_scores, bbox_preds, centernesses, img_shape, score_thr=0.3, nms_thr=0.5):
        """批量推理：返回 [B, N, 6] YOLO 兼容格式。

        Args:
            cls_scores, bbox_preds, centernesses: forward 输出的 3 个 list
            img_shape: (H, W) 输入图像尺寸
        Returns:
            torch.Tensor [B, N, 6] — [x1, y1, x2, y2, conf, cls]
        """
        B = cls_scores[0].shape[0]
        batch_results = []

        for b in range(B):
            img_boxes = []
            img_scores = []
            img_labels = []

            for level_idx, (cls_score, bbox_pred, centerness, stride) in enumerate(
                zip(cls_scores, bbox_preds, centernesses, self.strides)):
                _, _, H, W = cls_score.shape
                anchor = generate_anchors_strides([cls_score], [stride], img_shape)[0].to(cls_score.device)

                cls_flat = cls_score[b].permute(1, 2, 0).reshape(H * W, self.num_classes)
                bbox_flat = bbox_pred[b].permute(1, 2, 0).reshape(H * W, 4)
                ctr_flat = centerness[b].permute(1, 2, 0).reshape(H * W, 1)

                cls_sigmoid = torch.sigmoid(cls_flat)
                ctr_sigmoid = torch.sigmoid(ctr_flat)
                scores = cls_sigmoid * ctr_sigmoid  # (H*W, num_classes)

                for cls_idx in range(self.num_classes):
                    cls_scores_single = scores[:, cls_idx]
                    keep_idx = cls_scores_single > score_thr
                    if not keep_idx.any():
                        continue

                    valid_scores = cls_scores_single[keep_idx]
                    valid_bbox = bbox_flat[keep_idx]
                    valid_anchor = anchor[keep_idx]

                    l, t, r, b_dist = valid_bbox.unbind(-1)
                    x1 = valid_anchor[:, 0] - l
                    y1 = valid_anchor[:, 1] - t
                    x2 = valid_anchor[:, 0] + r
                    y2 = valid_anchor[:, 1] + b_dist
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    # 分别限制 x/y 坐标到各自维度范围内
                    boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, img_shape[1] - 1)
                    boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, img_shape[0] - 1)

                    img_boxes.append(boxes)
                    img_scores.append(valid_scores)
                    img_labels.append(torch.full_like(valid_scores, cls_idx, dtype=torch.long))

            if img_boxes:
                img_boxes = torch.cat(img_boxes)
                img_scores = torch.cat(img_scores)
                img_labels = torch.cat(img_labels)
                keep = nms(img_boxes, img_scores, nms_thr)
                det = torch.cat([
                    img_boxes[keep],
                    img_scores[keep].unsqueeze(1),
                    img_labels[keep].float().unsqueeze(1)
                ], dim=1)  # [N, 6]
            else:
                det = torch.empty(0, 6, device=cls_scores[0].device, dtype=torch.float32)
            batch_results.append(det)

        return batch_results if B == 1 else torch.stack(batch_results) if all(
            r.shape == batch_results[0].shape for r in batch_results
        ) else batch_results

    def inference(self, feats, img_size=(640, 640), score_thr=0.3, nms_thr=0.5):
        """单图推理（兼容旧接口），返回 [N, 6]。"""
        cls_scores, bbox_preds, centernesses = self.forward(feats)
        results = self.inference_batch(cls_scores, bbox_preds, centernesses,
                                       img_shape=img_size, score_thr=score_thr, nms_thr=nms_thr)
        if isinstance(results, list):
            return results[0] if results else torch.empty(0, 6)
        return results[0] if results.dim() == 2 else results

# ===================== 测试代码（可直接运行） =====================
if __name__ == "__main__":
    # 1. 创建 CEASCHead 实例
    head = CEASC(
        num_classes=10,  # VisDrone 数据集 10 类
        in_channels=256,
        feat_channels=256,
        strides=[8,16,32],  # 简化为 3 个尺度
        amm_cfg=dict(gumbel_noise=True, threshold=0.5),  # AMM配置
        cesc_cfg=dict(num_groups=32, dilation=1, base=0.0)  # CESC配置
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
    cls_scores, bbox_preds, centernesses, aux_loss = head.forward(feats)
    losses = head.compute_loss(cls_scores, bbox_preds, centernesses, aux_loss,
                                gt_bboxes, gt_labels, img_size=(640, 640))
    print("训练损失：")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # 4. 推理模式测试
    head.eval()
    with torch.no_grad():
        detections = head.inference(feats, img_size=(640, 640))
    print("\n推理结果：")
    print(f"检测框数量：{len(detections)}")
    if len(detections) > 0:
        print(f"第一个框：{detections[0].tolist()}")