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
    def __init__(self, in_channels, feat_channels=256, gumbel_noise=True, threshold=0.5):
        super().__init__()
        self.gumbel_noise = gumbel_noise  # 可修改参数：是否启用Gumbel噪声
        self.threshold = threshold        # 可修改参数：掩码二值化阈值
        self.gumbel_module = Gumbel(eps=1e-8)  # 引入Gumbel模块
        
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
            hard_mask: Mask类实例（替代原二值掩码）
            loss_ars: 激活率监督损失（训练时返回）
        """
        # 1. 生成软掩码 S_i (B,1,H,W)
        soft_mask = self.mask_conv(x)
        
        # 2. 使用Gumbel-Softmax生成可微分的硬掩码（替换原手动Gumbel噪声）
        hard_mask = self.gumbel_module(
            soft_mask, 
            gumbel_temp=1.0,  # 可修改参数：Gumbel温度系数，越小越接近硬阈值
            gumbel_noise=self.gumbel_noise and self.training
        )
        
        # 3. 激活率监督损失计算
        loss_ars = torch.tensor(0.0, device=x.device)
        if self.training and mask_gt is not None:
            ar_pred = torch.mean(torch.sigmoid(soft_mask), dim=[2,3])  # (B,1) 激活率
            ar_gt = torch.mean(mask_gt, dim=[2,3])      # (B,1) GT激活率
            loss_ars = self.ars_loss(ar_pred, ar_gt)
        
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

    def forward(self, x, hard_mask, global_feat=None, pw=None):  # 新增pw参数（用于sparse_gn）
        """
        Args:
            x: 输入特征 (B,C,H,W)
            hard_mask: AMM 生成的Mask类实例（替代原二值掩码）
            global_feat: 全局特征 G_i（None 时自动生成）
            pw: (mean, rstd) 用于sparse_gn的逐点统计特征（可选）
        Returns:
            out: 上下文增强稀疏卷积输出
        """
        # 1. 生成全局上下文特征 G_i
        if global_feat is None:
            global_feat = self.point_conv(x)
            global_feat = F.adaptive_avg_pool2d(global_feat, 1)  # (B,C,1,1)
        
        # 2. 稀疏卷积：使用SparseConv2d替代原普通卷积
        sparse_result = self.sparse_conv(x, hard_mask, pw=pw, gn=self.ce_gn.gn)
        
        # 处理sparse_conv的返回值（训练时可能返回元组）
        if isinstance(sparse_result, tuple):
            x_sparse, mse_loss = sparse_result
            # 如果需要使用mse_loss，可以在这里处理
            # 目前我们只需要特征部分
        else:
            x_sparse = sparse_result
        
        # 3. CE-GN 增强
        x_cegn = self.ce_gn(x_sparse, global_feat)
        
        # 4. 残差连接
        out = x_cegn + self.residual(x)
        return out

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
        
        # 1. AMM 生成掩码（返回Mask类实例）
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
        """多尺度特征前向传播 - 接收3个张量的元组输入"""
        # 确保输入是元组或列表，且包含3个特征图
        if isinstance(feats, (tuple, list)):
            assert len(feats) == 3, f"Expected 3 feature maps, got {len(feats)}"
            feats = list(feats)
        else:
            # 如果输入是单个张量，复制三次（这种情况不应该发生）
            feats = [feats] * 3
        
        # 确保特征数量与步长数匹配
        assert len(feats) == len(self.strides), f"特征数 {len(feats)} 与步长数 {len(self.strides)} 不匹配"
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
            # 返回预测结果，以便后续inference处理
            return cls_scores, bbox_preds, centernesses

    def inference(self, feats, img_size=(640,640), score_thr=0.3, nms_thr=0.5):  # 可修改参数：score_thr（置信度阈值）、nms_thr（NMS阈值）
        """推理阶段：生成最终检测框，输出YOLO兼容格式"""
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
            # 返回空的YOLO格式张量 [x1, y1, x2, y2, conf, cls]
            return torch.empty(0, 6, device=feats[0].device, dtype=torch.float32)
        
        all_boxes = torch.cat(batch_boxes)
        all_scores = torch.cat(batch_scores)
        all_labels = torch.cat(batch_labels)
        
        # 最终跨类NMS（可选，根据需求调整）
        keep = nms(all_boxes, all_scores, nms_thr)
        
        # 将结果组织为YOLO格式：[x1, y1, x2, y2, conf, cls]
        detections = torch.cat([
            all_boxes[keep],
            all_scores[keep].unsqueeze(1),
            all_labels[keep].float().unsqueeze(1)
        ], dim=1)  # [N, 6] -> [x1,y1,x2,y2,conf,cls]
        
        return detections

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