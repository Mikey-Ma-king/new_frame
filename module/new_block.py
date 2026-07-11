import torch.nn as nn
import torch as torch  
import torch.nn.functional as F 
import math

from ultralytics.nn.modules import C3k2


class space_to_depth(nn.Module):
    """空间到深度：将 H×W 上的 2×2 邻域像素展开到通道维度，输出 [B, 4C, H/2, W/2]"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    
class SPD(space_to_depth):
    """通用 space-to-depth，支持 scale>2。将 H×W 上 scale×scale 邻域展开到通道维度。"""

    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.scale == 2:
            return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)
        patches = [x[..., i::self.scale, j::self.scale] for i in range(self.scale) for j in range(self.scale)]
        return torch.cat(patches, dim=1)

class DySample(nn.Module):
    
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        """
        初始化 DySample 模块
        
        Args:
            in_channels (int): 输入通道数
            scale (int): 上采样倍数,默认为2
            style (str): 采样风格，可选 'lp'(learnable position) 或 'pl'(pixel shuffle like)
            groups (int): 分组数，用于分组卷积操作
            dyscope (bool): 是否启用动态范围控制机制

            groups 小 → 每组处理更多通道，模型容量大，精度高
            groups 大 → 每组处理更少通道，模型轻量化，速度更快
            
        使用示例:
            # 基本用法 - 2倍上采样
            upsample_layer = DySample(in_channels=64, scale=2)
            
            # 使用 pixel shuffle 风格
            upsample_layer = DySample(in_channels=64, scale=2, style='pl')
            
            # 启用动态范围控制
            upsample_layer = DySample(in_channels=64, scale=2, dyscope=True)
            
            # 高倍上采样
            upsample_layer = DySample(in_channels=128, scale=4, groups=8)
            
        参数约束:
            - 当 style='pl' 时,in_channels 必须能被 scale² 整除
            - in_channels 必须大于等于 groups 且能被 groups 整除,即为4的倍数
        """
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl'], f"style 必须是 'lp' 或 'pl'，当前为 {style}"
        
        # PL风格的通道数约束检查
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0, \
                f"PL风格要求 in_channels({in_channels}) >= scale²({scale**2}) 且能被 scale² 整除"
                
        # 分组约束检查
        assert in_channels >= groups and in_channels % groups == 0, \
            f"in_channels({in_channels}) 必须 >= groups({groups}) 且能被 groups 整除"

        # 根据不同风格设置输入输出通道数
        if style == 'pl':
            in_channels = in_channels // scale ** 2  # PL风格先进行 pixel shuffle
            out_channels = 2 * groups               # 输出2D坐标偏移量
        else:
            out_channels = 2 * groups * scale ** 2   # LP风格直接输出偏移量

        # 偏移量预测卷积层
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)  # 初始化偏移量预测网络
        
        # 可选的动态范围控制机制
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)    # scope初始值为0，逐渐学习

        # 注册初始位置缓冲区
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        """
        初始化规则采样网格位置
        
        生成标准化的采样坐标网格，范围在 [-0.5, 0.5] 之间
        这些初始位置会在训练过程中被学习到的偏移量进行调整
        
        Returns:
            torch.Tensor: 初始位置网格，形状为 [1, 2*groups, 1, 1]
                         对于 groups=4,输出形状为 [1, 8, 1, 1]
        """
        # 生成 [-0.5, 0.5] 范围内的均匀分布坐标
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        # 创建2D网格并重复到所有分组
        pos_grid = torch.stack(torch.meshgrid([h, h],indexing='ij')).transpose(1, 2)
        pos_grid = pos_grid.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)
        return pos_grid

    def sample(self, x, offset):
        """
        核心采样函数 - 使用学习到的偏移量进行动态采样
        
        Args:
            x (torch.Tensor): 输入特征图，形状 [B, C, H, W]
            offset (torch.Tensor): 学习到的坐标偏移量，形状 [B, 2*groups, H, W]
            
        Returns:
            torch.Tensor: 上采样后的特征图，形状 [B, C, scale*H, scale*W]
            
        工作流程:
        1. 生成规则采样网格坐标
        2. 加上学习到的偏移量得到动态采样位置
        3. 标准化坐标到 [-1, 1] 范围
        4. 使用 grid_sample 进行亚像素精度采样
        """
        B, _, H, W = offset.shape
        # 重塑偏移量为 [B, 2, groups, H, W] 格式
        offset = offset.view(B, 2, -1, H, W)
        
        # 生成规则网格坐标
        coords_h = torch.arange(H) + 0.5  # 像素中心对齐
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h],indexing='ij')).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        
        # 坐标标准化 - 转换到 [-1, 1] 范围用于 grid_sample
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        
        # 处理不同采样风格的坐标格式
        if self.style == 'pl':
            # PL风格：先 pixel shuffle 再 reshape
            coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
            coords = coords.reshape(B, 2, -1, self.scale * H, self.scale * W)
        else:
            # LP风格：直接 reshape
            coords = coords.reshape(B, 2, -1, self.scale * H, self.scale * W)
            
        # 调整维度顺序并展平批次
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        
        # 执行分组采样
        x_grouped = x.reshape(B * self.groups, -1, H, W)
        sampled = F.grid_sample(x_grouped, coords, mode='bilinear',
                               align_corners=False, padding_mode="border")
        
        # 恢复原始批次维度
        return sampled.view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        """
        LP (Learnable Position) 风格前向传播
        
        直接学习采样位置的偏移量，适用于一般的上采样任务
        
        Args:
            x (torch.Tensor): 输入特征图
            
        Returns:
            torch.Tensor: 上采样后的特征图
        """
        # 预测偏移量
        if hasattr(self, 'scope'):
            # 使用 scope 控制偏移量幅度
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            # 固定缩放因子的偏移量
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        """
        PL (Pixel Shuffle Like) 风格前向传播
        
        结合 pixel shuffle 操作的采样风格，适用于需要保持通道结构的任务
        
        Args:
            x (torch.Tensor): 输入特征图
            
        Returns:
            torch.Tensor: 上采样后的特征图
        """
        # 先进行 pixel shuffle 重排通道
        x_shuffled = F.pixel_shuffle(x, self.scale)
        
        # 预测并处理偏移量
        if hasattr(self, 'scope'):
            offset_pred = self.offset(x_shuffled) * self.scope(x_shuffled).sigmoid()
            offset = F.pixel_unshuffle(offset_pred, self.scale) * 0.5 + self.init_pos
        else:
            offset_pred = self.offset(x_shuffled)
            offset = F.pixel_unshuffle(offset_pred, self.scale) * 0.25 + self.init_pos
            
        return self.sample(x, offset)

    def forward(self, x):
        """
        前向传播主函数
        
        根据设置的 style 选择相应的前向传播路径
        
        Args:
            x (torch.Tensor): 输入特征图，形状 [B, C, H, W]
            
        Returns:
            torch.Tensor: 上采样后的特征图，形状 [B, C, scale*H, scale*W]
            
        使用示例:
            # 创建模块实例
            dy_upsample = DySample(in_channels=64, scale=2)
            
            # 输入特征图
            input_features = torch.randn(1, 64, 32, 32)
            
            # 执行上采样
            output_features = dy_upsample(input_features)  # 输出形状: [1, 64, 64, 64]
        """
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class SCBottleneck(nn.Module):
    """
    SCBottleneck

    定义SCBottleneck的时候，得确定好group参数，即：x.shape[1]//2
    """
    def __init__(self, in_channels, out_channels, shortcut=True, 
                 expansion=0.75, pooling_r=4, dilation=1,group=1):
        """
        初始化YOLO11适配的SCBottleneck
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            shortcut (bool): 是否使用残差连接
            expansion (float): 通道扩张比例，控制中间层通道数
            pooling_r (int): SCConv中池化下采样率
            dilation (int): 空洞卷积膨胀率
        """
        super(SCBottleneck, self).__init__()
        
        # 计算中间通道数
        # 为了之后的SC Conv中能够用到相加，尽量保证hidden_channels == in_channels
        # 即：int(out_channels * expansion) == in_channels
        hidden_channels = int(out_channels * expansion)
        
        # 双路径设计：传统卷积路径 + SCConv路径
        self.shortcut = shortcut and in_channels == out_channels
        
        self.c = in_channels // 2  # 每个分支处理一半通道数

        # 传统卷积分支
        self.conv_branch = nn.Sequential(
            nn.Conv2d(self.c, hidden_channels, 1,bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),  # YOLO常用激活函数
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 
                     padding=dilation, dilation=dilation,groups=1,bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )
        
        # SCConv分支 - 保留自校准机制
        self.scconv_branch = SCConv(
            self.c, hidden_channels, 
            pooling_r=pooling_r, dilation=dilation,group=self.c)
        # SCConv的参数是group，传统Conv2d卷积参数是groups
        
        # 特征融合和输出投影
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, 1,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.pre = nn.Conv2d(in_channels, 2 * self.c, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 输出特征图
        """
        # 双路径并行处理

        y = self.pre(x)
        y1,y2 = y.split([self.c,self.c],1)
        y = [y1,y2]

        conv_out = self.conv_branch(y[0])
        scconv_out = self.scconv_branch(y[1])
        
        # 特征融合
        fused = torch.cat([conv_out, scconv_out], dim=1)
        out = self.fusion(fused)
            
        return out

class SCConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_r=4, dilation=1,group=1):
        """
        初始化简化版SCConv
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            pooling_r (int): 平均池化下采样率
            dilation (int): 空洞卷积膨胀率
        """
        super(SCConv, self).__init__()
        
        # 使用相同的中间通道数
        mid_channels = in_channels * 2
        
        # k2分支: 全局上下文分支
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1,
                     padding=dilation, dilation=dilation,groups=group,bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        
        # k3分支: 局部特征分支
        self.k3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1,
                     padding=dilation, dilation=dilation,groups=group,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )
        
        # k4: 输出投影，groups=1 保证兼容任意 out_channels
        self.k4 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1,
                     padding=dilation, dilation=dilation, groups=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 输出特征图
        """
        identity = x
        
        # 自校准过程：捕获全局上下文信息

        # 这一步，上采样到x的尺寸，通道：in_channels->mid_channels = out_channels
        global_context = F.interpolate(
            self.k2(x), 
            size=identity.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 生成注意力权重
        # 这一步如何保证相加的时候通道匹配？
        # 因为通过k2以后的通道数已经变为out_channels了，但是x还是in_channels
        # 要加一个判断的步骤
        if identity.shape[1] != global_context.shape[1]: 
            attention = torch.sigmoid(global_context)
        else:
            attention = torch.sigmoid(identity + global_context)
        
        # 局部特征与注意力结合
        local_features = self.k3(x)
        attended_features = local_features * attention
        
        # 最终输出投影
        out = self.k4(attended_features)

        return out 

class SPD_SCConv(SCBottleneck):

    def __init__(self, in_channels, out_channels,
                 pooling_r=4, dilation=1, scale=2, group=1):
        super().__init__(in_channels * (scale ** 2), out_channels,
                         pooling_r=pooling_r, dilation=dilation, group=group)

        self.SPD = SPD(scale)
        self.Upsample = DySample(in_channels * (scale ** 2), scale, groups=1)

    def forward(self, x):
        spd_output = self.SPD(x)
        upsampled_features = self.Upsample(spd_output)
        return super().forward(upsampled_features)
    
class SimAM(torch.nn.Module):

    def __init__(self, channels=None, e_lambda=1e-4):
        """
        初始化SimAM模块
        
        Args:
            channels (int, optional): 输入通道数（该参数实际未使用，保留是为了接口兼容性）
            e_lambda (float): 正则化参数，用于数值稳定性,默认值1e-4
            e_lambda什么情况下是可学习的?
        """
        super(SimAM, self).__init__()

        # Sigmoid激活函数，用于将注意力权重映射到[0,1]区间
        self.activaton = nn.Sigmoid()
        # 正则化系数，确保数值计算稳定
        self.e_lambda = e_lambda

    def __repr__(self):
        """
        返回模块的字符串表示形式
        
        Returns:
            str: 包含模块名称和正则化参数的描述字符串
        """
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
    # 返回 SimAM（lambda = ？）
    @staticmethod
    def get_module_name():
        """
        获取模块的标准名称
        
        Returns:
            str: 模块名称 "SimAM"
        """
        return "SimAM"

    def forward(self, x):
        # 获取输入张量的维度信息 [批次大小, 通道数, 高度, 宽度]
        b, c, h, w = x.size()
        
        # 计算空间位置总数减1（用于无偏方差计算的自由度修正）
        # 这是概率论，无偏方差是样本值减去均值统计量的平方和的n-1分之一，因此n=h*w-1
        n = w * h - 1

        # x.mean(dim=[2,3], keepdim=True)形状：[B, C, 1, 1](每个批次、每个通道一个均值)
        # 这个减法是一个广播减法:  [B, C, 1, 1] 自动扩展到匹配形状[B.C.H.W],每个值都相等
        # keepdim为True，可以保证结果维度即使是1，也仍然回保持被聚合的维度
        # dim参数是用来表示计算均值的维度，如果没有，则表示计算全局均值，得到的是标量
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        
        # 核心注意力权重计算公式
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 应用Sigmoid激活函数将权重限制在[0,1]范围内
        # 然后与原始输入特征相乘，实现自适应的特征增强
        # 重要特征得到放大，不相关特征被抑制
        return x * self.activaton(y)
    
class SimAM_C3k2(C3k2):
    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.SimAM = SimAM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.SimAM(super().forward(x))
    
class SeparableConvBlock(nn.Module):
    """
    created by Ma
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x
    
# 手动计算图片填充并且卷积
class Conv2dStaticSamePadding(nn.Module):
    """
    created by Ma
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,dilation = dilation,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x 
    
class Conv2dDynamicSamePadding(nn.Conv2d):
    """
    动态Same Padding的2D卷积层,类似TensorFlow的实现方式
    
    该类在每次前向传播时动态计算所需的padding,确保输出特征图
    的空间尺寸与输入保持相同的尺寸关系(当stride=1时完全相等)
    
    实现原理：
    1. 计算期望输出尺寸:oh = ceil(ih / sh), ow = ceil(iw / sw)
    2. 根据卷积公式反推所需padding:
       pad = max((output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size, 0)
    3. 将padding均匀分配到四周边界
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        """
        初始化动态Same Padding卷积层
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int or tuple): 卷积核大小
            stride (int or tuple): 步长,默认为1
            dilation (int or tuple): 膨胀率,默认为1
            groups (int): 分组卷积的组数,默认为1
            bias (bool): 是否使用偏置,默认为True
        """
        # 调用父类初始化，padding设为0（后面会动态计算）
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        # 确保stride是长度为2的列表格式 [sh, sw]
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        """
        前向传播,动态计算并应用Same Padding
        
        Args:
            x (Tensor): 输入特征图，形状为[B, C, H, W]
            
        Returns:
            Tensor: 输出特征图,空间尺寸根据stride和padding计算得出
        """
        # 获取输入特征图的空间尺寸
        ih, iw = x.size()[-2:]  # 输入高度和宽度
        
        # 获取卷积核的空间尺寸
        # self.weight 实际上就是这个卷积层的所有可学习参数
        # self.weight的形状为: [out_channels, in_channels/groups, kernel_height, kernel_width]
        kh, kw = self.weight.size()[-2:]  # 卷积核高度和宽度
        
        # 获取步长
        sh, sw = self.stride  # 高度和宽度方向的步长
        
        # 计算期望输出尺寸（向上取整）
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        
        # 根据卷积公式计算所需的padding
        # 公式：(output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - input_size
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        
        # 如果需要padding，则进行填充
        if pad_h > 0 or pad_w > 0:
            # 将padding均匀分配：左/上边界填充较少，右/下边界填充较多
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            
        # 执行标准卷积操作
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
# static版本	                 Dynamic版本
# nn.Module包装	                直接继承nn.Conv2d
# 本质上都是动态计算padding的

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):# ctx是torch.autograd.Function的forward方法中的一个参数
        result = i * torch.sigmoid(i) # 计算输入i对应的swish函数的值
        ctx.save_for_backward(i)# 使用 ctx.save_for_backward(i) 保存输入值 i，以便在反向传播时使用
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
    # 计算i对应的 Swish 函数的导数，之后返回梯度乘以 Swish 的导数，完成链式法则的计算
    # grad_output出现在自定义torch.autograd.Function的backward方法

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    # 继承了torch.autograd.Function里面apply函数，可以直接调用


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class MaxPool2dDynamicSamePadding(nn.Module):
    """
    created by Ma
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x   

class BiFPN(nn.Module):
    """
    modified by Ma
    """

    def __init__(self, num_channels: int , conv_channels : tuple = (), epsilon=1e-4, onnx_export=False, attention=False
                 ):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the backbone,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon

        # Conv layers
        self.conv_8_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv_6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv_4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv_2_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv_4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv_6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv_8_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv_10_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        # Use small group size for upsample to avoid exploding batch-group reshape
        # setting groups=num_channels causes x to be reshaped to (B*num_channels, ...)
        # which massively increases memory. Use groups=1 (or a small fixed value) instead.
        self.upsample_8 = DySample(in_channels=num_channels, scale=2, groups=1)
        self.upsample_6 = DySample(in_channels=num_channels, scale=2, groups=1)
        self.upsample_4 = DySample(in_channels=num_channels, scale=2, groups=1)
        self.upsample_2 = DySample(in_channels=num_channels, scale=2, groups=1)

        self.downsample_4 = MaxPool2dDynamicSamePadding(3, 2)
        self.downsample_6 = MaxPool2dDynamicSamePadding(3, 2)
        self.downsample_8 = MaxPool2dDynamicSamePadding(3, 2)
        self.downsample_10 = MaxPool2dDynamicSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        if any(x for x in conv_channels) != num_channels:
            self.first_time = True
        else:
            self.first_time = False

        if self.first_time:
            self.down_channel_2 = nn.Sequential(
                SPD_SCConv(conv_channels[0], num_channels, group=conv_channels[0]),
                # SPD_SC Conv继承SCConvBottleneck，没有stride参数
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.down_channel_4 = nn.Sequential(
                SPD_SCConv(conv_channels[1], num_channels, group=conv_channels[1]),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.down_channel_6 = nn.Sequential(
                SPD_SCConv(conv_channels[2], num_channels, group=conv_channels[2]),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.down_channel_8 = nn.Sequential(
                SPD_SCConv(conv_channels[3], num_channels, group=conv_channels[3]),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.down_channel_10 = nn.Sequential(
                SPD_SCConv(conv_channels[4], num_channels, group=conv_channels[4]),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )


        # Weight
        self.w1_8 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w1_8_relu = nn.ReLU()
        self.w1_6 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w1_6_relu = nn.ReLU()
        self.w1_4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w1_4_relu = nn.ReLU()
        self.w1_2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w1_2_relu = nn.ReLU()

        self.w2_4 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w2_4_relu = nn.ReLU()
        self.w2_6 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w2_6_relu = nn.ReLU()
        self.w2_8 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w2_8_relu = nn.ReLU()
        self.w2_10 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_10_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            10_0---------------------------> 10_2 -------->
               |-------------|                ↑
                             ↓                |
            8_0 ---------> 8_1 ------------> 8_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            6_0 ---------> 6_1 ------------> 6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            4_0 ---------> 4_1 ------------> 4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            2_0 ---------------------------> 2_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)# 带权重参数学习
        else:
            outs = self._forward(inputs)# 不带权重参数学习

        return outs

    def _forward_fast_attention(self, inputs):

        if self.first_time:
            l2, l4, l6 , l8 , l10 = inputs

            l2_in = self.down_channel_2(l2)
            l4_in = self.down_channel_4(l4)
            l6_in = self.down_channel_6(l6)
            l8_in = self.down_channel_8(l8)
            l10_in = self.down_channel_10(l10)
        else:
        
            l2_in, l4_in, l6_in, l8_in, l10_in = inputs

        w1_8 = self.w1_8_relu(self.w1_8)
        weight = w1_8 / (torch.sum(w1_8, dim=0) + self.epsilon)
        l8_up = self.conv_8_up(self.swish(weight[0] * l8_in + weight[1] * self.upsample_8(l10_in)))

        
        w1_6 = self.w1_6_relu(self.w1_6)
        weight = w1_6 / (torch.sum(w1_6, dim=0) + self.epsilon)
        l6_up = self.conv_6_up(self.swish(weight[0] * l6_in + weight[1] * self.upsample_6(l8_up)))

        w1_4 = self.w1_4_relu(self.w1_4)
        weight = w1_4 / (torch.sum(w1_4, dim=0) + self.epsilon)
        l4_up = self.conv_4_up(self.swish(weight[0] * l4_in + weight[1] * self.upsample_6(l6_up)))

        w1_2 = self.w1_2_relu(self.w1_2)
        weight = w1_2 / (torch.sum(w1_2, dim=0) + self.epsilon)
        l2_out = self.conv_2_up(self.swish(weight[0] * l2_in + weight[1] * self.upsample_2(l4_up)))

        
        w2_4 = self.w2_4_relu(self.w2_4)
        weight = w2_4 / (torch.sum(w2_4, dim=0) + self.epsilon)
        l4_out = self.conv_4_down(
            self.swish(weight[0] * l4_in + weight[1] * l4_up + weight[2] * self.downsample_4(l2_out)))

        w2_6 = self.w2_6_relu(self.w2_6)
        weight = w2_6 / (torch.sum(w2_6, dim=0) + self.epsilon)
        l6_out = self.conv_6_down(
            self.swish(weight[0] * l6_in + weight[1] * l6_up + weight[2] * self.downsample_6(l4_out)))

        w2_8 = self.w2_8_relu(self.w2_8)
        weight = w2_8 / (torch.sum(w2_8, dim=0) + self.epsilon)
        l8_out = self.conv_8_down(
            self.swish(weight[0] * l8_in + weight[1] * l8_up + weight[2] * self.downsample_8(l6_out)))


        w2_10 = self.w2_10_relu(self.w2_10)
        weight = w2_10 / (torch.sum(w2_10, dim=0) + self.epsilon)
        l10_out = self.conv_10_down(
            self.swish(weight[0] * l10_in + weight[1] * self.downsample_10(l8_out)))


        return l2_out, l4_out, l6_out, l8_out, l10_out

    def _forward(self, inputs):
       
        if self.first_time:
            l2, l4, l6 , l8 , l10 = inputs

            l2_in = self.down_channel_2(l2)
            l4_in = self.down_channel_4(l4)
            l6_in = self.down_channel_6(l6)
            l8_in = self.down_channel_8(l8)
            l10_in = self.down_channel_10(l10)
        else:
        
            l2_in, l4_in, l6_in, l8_in, l10_in = inputs

        
        l8_up = self.conv_8_up(self.swish( l8_in + self.upsample_8(l10_in)))

        l6_up = self.conv_6_up(self.swish( l6_in +  self.upsample_6(l8_up)))

        l4_up = self.conv_4_up(self.swish( l4_in + self.upsample_6(l6_up)))

        l2_out = self.conv_2_up(self.swish( l2_in +  self.upsample_2(l4_up)))

        l4_out = self.conv_4_down(
            self.swish(l4_in + l4_up + self.downsample_4(l2_out)))

        l6_out = self.conv_6_down(
            self.swish(l6_in + l6_up + self.downsample_6(l4_out)))

        l8_out = self.conv_8_down(
            self.swish(l8_in +  l8_up + self.downsample_8(l6_out)))


        l10_out = self.conv_10_down(
            self.swish(l10_in + self.downsample_10(l8_out)))

        return l2_out, l4_out, l6_out, l8_out, l10_out


class Sequential_BiFPN(nn.Module):
    """顺序处理的BiFPN模块，将多个BiFPN层按顺序堆叠形成深层特征融合网络。"""

    def __init__(self, num_channels: int, num_layers: int = 2, conv_channels: tuple = (),
                 epsilon=1e-4, onnx_export=False, attention=False,
                 output_layers=None):
        """初始化顺序BiFPN模块。

        Args:
            num_channels: 特征通道数
            num_layers: BiFPN层数量
            conv_channels: 输入特征通道数元组
            epsilon: 权重归一化的小数值
            onnx_export: 是否为ONNX导出优化
            attention: 是否使用注意力权重
            output_layers: 输出层索引元组，None 时使用默认 3 输出融合 (P3/P4/P5)
        """
        super(Sequential_BiFPN, self).__init__()

        self.bifpn_layers = nn.ModuleList([
            BiFPN(num_channels, conv_channels, epsilon, onnx_export, attention)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.output_layers = output_layers
        self.downsample = MaxPool2dDynamicSamePadding(3, 2)
        self.upsample = DySample(in_channels=num_channels, scale=2, groups=num_channels)

    def forward(self, inputs):
        """前向传播，依次通过多层BiFPN，输出指定层的特征。"""
        x = self.bifpn_layers[0](inputs)
        for i in range(1, self.num_layers):
            if hasattr(self.bifpn_layers[i], "first_time"):
                self.bifpn_layers[i].first_time = False
            x = self.bifpn_layers[i](x)

        if self.output_layers is None:
            # 默认：5→3 层融合输出，兼容 CEASC (P3/P4/P5 三个尺度)
            out1 = self.downsample(x[0]) + x[1]
            out2 = self.upsample(x[3]) + x[2]
            out3 = x[4]
            return out1, out2, out3
        return tuple(x[i] for i in self.output_layers)