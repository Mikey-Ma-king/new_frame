import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8DetectionLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)
from module.new_block import SPD_SCConv, DySample,  SimAM_C3k2, BiFPN,Sequential_BiFPN
from module.Head import CEASC


class BaseModel(torch.nn.Module):
    """
    Ultralytics家族中所有YOLO模型的基类。

    该类提供了YOLO模型的通用功能，包括前向传播处理、模型融合、
    信息显示和权重加载功能。

    属性:
        model (torch.nn.Module): 神经网络模型。
        save (list): 需要保存输出的层索引列表。
        stride (torch.Tensor): 模型步长值。

    方法:
        forward: 执行训练或推理的前向传播。
        predict: 对输入张量执行推理。
        fuse: 融合Conv2d和BatchNorm2d层以优化。
        info: 打印模型信息。
        load: 加载权重到模型。
        loss: 计算训练损失。

    示例:
        创建一个BaseModel实例
        >>> model = BaseModel()
        >>> model.info()  # 显示模型信息
    """

    def forward(self, x, *args, **kwargs):
        """
        执行模型的前向传播，用于训练或推理。

        如果x是一个字典，则计算并返回训练的损失。否则，返回推理的预测结果。

        Args:
            x (torch.Tensor | dict): 推理的输入张量，或者包含图像张量和标签的字典用于训练。
            *args (Any): 可变长度参数列表。
            **kwargs (Any): 任意关键字参数。

        Returns:
            (torch.Tensor): 如果x是字典（训练），则为损失；否则为网络预测结果（推理）。
        """
        if isinstance(x, dict):  # 适用于训练和验证的情况
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        通过网络执行前向传播。

        Args:
            x (torch.Tensor): 模型的输入张量。
            profile (bool): 如果为True，则打印每层的计算时间。
            visualize (bool): 如果为True，则保存模型的特征图。
            augment (bool): 预测时增强图像。
            embed (list, optional): 要返回的特征向量/嵌入列表。

        Returns:
            (torch.Tensor): 模型的最后输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        通过网络执行一次前向传播。

        Args:
            x (torch.Tensor): 模型的输入张量。
            profile (bool): 如果为True，则打印每层的计算时间。
            visualize (bool): 如果为True，则保存模型的特征图。
            embed (list, optional): 要返回的特征向量/嵌入列表。

        Returns:
            (torch.Tensor): 模型的最后输出。
        """
        y, dt, embeddings = [], [], [] # 输出列表
        embed = frozenset(embed) if embed is not None else {-1}  # 将embed转换为冻结集，如果为None则默认为{-1}
        max_idx = max(embed)  # 获取embed中最大的索引
        for m in self.model:  # 遍历模型中的每一层
            if m.f != -1:  # 如果不是来自前一层（即需要从其他层获取输入）
                # 从之前的层获取输入：如果m.f是整数，则从y[m.f]获取；如果是列表，则从多个层获取
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从前面的层获取
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行当前层
            # 保存输出：如果当前层索引在self.save中则保存输出，否则保存None
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:  # 如果当前层索引在embed列表中
                # 对特征进行自适应平均池化并展平
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:  # 如果是embed中最大的索引，则返回嵌入
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x  # 返回最终输出

    def _predict_augment(self, x):
        """对输入图像x执行增强并返回增强推理结果。"""
        LOGGER.warning(
            f"{self.__class__.__name__} 不支持 'augment=True' 的预测。"
            f"返回单尺度预测。"
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        配置模型单层在给定输入下的计算时间和FLOPs。

        Args:
            m (torch.nn.Module): 需要配置的层。
            x (torch.Tensor): 层的输入数据。
            dt (list): 存储层计算时间的列表。
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda支持，不需要安装'ultralytics-thop'

        c = m == self.model[-1] and isinstance(x, list)  # 是否是最后一层列表，复制输入作为inplace修复
        # 计算每秒浮点运算次数，如果有thop则计算，否则为0
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        将模型的`Conv2d()`和`BatchNorm2d()`层融合成单个层，以提高计算效率。

        Returns:
            (torch.nn.Module): 返回融合后的模型。
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积
                    delattr(m, "bn")  # 移除批归一化
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # 移除批归一化
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # 更新前向传播
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # 移除one2many头
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        检查模型是否具有少于某个阈值的BatchNorm层。

        Args:
            thresh (int, optional): BatchNorm层的阈值数量。

        Returns:
            (bool): 如果模型中BatchNorm层的数量少于阈值，则为True，否则为False。
        """
        # 获取标准化层，即BatchNorm2d()
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # 如果模型中< 'thresh'个BatchNorm层则为True

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        打印模型信息。

        Args:
            detailed (bool): 如果为True，则打印模型的详细信息。
            verbose (bool): 如果为True，则打印模型信息。
            imgsz (int): 模型将训练的图像大小。
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        将函数应用于模型中不是参数或已注册缓冲区的所有张量。

        Args:
            fn (function): 应用于模型的函数。

        Returns:
            (BaseModel): 更新后的BaseModel对象。
        """
        self = super()._apply(fn)  # 调用父类的_apply方法
        m = self.model[-1]  # 获取最后一个模块 (Detect())
        # 如果是Detect类（包括Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment等子类）
        if isinstance(
            m, Detect
        ):
            m.stride = fn(m.stride)  # 对stride应用函数
            m.anchors = fn(m.anchors)  # 对anchors应用函数
            m.strides = fn(m.strides)  # 对strides应用函数
        return self

    def load(self, weights, verbose=True):
        """
        将权重加载到模型中。

        Args:
            weights (dict | torch.nn.Module): 要加载的预训练权重。
            verbose (bool, optional): 是否记录传输进度。
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision模型不是字典
        csd = model.float().state_dict()  # 检查点状态字典转为FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # 交集
        self.load_state_dict(updated_csd, strict=False)  # 加载
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # 硬编码到yolo模型
        # 主要用于提升多通道训练
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"从预训练权重转移了 {len_updated_csd}/{len(self.model.state_dict())} 个项目")

    def loss(self, batch, preds=None):
        """
        计算损失。

        Args:
            batch (dict): 要计算损失的批次。
            preds (torch.Tensor | list[torch.Tensor], optional): 预测结果。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """created by ma 初始化BaseModel的损失准则。"""
        raise NotImplementedError("compute_loss() 需要由任务头实现")


class DetectionModel(BaseModel):
    """
    YOLO检测模型。

    该类实现了YOLO检测架构，处理模型初始化、前向传播、
    增强推理和损失计算等目标检测任务。

    属性:
        yaml (dict): 模型配置字典。
        model (torch.nn.Sequential): 神经网络模型。
        save (list): 需要保存输出的层索引列表。
        names (dict): 类名字典。
        inplace (bool): 是否使用就地操作。
        end2end (bool): 模型是否使用端到端检测。
        stride (torch.Tensor): 模型步长值。

    方法:
        __init__: 初始化YOLO检测模型。
        _predict_augment: 执行增强推理。
        _descale_pred: 反向缩放增强推理后的预测。
        _clip_augmented: 裁剪YOLO增强推理的尾部。
        init_criterion: 初始化损失准则。

    示例:
        初始化检测模型
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """
        使用给定的配置和参数初始化YOLO检测模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数。
            verbose (bool): 是否显示模型信息。
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict，yaml - > dict

        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` 模块已被弃用，建议使用torch.nn.Identity。"
                "请删除本地*.pt文件并重新下载最新的模型检查点。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # 定义模型
        self.yaml["channels"] = ch  # 保存通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"覆盖model.yaml nc={self.yaml['nc']} 为 nc={nc}")
            self.yaml["nc"] = nc  # 覆盖YAML值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist ,dict -> model
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认名称字典
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 构建步长
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包括所有Detect子类如Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x 最小步长
            m.inplace = self.inplace

            def _forward(x):
                """created by ma 通过模型执行前向传播，处理不同的Detect子类类型。"""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB)) else self.forward(x)

            self.model.eval()  # 避免在训练开始前更改批次统计信息
            m.training = True  # 设置为True以正确返回步长
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 前向传播
            self.stride = m.stride
            self.model.train()  # 将模型设置回训练模式（默认）
            m.bias_init()  # 只运行一次
        else:
            self.stride = torch.Tensor([32])  # 默认步长，例如RTDETR

        # 初始化权重、偏置
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """
        对输入图像x执行增强并返回增强推理和训练输出。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            (torch.Tensor): 增强推理输出。
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("模型不支持 'augment=True'，返回单尺度预测。")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # 高度, 宽度
        s = [1, 0.83, 0.67]  # 缩放比例
        f = [None, 3, None]  # 翻转（2-上下，3-左右）
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # 前向传播
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强尾部
        return torch.cat(y, -1), None  # 增强推理, 训练

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        反向缩放增强推理后的预测（逆操作）。

        Args:
            p (torch.Tensor): 预测张量。
            flips (int): 翻转类型（0=无，2=上下，3=左右）。
            scale (float): 缩放因子。
            img_size (tuple): 原始图像大小（高度，宽度）。
            dim (int): 分割的维度。

        Returns:
            (torch.Tensor): 反向缩放后的预测。
        """
        p[:, :4] /= scale  # 反向缩放
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # 反向翻转上下
        elif flips == 3:
            x = img_size[1] - x  # 反向翻转左右
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        裁剪YOLO增强推理的尾部。

        Args:
            y (list[torch.Tensor]): 检测张量列表。

        Returns:
            (list[torch.Tensor]): 裁剪后的检测张量。
        """
        nl = self.model[-1].nl  # 检测层的数量（P3-P5）
        g = sum(4**x for x in range(nl))  # 网格点
        e = 1  # 排除层计数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][..., :-i]  # 大
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][..., i:]  # 小
        return y

    def init_criterion(self):
        """created by ma 初始化DetectionModel的损失准则。"""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class MyModel(DetectionModel):
    """
    created by ma

    自定义模型类,用于加载model_0到model_8的YAML配置文件。
    该类扩展了BaseModel,支持加载特定的模型配置文件，用于目标检测任务。
    
    属性:
        yaml (dict): 模型配置字典。
        model (torch.nn.Sequential): 神经网络模型。
        save (list): 需要保存输出的层索引列表。
        names (dict): 类名字典。
        inplace (bool): 是否使用就地操作。
        stride (torch.Tensor): 模型步长值。
    
    方法:
        __init__: 初始化自定义检测模型。
        init_criterion: 初始化损失准则。
    
    示例:
        >>> model = MyModel("model_0.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """
    
    def __init__(self, cfg="model_0.yaml", ch=3, nc=None, verbose=True):
        """Wrap Ultralytics' DetectionModel initialization so `MyModel` is fully compatible with
        Ultralytics trainer while preserving existing YAML-based behavior.

        This delegates construction to `DetectionModel.__init__` which already handles YAML
        parsing, channel override, stride computation and weight initialization. We keep the
        class name and location so other code importing `pram.tasks.MyModel` continues to work.
        """
        # Delegate to DetectionModel to ensure full compatibility with trainer expectations
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # preserve explicit task attribute
        self.task = getattr(self, 'task', 'detect')

    def init_criterion(self):
        """初始化MyModel的损失准则。"""
        # 返回检测损失函数实例
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class Ensemble(torch.nn.ModuleList):
    """
    Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through
    model averaging or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """
        Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """
    Load a single model weights.

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model

    # Model compatibility updates
    model.args = args  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """
    created by ma

    将model.yaml字典解析为PyTorch模型
    Parse a model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): 模型字典
        ch (int): 输入通道数
        verbose (bool): 是否打印模型详情

    Returns:
        model (torch.nn.Sequential): PyTorch模型
        save (list): 输出层的有序列表
    """
    import ast

    # Args
    legacy = True  # 向后兼容性标志，用于Yolo v3/v5/v8/v9模型
    max_channels = float("inf")  # 最大通道数，默认为无穷大
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))  # 获取类别数、激活函数和缩放参数
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))  # 获取深度倍数、宽度倍数和关键点形状
    scale = d.get("scale")
    if scales:
        if not scale:# 没有指定"scale"参数
            scale = tuple(scales.keys())[0]# 使用n的"scale"参数
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:# 指定激活函数
        # 重新定义默认激活函数
        Conv.default_act = eval(act) # eval(act) 将act这个字符串转换为实际的PyTorch激活函数对象后，赋值给另一个激活函数对象
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印激活函数信息

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
        # : - 这是格式说明符的开始标记，它分隔了要格式化的值和格式说明
        # > -这是对齐方式的标识符，表示右对齐（Right Align）。还有其他对齐选项：
        # > 右对齐（Right Align）：内容靠右，左边填充空格
        # < 左对齐（Left Align）：内容靠左，右边填充空格
        # ^ 居中对齐（Center Align）：内容居中，两边填充空格
        # 如果没有对齐符号，默认情况下数字是右对齐，字符串是左对齐
        # 数字 - 这是字段的宽度，表示该字段占用的字符数。

    ch = [ch]  
    # 将输入通道数转换为列表
    def get_ch(idx):
        return ch[idx] if isinstance(idx, int) else sum(ch[x] for x in idx)

    layers, save, c2 = [], [], ch[-1]  
    # 初始化层列表、保存列表和输出通道数
    
    # 定义基础模块集合，这些模块是YOLO模型的基本组成部分
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            SPD_SCConv,
            SimAM_C3k2,
        }
    )
    
    # 定义可重复模块集合，这些模块支持'repeat'参数
    repeat_modules = frozenset(  # 具有'repeat'参数的模块
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
            SimAM_C3k2,
        }
    )
    
    # 遍历骨干网络(backbone)和头部(head)的所有层
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        # enumerate() 的功能：
        #将一个可迭代对象（如列表）转换为索引-值对的迭代器
        #返回的每个元素是一个元组，包含索引和对应的值
        # 此处返回索引是i，值对是元组（from, repeat_number, module, args），其中f,n,m,args都是字符串
        # 接下来需要知道：default是默认值，default arguments是默认参数，attribute是属性，分为实例属性和类属性
        # 实例属性只有实例化以后具有，不同的实例具有不同的实例属性，同一个类具有相同的类属性

        # 这段代码是用于动态获取模块类的，根据模块字符串m的不同格式选择不同的获取方式
        m = (
            getattr(torch.nn, m[3:]) if "nn." in m  
            # 检查字符串m是否包含"nn."，如果是则从torch.nn模块中获取对应类
            else getattr(__import__("torchvision").ops, m[16:])  if "torchvision.ops." in m  
            # 现场导入torchvision库
            # 检查字符串m是否包含"torchvision.ops."，如果是则从torchvision.ops中获取对应类
            else globals()[m]  
            # globals包含的范围是什么？内置模块，文件中的全局变量，所有导入的模块或者模块里的内容
            # 如果以上都不匹配，则从全局命名空间中获取模块，即直接使用globals()字典获取模块类
        )
        
        # 处理参数，将字符串参数转换为实际值
        for j, a in enumerate(args):
            # enumerate只要是可迭代对象就可以（列表 元组 字符串 集合 字典）
            if isinstance(a, str):
                # 如果不是字符串，则跳过处理
                with contextlib.suppress(ValueError):
                    # 使用上下文管理器捕获 ValueError 异常。
                    # 如果参数转换过程中出现 ValueError，该异常将被静默处理，不会中断程序执行。
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    # 这里locals的作用域是什么？
                    # 包括函数参数和函数内部定义的变量，包括nc, act, scales，depth, width, kpt_shape
                    # 但不包含base_modules里面的模块，他只是作为一个变量囊括进来了
                    # i, f, n, m, args
        
        # 应用深度倍数，调整模块重复次数
        n = n_ = max(round(n * depth), 1) if n > 1 else n  
        # 深度增益
        # 对重复次数进行处理，将重复次数调整为至少为1，并取其整数部分
        # 重复次数对应着模块图片里的2*d（只有在C3k2里才有，其余都是一次）
        


        # 处理基础模块
        if m in base_modules:
            c1, c2 = get_ch(f), args[0]  
            # 更新输出通道数
            # 结合实际yaml里面的backbone和head中的from参数，不难发现都是-1
            # 则ch[f]都是ch[-1]，是ch中的最后一个元素，即输入通道数，一般都是提前给定的默认值
            # 这也是为什么要在i=0的时候先清除列表，再添加新值的原因
            # 对新值的添加在最后面
            
            # 如果输出通道数不等于类别数（如Classify()输出）
            if c2 != nc:  # 在原来的yaml文件中，nc = 80，因此不可能相等，所以全局的输出通道数都要这么处理
                c2 = make_divisible(min(c2, max_channels) * width, 8)  
                # divisor 除数 ，make_divisible 
                # nc是number of classes的意思，即要求识别的类别总数
                # 此时max_channels默认为无穷大，但图中要求对于不同的模型有不同的max_channels（mc）
                # 为什么要求能被8整除？至少保证输出通道数是8的倍数，不至于比8小

            # 特殊处理C2fAttn模块：设置嵌入通道数和头数
            if m is C2fAttn:  # 设置 1) 嵌入通道数 2) 注意力头数
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            # 构建模块参数列表，args[1:]为从输出通道数以后的维度，最终为输入、输出、...
            args = [c1, c2, *args[1:]]
            # 如果是可重复模块，将重复次数插入到参数列表的第2个位置
            if m in repeat_modules:
                args.insert(2, n)  # 重复次数,将n插入到列表的索引为2的地方
                n = 1  # 重置为1，因为重复次数已包含在模块内部
            
            # 特殊处理C3k2模块（针对M/L/X尺寸）
            if m is C3k2 or m is SimAM_C3k2:  # for M/L/X sizes
                legacy = False  # 禁用向后兼容模式
                if scale in "mlx":  # 如果是m、l或x规模的模型
                    args[1] = True # 将第1个C3k2模块的参数改为True
                # 只有n、s模型才只使用bottleneck进行特征提取            
            # 特殊处理A2C2f模块（针对L/X尺寸）

            if m is A2C2f:
                legacy = False  # 禁用向后兼容模式
                if scale in "lx":  # 如果是l或x规模的模型
                    args.extend((True, 1.2))  # 添加额外参数
                    
            # 特殊处理C2fCIB模块
            if m is C2fCIB:
                legacy = False
                
        
        # 特殊处理AIFI模块
        elif m is AIFI:
            args = [get_ch(f), *args]

        # 特殊处理BiFPN模块
        elif m is BiFPN:
            args.append([ch[x] for x in f])
            c2 = args[0] * width  # 输出通道数为args[0]，即BiFPN模块的第一个参数

        elif m is Sequential_BiFPN:
            args.append(n)  # 添加重复次数参数
            args.append([ch[x] for x in f])
            c2 = args[0] * width  # 输出通道数为args[0]，即Sequential_BiFPN模块的第一个参数
            
        # 特殊处理HGStem和HGBlock模块
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = get_ch(f), args[0], args[1]  # 输入、中间和输出通道数
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:  # 如果是HGBlock，插入重复次数
                args.insert(4, n)  # 重复次数
                n = 1
                
        # 特殊处理ResNetLayer模块
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4  # 根据是否下采样确定输出通道数
            
        # 特殊处理BatchNorm2d模块
        elif m is torch.nn.BatchNorm2d:
            args = [get_ch(f)]  # 仅使用输入通道数作为参数
            
        # 特殊处理Concat模块（连接操作）
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # 输出通道数为所有输入通道数之和
            
        # 特殊处理检测和分割相关模块
        elif m in frozenset(
            {Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}
        ):
            # 没有对c2的处理，使得c2与上一个的输出通道数保持一致————检测头不改变通道数
            args.append([ch[x] for x in f])  # 添加输入通道数列表
            if m is Segment or m is YOLOESegment:  
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)  # 应用宽度倍数
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy  # 设置向后兼容标志

        elif m is CEASC:
            c2 = ch[-1]
                
        # 特殊处理RTDETRDecoder模块
        elif m is RTDETRDecoder:  # 特殊情况，通道参数必须放在索引1位置
            args.insert(1, [ch[x] for x in f])  # 插入输入通道数列表
            
        # 特殊处理CBLinear模块
        elif m is CBLinear:
            c2 = args[0]  # 输出通道数
            c1 = get_ch(f)  # 输入通道数
            args = [c1, c2, *args[1:]]  # 构建参数列表
            
        # 特殊处理CBFuse模块
        elif m is CBFuse:
            last = f[-1] if isinstance(f, (list, tuple)) else f
            c2 = get_ch(last)  # 输出通道数为最后一个输入的通道数
            
        # 特殊处理TorchVision和Index模块
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]  # 输出通道数
            c1 = get_ch(f)  # 输入通道数
            args = [*args[1:]]  # 移除第一个参数
            
        # 默认情况：输出通道数等于输入通道数
        # m既不在base_modules中，也不是特殊模块
        else:
            c2 = get_ch(f)

        # 创建模块实例，如果重复次数大于1则使用Sequential包装
        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  
        # 创建模块
        # 解包的时候将被解包的对象里的元素作为独立参数传递；for循环时必须使用列表或者其他可迭代对象；while循环次数是长度（具体数值）
        t = str(m)[8:-2].replace("__main__.", "")  # 获取模块类型字符串
        m_.np = sum(x.numel() for x in m_.parameters())  # 计算参数数量
        m_.i, m_.f, m_.type = i, f, t  # 附加索引、'from'索引和类型信息
        
        # 如果需要详细输出，打印模块信息
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # 打印信息
            
        # 扩展保存列表，记录需要保存的层索引
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)  # 将模块添加到层列表

        # 如果是第一层，清空通道数列表（把初始化的ch里面的输入通道数删掉）
        if i == 0:
            ch = []
        if m is not BiFPN and m is not CEASC and m is not Sequential_BiFPN:  # BiFPN模块的输出通道数在模块内部定义，不需要添加到通道数列表
            ch.append(c2)  
        elif m is BiFPN:  # BiFPN模块的输出通道数在模块内部定义，需要添加到通道数列表
            for _ in range(5):
                ch.append(c2)
        else:
            for _ in range(3):  # 如果模块重复，添加n次输出通道数到通道数列表
                ch.append(c2)
        # 添加输出通道数到通道数列表
        # ch和c2都是locals里的变量，循环更新
        
        
    return torch.nn.Sequential(*layers), sorted(save)  # 返回模型序列和保存列表


def yaml_model_load(path):
    """
    created by ma

    Load  mymodel from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict，将yaml内容解析出来，生成字典
    
    # 只有当scale未定义时才猜测
    if "scale" not in d or not d["scale"]:
        d["scale"] = guess_model_scale(path)
    
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    created by ma

    从模型路径中提取模型尺寸字符 n, s, m, l, 或 x

    Args:
        model_path (str | Path): YOLO模型的YAML文件路径

    Returns:
        (str): 模型尺寸字符 (n, s, m, l, 或 x)
    """
    try:
        # 尝试匹配 model_0, model_1, ..., model_8 这样的格式
        match = re.search(r"model_(\d+)", Path(model_path).stem)
        if match:
            return ""
        # 如果没找到 model_N 格式，则尝试匹配 yolo 格式
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)  # noqa
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    从PyTorch模型的架构或配置中推测任务类型

    Args:
        model (torch.nn.Module | dict): PyTorch模型或YAML格式的模型配置

    Returns:
        (str): 模型的任务类型 ('detect', 'segment', 'classify', 'pose', 'obb')
    """

    def cfg2task(cfg):
        """从YAML字典推测任务"""
        m = cfg["head"][-1][-2].lower()  # 输出模块名称
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # 从模型配置推测
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # 从PyTorch模型推测
    if isinstance(model, torch.nn.Module):  # PyTorch模型
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            # 添加对MyModel的支持
            if m.__class__.__name__ == 'MyModel':
                return "detect"
            elif isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # 从模型文件名推测
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts or "my" in model.stem.lower():  # 添加对my model的支持
            return "detect"

    # 无法从模型确定任务
    LOGGER.warning(
        "无法自动猜测模型任务，假设 'task=detect'。"
        "为您的模型明确定义任务，例如 'task=detect', 'segment', 'classify','pose' 或 'obb'。"
    )
    return "detect"  # 假设检测任务
