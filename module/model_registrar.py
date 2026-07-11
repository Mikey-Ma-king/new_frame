"""自定义模型适配器：将 MyModel 接入 Ultralytics 训练/验证/预测流程。

make_adapter: 构造 Adapter 包装器，自动处理延迟实例化。
MyModelTrainer/Validator/Predictor: 继承 ultralytics 默认实现。
"""

import torch
from ultralytics.models.yolo import detect


class MyModelTrainer(detect.DetectionTrainer):
    """适配器：适配MyModel到训练流程"""
    pass


class MyModelValidator(detect.DetectionValidator):
    """适配器：适配MyModel到验证流程"""
    pass


class MyModelPredictor(detect.DetectionPredictor):
    """适配器：适配MyModel到预测流程"""
    pass


def make_adapter(cls):
    """为自定义层创建适配器包装器，兼容 ultralytics parse_model 的参数传递格式。

    支持三种构造路径：
    1. 直接构造（参数≥2个且匹配）
    2. forward 时从输入张量推断 in_channels 再构造
    3. 回退到原始参数构造
    """

    class Adapter(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.module = None
            self._delayed_args = None

            if len(args) >= 2:
                try:
                    self.module = cls(*args, **kwargs)
                except TypeError:
                    try:
                        self.module = cls(*args)
                    except Exception:
                        self.module = None
                        self._delayed_args = (args, kwargs)
            else:
                self._delayed_args = (args, kwargs)

        def forward(self, x, *a, **k):
            if self.module is None:
                args, kwargs = self._delayed_args
                try:
                    out_ch = args[0] if len(args) > 0 else kwargs.get("out_channels")
                    in_ch = int(x.shape[1])
                    rest = list(args[1:]) if len(args) > 1 else []
                    new_args = (in_ch, out_ch, *rest)
                    try:
                        self.module = cls(*new_args, **(kwargs or {}))
                    except TypeError:
                        try:
                            self.module = cls(*new_args)
                        except Exception:
                            self.module = cls(*args, **(kwargs or {}))
                except Exception:
                    self.module = cls(*args, **(kwargs or {}))

            return self.module(x, *a, **k)

    Adapter.__name__ = cls.__name__
    return Adapter
