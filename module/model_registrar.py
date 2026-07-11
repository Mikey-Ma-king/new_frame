# register_mymodel.py
import sys
import os
import torch

# 添加项目路径
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

# 导入你的自定义模型
from pram.tasks import MyModel
from ultralytics import YOLO
from ultralytics.models.yolo import detect

# 定义适配器类 - 可以空定义，因为继承自基类
class MyModelTrainer(detect.DetectionTrainer):
    """适配器：适配MyModel到训练流程"""
    pass

class MyModelValidator(detect.DetectionValidator):
    """适配器：适配MyModel到验证流程"""
    pass

class MyModelPredictor(detect.DetectionPredictor):
    """适配器：适配MyModel到预测流程"""
    pass


# 适配器：封装自定义类，使其能够接收 parse_model 函数常用的位置参数
def make_adapter(cls):
    """
    为自定义层创建适配器包装器
    适配器的主要功能：
    1. 接受parse_model传递的参数格式
    2. 支持延迟实例化（在forward时才确定输入通道数）
    3. 兼容不同的自定义层构造函数签名
    """
    class Adapter(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # 模块实例，在forward时才会真正创建
            self.module = None
            # 延迟的参数，在forward时使用
            self._delayed_args = None
            
            # 如果parse_model传递了多个位置参数，首先尝试直接构造
            if len(args) >= 2:
                try:
                    # 尝试使用所有参数直接构造
                    self.module = cls(*args, **kwargs)
                except TypeError:
                    try:
                        # 如果失败，尝试不使用关键字参数构造
                        self.module = cls(*args)
                    except Exception:
                        # 如果还是失败，将参数存储起来，在forward时再尝试构造
                        self.module = None
                        self._delayed_args = (args, kwargs)
            else:
                # 如果只有一个参数（通常是out_channels），则延迟实例化
                self._delayed_args = (args, kwargs)

        def forward(self, x, *a, **k):
            """
            前向传播方法
            x: 输入张量
            这里是适配器的核心：在forward时根据输入张量动态确定输入通道数
            """
            if self.module is None:
                # 延迟实例化：从输入张量推断输入通道数
                args, kwargs = self._delayed_args
                
                try:
                    # 获取输出通道数
                    out_ch = args[0] if len(args) > 0 else kwargs.get('out_channels')
                    # 从输入张量动态推断输入通道数
                    in_ch = int(x.shape[1])
                    # 获取其他参数
                    rest = list(args[1:]) if len(args) > 1 else []
                    # 构造新的参数元组 (in_ch, out_ch, *rest)
                    new_args = (in_ch, out_ch, *rest)
                    
                    try:
                        # 尝试使用推断的输入通道数构造模块
                        self.module = cls(*new_args, **(kwargs or {}))
                    except TypeError:
                        try:
                            # 如果失败，尝试不使用关键字参数构造
                            self.module = cls(*new_args)
                        except Exception:
                            # 最后的备选方案：使用原始参数构造
                            self.module = cls(*args, **(kwargs or {}))
                except Exception:
                    # 如果所有尝试都失败，使用原始参数构造
                    self.module = cls(*args, **(kwargs or {}))
                    
            # 执行前向传播
            return self.module(x, *a, **k)

    # 保持与原始类相同的名称
    Adapter.__name__ = cls.__name__
    return Adapter