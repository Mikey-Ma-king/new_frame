"""统一的自定义模块注册/注销入口。

所有需要将 SPD_SCConv、BiFPN、CEASC 等自定义层注入 Ultralytics 框架的脚本，
统一通过本模块的 register_custom_modules() / unregister_custom_modules() 完成。
"""

import os
import sys
import importlib
import torch

from ultralytics import YOLO
from ultralytics.models.yolo import detect

from module.model_registrar import MyModelPredictor, MyModelTrainer, MyModelValidator, make_adapter

_saved_state = {}


def register_custom_modules():
    """注册所有自定义模块到 ultralytics 框架。

    返回 YOLO 类（已替换 __init__），以及 saved_state dict（用于 unregister）。
    """
    global _saved_state

    _ul_modules = None
    _ul_tasks = None

    try:
        import ultralytics.nn.modules as _ul_modules
        import ultralytics.nn.tasks as _ul_tasks
    except ImportError as e:
        raise ImportError("ultralytics 未安装或版本不兼容") from e

    from module.new_block import SPD_SCConv, DySample, SimAM_C3k2, BiFPN, Sequential_BiFPN
    from module.Head import CEASC

    _custom_classes = (SPD_SCConv, DySample, SimAM_C3k2, BiFPN, Sequential_BiFPN, CEASC)

    saved_attrs_modules = {}
    saved_attrs_tasks = {}

    for _cls in _custom_classes:
        name = _cls.__name__
        adapter = make_adapter(_cls)
        saved_attrs_modules[name] = getattr(_ul_modules, name, None)
        setattr(_ul_modules, name, adapter)
        saved_attrs_tasks[name] = getattr(_ul_tasks, name, None)
        setattr(_ul_tasks, name, adapter)

    # 替换 parse_model
    import pram.tasks as _pram_tasks
    saved_attrs_tasks["parse_model"] = getattr(_ul_tasks, "parse_model", None)
    setattr(_ul_tasks, "parse_model", _pram_tasks.parse_model)
    try:
        importlib.reload(_pram_tasks)
    except Exception:
        pass

    _saved_state = {
        "_ul_modules": _ul_modules,
        "_ul_tasks": _ul_tasks,
        "saved_attrs_modules": saved_attrs_modules,
        "saved_attrs_tasks": saved_attrs_tasks,
    }

    # 替换 YOLO.__init__
    from pram.tasks import MyModel
    _original_init = YOLO.__init__

    def _new_init(self, model="yolo11n.pt", task=None, verbose=False):
        if isinstance(model, str) and "model_" in model and model.endswith(".yaml"):
            self.ckpt = None
            self.cfg = model
            self.task = "detect"
            self.overrides = {"model": model}
            self.ModelClass = MyModel
            self.TrainerClass = MyModelTrainer
            self.ValidatorClass = MyModelValidator
            self.PredictorClass = MyModelPredictor
            torch.nn.Module.__init__(self)
            self.session = None
            self.callbacks = []
            self.model = MyModel(cfg=self.cfg, verbose=verbose)
            self.current_model = model
        else:
            _original_init(self, model, task, verbose)

    YOLO.__init__ = _new_init
    _saved_state["original_init"] = _original_init

    return YOLO


def unregister_custom_modules():
    """恢复 ultralytics 框架中的原始属性。"""
    global _saved_state
    if not _saved_state:
        return

    try:
        import ultralytics
        YOLO.__init__ = _saved_state.get("original_init", YOLO.__init__)
    except Exception:
        pass

    ulm = _saved_state.get("_ul_modules")
    ult = _saved_state.get("_ul_tasks")
    if ulm and ult:
        for name, val in _saved_state.get("saved_attrs_modules", {}).items():
            if val is None:
                try:
                    delattr(ulm, name)
                except Exception:
                    pass
            else:
                setattr(ulm, name, val)
        for name, val in _saved_state.get("saved_attrs_tasks", {}).items():
            if val is None:
                try:
                    delattr(ult, name)
                except Exception:
                    pass
            else:
                setattr(ult, name, val)

    _saved_state = {}
