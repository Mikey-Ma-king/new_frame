# train_mymodel.py
import os
# Ensure PyTorch CUDA allocator config is set before importing torch to avoid
# allocator fragmentation / non-effect of env var when torch already initialized.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import gc
import torch

def register_and_train():
    # 添加项目路径
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)

    # 直接执行注册逻辑
    from ultralytics import YOLO
    from ultralytics.models.yolo import detect
    from module.model_registrar import MyModelPredictor,MyModelTrainer,MyModelValidator

    # （已移至训练时临时注册）自定义层的注册在创建模型并训练前进行，训练后恢复原始属性。

    # NOTE: 我们在完成下面的临时注册（替换 parse_model / 注入 adapter）后
    # 再导入本地 `pram.tasks.MyModel` 并替换 `YOLO.__init__`，以确保本地
    # `parse_model` 在模型构建时可用（避免导入时绑定老的引用）。
    
    print("开始训练MyModel...")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 0  # 使用GPU 0
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'  # 使用CPU
        print("使用CPU进行训练")
    
    # 临时注册自定义层到 ultralytics 的解析上下文（modules 与 tasks），训练结束后恢复原始属性
    _registered = False
    _saved_attrs_modules = {}
    _saved_attrs_tasks = {}
    # ensure these exist in outer scope for finally() restore
    _ul_modules = None
    _ul_tasks = None
    try:
        import ultralytics.nn.modules as _ul_modules
        import ultralytics.nn.tasks as _ul_tasks
        from module.new_block import SPD_SCConv, DySample, SimAM_C3k2, BiFPN, Sequential_BiFPN
        from module.Head import CEASC
        _custom_classes = (SPD_SCConv, DySample, SimAM_C3k2, BiFPN, Sequential_BiFPN, CEASC)

        from module.model_registrar import make_adapter

        

        for _cls in _custom_classes:
            name = _cls.__name__
            adapter = make_adapter(_cls)
            _saved_attrs_modules[name] = getattr(_ul_modules, name, None)
            setattr(_ul_modules, name, adapter)
            # some parsing routines lookup classes in ultralytics.nn.tasks globals()
            _saved_attrs_tasks[name] = getattr(_ul_tasks, name, None)
            setattr(_ul_tasks, name, adapter)
        # Replace ultralytics' parse_model with our local fixed version if available
        try:
            import pram.tasks as _pram_tasks
            # 保存原始 parse_model 引用，替换为本地实现
            _saved_attrs_tasks['parse_model'] = getattr(_ul_tasks, 'parse_model', None)
            setattr(_ul_tasks, 'parse_model', _pram_tasks.parse_model)
            # reload pram.tasks to ensure any module-level bindings are fresh (safe no-op if not needed)
            try:
                import importlib
                importlib.reload(_pram_tasks)
            except Exception:
                pass
            # 现在导入本地 MyModel（在替换 parse_model / 注入 adapter 之后导入）
            from pram.tasks import MyModel
            print("已使用本地 pram.tasks.parse_model 替换 ultralytics.nn.tasks.parse_model ->",
                  getattr(_ul_tasks, 'parse_model').__module__,
                  getattr(_ul_tasks, 'parse_model').__name__)
        except Exception:
            pass
        _registered = True
    except Exception:
        _registered = False

    try:
        # 在本地实现已注册后，再替换 YOLO.__init__ 以引用本地 MyModel
        try:
            original_init = YOLO.__init__

            def new_init(self, model='yolo11n.pt', task=None, verbose=False):
                """新的初始化方法，支持MyModel"""
                if isinstance(model, str) and 'model_' in model and model.endswith('.yaml'):
                    # 识别为MyModel配置
                    self.ckpt = None
                    self.cfg = model
                    self.task = 'detect'
                    # 保存 model 字段，Ultralytics.train() 会访问 self.overrides['model']
                    self.overrides = {"model": model}
                    self.ModelClass = MyModel
                    self.TrainerClass = MyModelTrainer
                    self.ValidatorClass = MyModelValidator
                    self.PredictorClass = MyModelPredictor

                    # 关键：先初始化 Module 基类，才能把子模块赋给 self
                    torch.nn.Module.__init__(self)

                    # 确保 wrapper 对象包含 session，避免属性访问被委托到 MyModel
                    self.session = None

                    # 避免 ultralytics.train() 访问不存在的 callbacks 导致 AttributeError
                    self.callbacks = []

                    self.model = MyModel(cfg=self.cfg, verbose=verbose)
                    self.current_model = model
                    print(f"MyModel已加载: {model}")
                else:
                    # 其他模型使用原始初始化
                    original_init(self, model, task, verbose)

            YOLO.__init__ = new_init
            print("MyModel 注册并替换 YOLO.__init__ 完成！")
        except Exception:
            # 如果替换失败，继续尝试创建模型（可能会使用原始 YOLO 行为）
            pass

        # 加载模型（使用你的自定义模型）
        # 清理并打印显存摘要，帮助诊断 OOM
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print('显存摘要（模型加载前）:')
                print(torch.cuda.memory_summary())
        except Exception:
            pass

        model = YOLO('pram/cfg/model_0.yaml')

        # 清理 GPU 缓存以尽量减少内存碎片
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 开始训练（使用更保守的资源配置以避免 OOM）
        # 建议：如果仍然 OOM，请进一步降低 batch 或 imgsz，或使用 workers=0
        try:
            # 再次清理并打印显存摘要（训练前）
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print('显存摘要（训练前）:')
                    print(torch.cuda.memory_summary())
            except Exception:
                pass

            model.train(
            data='datasets/coco128.yaml',  # COCO128数据集
            epochs=100,
            imgsz=320,       # 降低输入尺寸减少显存占用
            batch=1,         # 减小 batch
            device=device,  # 自动选择设备
            project='runs/train',
            name='mymodel_coco128',
            save_period=10,
            patience=50,
            workers=0,      # 在 Windows 上调试时建议禁用多进程 dataloader
            half=True       # 尝试启用半精度训练以降低显存
        )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('捕获到 OOM错误,建议进一步降低 batch/imgsz 或在命令行设置 PYTORCH_CUDA_ALLOC_CONF 后重试')
            raise
    finally:
        # 恢复 ultralytics 中的原始属性，保持原子性
        if _registered:
            try:
                for name, val in _saved_attrs_modules.items():
                    if val is None:
                        if hasattr(_ul_modules, name):
                            delattr(_ul_modules, name)
                    else:
                        setattr(_ul_modules, name, val)
                for name, val in _saved_attrs_tasks.items():
                    if val is None:
                        if hasattr(_ul_tasks, name):
                            delattr(_ul_tasks, name)
                    else:
                        setattr(_ul_tasks, name, val)
            except Exception:
                pass
    
    print("训练完成！")

if __name__ == "__main__":
    register_and_train()