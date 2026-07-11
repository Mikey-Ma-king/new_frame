"""自定义 YOLO 模型训练入口。

使用方式:
    python train.py                              # 默认参数训练
    python train.py --subset 10                  # 用 COCO128 前 10 张图片快速验证
    python train.py --cfg pram/cfg/model_1.yaml  # 切换模型变体
    python train.py --data datasets/coco128.yaml --epochs 200 --batch 8 --imgsz 640
"""

import os
import sys
import gc
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Custom YOLO training")
    p.add_argument("--cfg", default="pram/cfg/model_0.yaml", help="模型 YAML 配置路径")
    p.add_argument("--data", default="datasets/coco128.yaml", help="数据集 YAML 路径")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="mymodel_coco128")
    p.add_argument("--save_period", type=int, default=10)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--half", action="store_true", default=False)
    p.add_argument("--subset", type=int, default=0, help="使用 COCO128 前 N 张图片做子集训练 (0=不启用)")
    return p.parse_args()


def main():
    args = parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from module.registrar import register_custom_modules, unregister_custom_modules

    # 子集模式
    data_yaml = args.data
    if args.subset > 0:
        from module.data_utils import prepare_subset
        data_yaml = prepare_subset(num_images=args.subset)
        print(f"子集模式: {args.subset} 张图片, data={data_yaml}")

    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU训练")

    try:
        YOLO = register_custom_modules()
        print("自定义模块注册完成")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = YOLO(args.cfg)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            save_period=args.save_period,
            patience=args.patience,
            half=args.half,
        )
    finally:
        unregister_custom_modules()

    print("训练完成。")


if __name__ == "__main__":
    main()
