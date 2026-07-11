"""自定义模型推理测试。

使用方式:
    python test.py --weights runs/train/mymodel_coco128/weights/best.pt --img test.jpg
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Custom YOLO inference")
    p.add_argument("--weights", required=True, help="训练好的权重文件路径 (.pt)")
    p.add_argument("--img", default="test.jpg", help="测试图片路径")
    p.add_argument("--imgsz", type=int, default=320, help="推理输入尺寸")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    p.add_argument("--cfg", default="pram/cfg/model_0.yaml", help="模型 YAML 配置")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_state_dict_from_best(best_path):
    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "ema" in ckpt and hasattr(ckpt["ema"], "state_dict"):
            return ckpt["ema"].state_dict()
        if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            return ckpt["model"].state_dict()
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]

    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()

    try:
        ckpt_wo = torch.load(best_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt_wo, dict):
            return ckpt_wo
    except Exception:
        pass

    raise RuntimeError("无法从 checkpoint 提取 state_dict")


def main():
    args = parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    os.environ.setdefault("CEASC_DEBUG_BBOX", "1")

    weight_path = args.weights
    img_path = args.img

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片不存在: {img_path}")

    # 注册自定义模块（不使用 YOLO monkey-patch，只用模块注入）
    from module.registrar import register_custom_modules, unregister_custom_modules
    try:
        register_custom_modules()
    except Exception:
        pass  # 注册部分成功即可，MyModel 已可用

    from pram.tasks import MyModel
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_coords
    from ultralytics.utils.plotting import Annotator
    from ultralytics.data.augment import LetterBox

    # 构建模型并加载权重
    model = MyModel(cfg=args.cfg, verbose=False)
    state_dict = load_state_dict_from_best(weight_path)
    load_ret = model.load_state_dict(state_dict, strict=False)
    try:
        missing = list(load_ret.missing_keys)
        unexpected = list(load_ret.unexpected_keys)
        print(f"权重加载: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing 前5个: {missing[:5]}")
        if unexpected:
            print(f"  unexpected 前5个: {unexpected[:5]}")
    except Exception:
        pass
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    # 加载并预处理图片
    im0 = cv2.imread(img_path)
    if im0 is None:
        raise FileNotFoundError(f"读取图片失败: {img_path}")

    lb = LetterBox(new_shape=args.imgsz, auto=False, stride=32)
    im = lb(image=im0)
    im = im.transpose((2, 0, 1))[::-1]  # BGR→RGB, HWC→CHW
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    im = im.unsqueeze(0).to(device)

    # 前向传播
    with torch.no_grad():
        preds = model(im)

    # NMS 处理
    if isinstance(preds, (list, tuple)) and hasattr(model.model[-1], "inference_batch"):
        preds = model.model[-1].inference_batch(
            preds[0], preds[1], preds[2], img_shape=im.shape[-2:]
        )

    if isinstance(preds, torch.Tensor) and preds.ndim == 3 and preds.shape[-1] >= 6:
        confs = preds[..., 4]
        print(f"预测张量形状: {tuple(preds.shape)}")
        if confs.numel() > 0:
            print(f"置信度: min={confs.min().item():.4f}, max={confs.max().item():.4f}, "
                  f"mean={confs.mean().item():.4f}")

    det = non_max_suppression(preds, conf_thres=args.conf, iou_thres=args.iou, max_det=300)[0]
    kept = 0 if det is None else len(det)
    print(f"NMS 保留框数 (conf={args.conf}, iou={args.iou}): {kept}")

    # 可视化保存
    save_dir = os.path.join("runs", "predict", "test")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(img_path))

    annotator = Annotator(im0.copy(), line_width=2)
    if det is not None and len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        det_np = det.cpu().numpy()
        for *xyxy, conf, cls in det_np:
            label = f"{int(cls)} {conf:.2f}"
            annotator.box_label(xyxy, label, color=(0, 255, 0))

    result_img = annotator.result()
    cv2.imwrite(save_path, result_img)
    print(f"结果保存至: {save_path}")


if __name__ == "__main__":
    main()
