"""一次性工具：从 Ultralytics 预训练权重中提取 backbone / PANet 部分。

使用方式:
    python tools/weight_extract.py --weights path/to/yolo11n.pt --output-dir weights/
"""

import argparse
import os
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Extract partial weights from YOLO checkpoint")
    p.add_argument("--weights", required=True, help="输入权重文件路径 (.pt)")
    p.add_argument("--output-dir", default="weights", help="输出目录")
    return p.parse_args()


def print_state_dict(state_dict, max_items=None):
    if isinstance(state_dict, (dict,)):
        items = state_dict.items()
        items_to_print = list(items)[:max_items] if max_items else items
        for k, v in items_to_print:
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {v}")
    else:
        print(state_dict)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    weights = torch.load(args.weights, map_location="cpu", weights_only=False)
    weights_model = weights["model"] if isinstance(weights, dict) else weights
    full_state_dict = weights_model.state_dict()

    # Backbone: layers 0-10
    prefixes_backbone = [f"model.{i}." for i in range(0, 11)]
    backbone_sd = {k: v for k, v in full_state_dict.items()
                   if any(k.startswith(p) for p in prefixes_backbone)}
    backbone_path = os.path.join(args.output_dir, "yolo11n-backbone.pt")
    torch.save(backbone_sd, backbone_path)
    print(f"Backbone 权重已保存: {backbone_path} ({len(backbone_sd)} 个键)")

    # PANet: layers 0-22
    prefixes_panet = [f"model.{i}." for i in range(0, 23)]
    panet_sd = {k: v for k, v in full_state_dict.items()
                if any(k.startswith(p) for p in prefixes_panet)}
    panet_path = os.path.join(args.output_dir, "yolo11n-PANet.pt")
    torch.save(panet_sd, panet_path)
    print(f"PANet 权重已保存: {panet_path} ({len(panet_sd)} 个键)")

    # Full: layers 0-23
    prefixes_full = [f"model.{i}." for i in range(0, 24)]
    full_sd = {k: v for k, v in full_state_dict.items()
               if any(k.startswith(p) for p in prefixes_full)}
    full_path = os.path.join(args.output_dir, "yolo11n-Full.pt")
    torch.save(full_sd, full_path)
    print(f"Full 权重已保存: {full_path} ({len(full_sd)} 个键)")


if __name__ == "__main__":
    main()
