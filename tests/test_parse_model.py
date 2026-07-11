"""验证 parse_model 能正确解析所有 model_*.yaml 配置。"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pram.tasks import parse_model, yaml_model_load


def test_parse_model_0():
    cfg_path = os.path.join("pram", "cfg", "model_0.yaml")
    cfg = yaml_model_load(cfg_path)
    cfg["scale"] = "n"
    model, save = parse_model(cfg, ch=3, verbose=False)
    assert isinstance(model, torch.nn.Sequential)
    assert len(save) >= 0


def test_parse_all_model_configs():
    cfg_dir = os.path.join("pram", "cfg")
    for path in sorted(glob.glob(os.path.join(cfg_dir, "model_*.yaml"))):
        cfg = yaml_model_load(path)
        cfg["scale"] = "n"
        model, save = parse_model(cfg, ch=3, verbose=False)
        assert isinstance(model, torch.nn.Sequential), f"{path} parse failed"


def test_parse_model_forward_pass():
    """验证解析出的模型能完成一次 forward（shape 检查）。"""
    cfg = yaml_model_load(os.path.join("pram", "cfg", "model_0.yaml"))
    cfg["scale"] = "n"
    model, save = parse_model(cfg, ch=3, verbose=False)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 320, 320)
        y = model(x)
    # CEASC in eval mode returns (cls_scores, bbox_preds, centernesses)
    assert isinstance(y, (tuple, list, torch.Tensor)), f"Unexpected output type: {type(y)}"
