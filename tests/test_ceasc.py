"""验证 CEASC 检测头的 forward / loss / inference shape。"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.Head import CEASC


def make_ceasc(num_classes=80, strides=(8, 16, 32)):
    return CEASC(
        num_classes=num_classes,
        in_channels=256,
        feat_channels=256,
        strides=list(strides),
        amm_cfg=dict(target_ar=0.3),
    )


def make_dummy_feats(B=2, channels=256):
    return [
        torch.randn(B, channels, 80, 80),  # stride 8
        torch.randn(B, channels, 40, 40),  # stride 16
        torch.randn(B, channels, 20, 20),  # stride 32
    ]


class TestCEASCForward:
    def test_train_mode_outputs(self):
        head = make_ceasc().train()
        feats = make_dummy_feats()
        cls_scores, bbox_preds, centernesses, aux_loss = head(feats)
        assert len(cls_scores) == 3
        assert cls_scores[0].shape == (2, 80, 80, 80)
        assert bbox_preds[0].shape == (2, 4, 80, 80)
        assert centernesses[0].shape == (2, 1, 80, 80)
        assert aux_loss.ndim == 0

    def test_eval_mode_outputs(self):
        head = make_ceasc().eval()
        feats = make_dummy_feats()
        cls_scores, bbox_preds, centernesses = head(feats)
        assert len(cls_scores) == 3

    def test_aux_loss_participates(self):
        head = make_ceasc().train()
        feats = make_dummy_feats()
        _, _, _, aux = head(feats)
        assert aux.item() >= 0


class TestCEASCLoss:
    def test_compute_loss_with_gt(self):
        head = make_ceasc().train()
        feats = make_dummy_feats()
        cls_scores, bbox_preds, centernesses, aux = head(feats)

        # GT: 模拟 batch 中的框 (xywh 归一化)
        gt_bboxes = [
            torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.15, 0.15]], dtype=torch.float32),
            torch.tensor([[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32),
        ]
        gt_labels = [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        ]

        losses = head.compute_loss(
            cls_scores, bbox_preds, centernesses, aux,
            gt_bboxes, gt_labels, img_size=(640, 640),
        )
        assert "loss_cls" in losses
        assert "loss_reg" in losses
        assert "loss_ctr" in losses
        assert "loss_aux" in losses
        assert losses["loss_cls"].item() >= 0

    def test_compute_loss_no_gt(self):
        head = make_ceasc().train()
        feats = make_dummy_feats()
        cls_scores, bbox_preds, centernesses, aux = head(feats)

        gt_bboxes = [torch.empty(0, 4), torch.empty(0, 4)]
        gt_labels = [torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)]

        losses = head.compute_loss(
            cls_scores, bbox_preds, centernesses, aux,
            gt_bboxes, gt_labels, img_size=(640, 640),
        )
        assert losses["loss_cls"].item() >= 0


class TestCEASCInference:
    def test_inference_returns_detections(self):
        head = make_ceasc().eval()
        feats = make_dummy_feats(B=1)
        det = head.inference(feats, img_size=(640, 640), score_thr=0.1)
        assert det.ndim == 1 or det.ndim == 2  # [N, 6] or [0] empty

    def test_inference_batch(self):
        head = make_ceasc().eval()
        feats = make_dummy_feats(B=2)
        cls_scores, bbox_preds, centernesses = head(feats)
        results = head.inference_batch(
            cls_scores, bbox_preds, centernesses,
            img_shape=(640, 640), score_thr=0.1,
        )
        assert isinstance(results, list)
        assert len(results) == 2
