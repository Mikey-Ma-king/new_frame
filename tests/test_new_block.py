"""验证自定义 Block 的输入输出 shape。"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.new_block import (
    SPD, SCConv, SCBottleneck, SPD_SCConv, DySample,
    SimAM, SimAM_C3k2, BiFPN, Sequential_BiFPN,
    SeparableConvBlock, Conv2dStaticSamePadding,
)


class TestSPD:
    def test_spd_scale2(self):
        x = torch.randn(2, 3, 64, 64)
        out = SPD(scale=2)(x)
        assert out.shape == (2, 12, 32, 32)

    def test_spd_scale4(self):
        x = torch.randn(2, 3, 64, 64)
        out = SPD(scale=4)(x)
        assert out.shape == (2, 48, 16, 16)


class TestSCConv:
    def test_forward_shape(self):
        x = torch.randn(1, 32, 40, 40)
        m = SCConv(32, 64, group=1)
        out = m(x)
        assert out.shape == (1, 64, 40, 40)


class TestSCBottleneck:
    def test_forward_shape(self):
        x = torch.randn(1, 64, 40, 40)
        m = SCBottleneck(64, 64, group=32)
        out = m(x)
        assert out.shape == (1, 64, 40, 40)


class TestSPD_SCConv:
    def test_forward_shape(self):
        x = torch.randn(1, 16, 80, 80)
        m = SPD_SCConv(16, 32, scale=2, group=32)
        out = m(x)
        assert out.shape == (1, 32, 80, 80)


class TestDySample:
    def test_lp_style(self):
        x = torch.randn(1, 64, 32, 32)
        m = DySample(64, scale=2, style="lp", groups=4)
        out = m(x)
        assert out.shape == (1, 64, 64, 64)

    def test_pl_style(self):
        x = torch.randn(1, 64, 32, 32)
        m = DySample(64, scale=2, style="pl", groups=4)
        out = m(x)
        assert out.shape == (1, 64, 64, 64)


class TestSimAM:
    def test_forward_shape(self):
        x = torch.randn(1, 64, 32, 32)
        out = SimAM()(x)
        assert out.shape == x.shape


class TestBiFPN:
    def test_multi_scale(self):
        inputs = (
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 64, 40, 40),
            torch.randn(1, 64, 20, 20),
            torch.randn(1, 64, 10, 10),
            torch.randn(1, 64, 5, 5),
        )
        m = BiFPN(num_channels=64, conv_channels=(64, 64, 64, 64, 64))
        outs = m(inputs)
        assert len(outs) == 5
        for i, o in enumerate(outs):
            assert o.shape == inputs[i].shape


class TestSequentialBiFPN:
    def test_default_output(self):
        inputs = (
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 64, 40, 40),
            torch.randn(1, 64, 20, 20),
            torch.randn(1, 64, 10, 10),
            torch.randn(1, 64, 5, 5),
        )
        m = Sequential_BiFPN(num_channels=64, num_layers=2,
                              conv_channels=(64, 64, 64, 64, 64))
        outs = m(inputs)
        assert len(outs) == 3  # default 3 outputs


class TestConv2dStaticSamePadding:
    def test_same_padding(self):
        x = torch.randn(1, 3, 32, 32)
        m = Conv2dStaticSamePadding(3, 16, kernel_size=3, stride=1)
        out = m(x)
        assert out.shape[2:] == (32, 32)

    def test_stride2(self):
        x = torch.randn(1, 3, 32, 32)
        m = Conv2dStaticSamePadding(3, 16, kernel_size=3, stride=2)
        out = m(x)
        assert out.shape[2:] == (16, 16)


class TestSeparableConvBlock:
    def test_forward_shape(self):
        x = torch.randn(1, 32, 40, 40)
        m = SeparableConvBlock(32, 64)
        out = m(x)
        assert out.shape == (1, 64, 40, 40)
