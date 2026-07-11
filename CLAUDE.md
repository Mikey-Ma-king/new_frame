## 项目概述
- 自定义 YOLO 目标检测框架，基于 Ultralytics，三层定制：SPD_SCConv backbone + Sequential_BiFPN neck + CEASC (FCOS + 稀疏卷积) head
- 所有自定义模块通过运行时 monkey-patch 注入 Ultralytics（`module/registrar.py`），不修改 Ultralytics 源码

## Python 环境
- ultralytics / torch 在 conda 环境，WSL 系统 Python 3.12 无此依赖
- 运行训练/推理前需激活 conda 环境，否则 `import ultralytics` 失败
- pytest 未安装在系统 Python，需在 conda 环境中运行测试

## 关键架构约定
- MyModel.loss() 通过 isinstance(m, CEASC) 检测头类型 → CEASC 走 FCOS compute_loss，非 CEASC 走 v8DetectionLoss
- make_adapter() 延迟实例化：forward 时从输入张量推断 in_channels，避免 parse_model 阶段的通道数歧义
- parse_model 中自定义模块判断用类名字符串（__name__），不用 `is` 身份比较（adapter 包装后身份会变）

## 模型配置
- model_*.yaml 在 pram/cfg/，全部用 scale:n（depth=0.50, width=0.25）
- model_6/7/8 当前 BiFPN 配置完全相同（8 层 384 通道），为占位状态
- yolo11.yaml 是 YOLO11 参考配置（使用自定义 SPD_SCConv Block）

## 权重与数据集
- yolo11n.pt 不能直接用于训练/推理（结构与本项目完全不兼容），必须从头训练
- 数据集需 YOLO 格式标注，COCO128 下载：wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
- --subset N 从 COCO128 取前 N 张自动生成子集

## 稀疏卷积 CUDA 扩展
- Sparse_conv/ 是 CUDA C++ 扩展，编译产物是 .pyd (Windows DLL)
- WSL/Linux 下需重新编译：cd Sparse_conv && python setup.py build_ext --inplace
- 训练走纯 Python _slow_forward 回退，不依赖 .pyd；推理才需要 CUDA 加速
- module/my_sparse_conv_cpu.cp311-win_amd64.pyd 是 CPU 版本，代码已断言 not cpu 未使用

## 无用文件
- CEASC-main/ 是原始 MMDetection 参考实现，本项目不依赖
- module/*.pyd 和 Sparse_conv/*.pyd 有重复（同一 CUDA 扩展的两个副本）

## 代码风格
- 注释用中文，模块 docstring 用中文
- 自定义 Block 在 module/new_block.py，检测头在 module/Head.py
- 测试覆盖形状验证（不覆盖完整训练），在 tests/ 目录
