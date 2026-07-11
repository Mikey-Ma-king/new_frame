# New Frame — 自定义 YOLO 目标检测框架

基于 Ultralytics YOLO 深度定制的目标检测研究项目，使用自定义 SPD-SCConv backbone + BiFPN neck + CEASC 稀疏卷积检测头。

## 核心特性

- **SPD-SCConv Backbone**：空间到深度 + 自校准卷积，保留小目标细粒度信息
- **SimAM 注意力**：无参数 3D 注意力，集成在 C3k2 模块中
- **DySample 上采样**：可学习位置偏移的动态上采样（LP / PL 两种风格）
- **BiFPN Neck**：加权双向特征金字塔，支持多层堆叠（Sequential_BiFPN）
- **CEASC 检测头**：基于 FCOS 的 anchor-free 头，含自适应稀疏卷积（AMM + CESC + CEGN）
- **稀疏卷积 CUDA 扩展**：仅对被掩码激活的区域做卷积，降低推理计算量
- **9 个模型变体**：`model_0` 到 `model_8`，逐步增大 BiFPN 层数和通道数

## 技术栈

| 组件 | 说明 |
|------|------|
| 语言 | Python 3.8+ |
| 框架 | PyTorch + Ultralytics |
| 计算 | CUDA（推荐），CPU 可回退 |
| 稀疏卷积 | 自定义 CUDA C++ 扩展 |
| 数据格式 | YOLO 格式标注 |
| 测试 | pytest |

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics opencv-python torch torchvision numpy
```

### 2. 下载数据集

**COCO128**（128 张图，约 7 MB）：

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip -P datasets/
unzip datasets/coco128.zip -d datasets/
rm datasets/coco128.zip
```

目录结构：
```
datasets/coco128/
├── images/train2017/   # 128 张训练图
├── labels/train2017/   # YOLO 格式标注
├── LICENSE
└── README.txt
```

**COCO 完整版**（11.8 万张图）见 [Ultralytics 文档](https://docs.ultralytics.com/datasets/detect/coco/)。

**自定义数据集**按以下结构组织：
```
datasets/your_data/
├── images/train/
├── labels/train/       # 每张图一个 .txt: "cls x y w h"（归一化）
└── data.yaml
```

### 3. 下载预训练权重（可选）

> **注意**：`yolo11n.pt` **不能直接用作**本项目的训练或推理权重。YOLO11n 的 backbone（Conv+C3k2）、neck（PANet）、head（Detect）与本项目结构完全不兼容，只有 SPPF、C2PSA 等少数层能对上。本项目需**从头训练**。

仅用于权重提取工具：

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

### 4. 快速冒烟训练（10 张图验证流程）

```bash
python train.py --subset 10 --epochs 5 --batch 1
```

从 COCO128 取前 10 张做快速验证，确认整条 pipeline 无报错。

### 5. 正式训练

```bash
# 默认配置（model_0.yaml + COCO128 + 100 轮）
python train.py

# 自定义参数
python train.py \
  --cfg pram/cfg/model_1.yaml \
  --data datasets/coco128.yaml \
  --epochs 200 \
  --batch 8 \
  --imgsz 640
```

训练输出（checkpoint、日志、指标）保存在 `runs/train/<name>/`。

### 6. 推理

```bash
python test.py \
  --weights runs/train/mymodel_coco128/weights/best.pt \
  --img test.jpg \
  --imgsz 640 \
  --conf 0.25
```

结果保存在 `runs/predict/test/`。

## 项目结构

```
new_frame/
├── train.py                  # 训练入口
├── test.py                   # 推理入口
├── train_analysis.ipynb      # 训练分析 notebook
├── module/                   # 自定义网络模块
│   ├── new_block.py          # 核心 Block（SPD, SCConv, DySample, BiFPN 等）
│   ├── Head.py               # CEASC 检测头（FCOS + 稀疏卷积）
│   ├── registrar.py          # 统一注册/注销入口
│   ├── model_registrar.py    # 适配器类（Trainer/Validator/Predictor + make_adapter）
│   └── data_utils.py         # 数据集子集生成工具
├── pram/                     # 模型参数与任务定义
│   ├── tasks.py              # MyModel + parse_model() + YAML 加载
│   └── cfg/                  # 模型架构 YAML 配置
│       ├── model_0.yaml      # BiFPN=1 层, 通道=64（最小）
│       ├── model_1.yaml      # BiFPN=4 层, 通道=88
│       ├── model_2.yaml      # BiFPN=5 层, 通道=112
│       ├── model_3.yaml      # BiFPN=6 层, 通道=160
│       ├── model_4.yaml      # BiFPN=7 层, 通道=224
│       ├── model_5.yaml      # BiFPN=7 层, 通道=288
│       ├── model_6.yaml      # BiFPN=8 层, 通道=384
│       ├── model_7.yaml      # BiFPN=8 层, 通道=384
│       ├── model_8.yaml      # BiFPN=8 层, 通道=384
│       └── yolo11.yaml       # YOLO11 基线（使用自定义 Block）
├── cfg/
│   └── default.yaml          # Ultralytics 超参数配置
├── datasets/
│   ├── cococo128.yaml         # COCO128 数据集定义
│   ├── coco128_10.yaml       # 自动生成的 10 图子集配置
│   └── coco128/              # 数据集图片与标注
├── tests/
│   ├── test_new_block.py     # 自定义 Block 形状验证
│   ├── test_ceasc.py         # CEASC 头测试（前向、loss、推理）
│   └── test_parse_model.py   # YAML 解析测试
├── tools/
│   └── weight_extract.py     # 从 YOLO checkpoint 提取 backbone/PANet 权重
├── Sparse_conv/              # CUDA 稀疏卷积扩展源码
│   ├── setup.py
│   ├── sparse_conv_cuda.cpp
│   └── sparse_conv_cuda_kernel.cu
├── CEASC-main/               # 原始 CEASC 参考实现（MMDetection 框架，仅供参考）
├── yolo11n.pt                # YOLO11n 预训练权重
└── .gitignore
```

## 模型架构流程

```
输入图片 (3, H, W)
       │
       ▼
┌─────────────────────────────────┐
│  SPD-SCConv Backbone            │
│  SPD_SCConv → SimAM_C3k2  (×4) │  P1/2 → P2/4 → P3/8 → P4/16 → P5/32
│  SPPF → C2PSA                  │  P5 增强
└─────────────────────────────────┘
       │ 5 个尺度特征 (P2~P5 + 增强)
       ▼
┌─────────────────────────────────┐
│  Sequential BiFPN Neck          │
│  BiFPN × N 层（加权双向融合）     │
│  输出 3 个尺度: P3/8, P4/16, P5/32│
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  CEASC 检测头                   │
│  AMM → CESC → 卷积栈 (分类分支)  │
│  AMM → CESC → 卷积栈 (回归分支)  │
│  FCOS 风格: ltrb + centerness   │
└─────────────────────────────────┘
       │
       ▼
检测结果: [x1, y1, x2, y2, conf, cls]
```

## 训练流程（请求链路）

1. `train.py` 解析 CLI 参数，调用 `register_custom_modules()`
2. 自定义层类注入到 `ultralytics.nn.modules` 和 `ultralytics.nn.tasks`
3. `YOLO.__init__` 被 monkey-patch，使 `model_*.yaml` 路由到 `MyModel`
4. `YOLO(cfg_path)` 创建 `MyModel` → `parse_model()` 解析 YAML → `torch.nn.Sequential`
5. `model.train()` 启动 Ultralytics 训练循环
6. 训练中 `MyModel.loss()` 将 CEASC 路由到 FCOS loss，其他模型走标准 v8DetectionLoss

## 推理流程

1. `test.py` 注册自定义模块，手动创建 `MyModel`
2. 加载 `.pt` checkpoint，自动适配多种格式（ema / model / state_dict）
3. LetterBox 预处理 + 归一化
4. backbone → BiFPN → CEASC head 前向
5. `inference_batch()` 将 FCOS 输出转为边界框
6. `non_max_suppression()` + `Annotator` 可视化保存

## 模型变体

| 配置 | BiFPN 层数 | BiFPN 通道 | 说明 |
|------|:---------:|:---------:|------|
| model_0 | 1 | 64 | 最轻量 |
| model_1 | 4 | 88 | |
| model_2 | 5 | 112 | |
| model_3 | 6 | 160 | |
| model_4 | 7 | 224 | |
| model_5 | 7 | 288 | |
| model_6 | 8 | 384 | 与 model_7/8 YAML 配置相同 |
| model_7 | 8 | 384 | 与 model_6/8 YAML 配置相同 |
| model_8 | 8 | 384 | 与 model_6/7 YAML 配置相同 |

> model_6/7/8 的 BiFPN 配置完全相同（8 层 384 通道），当前为占位，可按需修改为不同参数做对比消融实验。

全部变体共用 `scale: n`（depth=0.50, width=0.25, max_channels=1024），可通过修改 YAML 中的 `scale` 切换 n/s/m/l/x。

## 命令速查

| 命令 | 说明 |
|------|------|
| `python train.py` | 默认 model_0 + COCO128 训练 |
| `python train.py --cfg pram/cfg/model_1.yaml` | 切换模型变体 |
| `python train.py --subset 10 --epochs 5` | 10 图冒烟验证 |
| `python train.py --data <path> --epochs 200 --batch 8 --imgsz 640` | 完整自定义训练 |
| `python test.py --weights <path> --img <path>` | 单图推理 |
| `python tools/weight_extract.py --weights yolo11n.pt` | 提取 backbone/PANet 权重 |
| `python -m pytest tests/ -v` | 运行全部测试 |

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfg` | `pram/cfg/model_0.yaml` | 模型 YAML 路径 |
| `--data` | `datasets/coco128.yaml` | 数据集 YAML 路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch` | 1 | 批次大小 |
| `--imgsz` | 320 | 输入图片尺寸 |
| `--device` | `0`(GPU) / `cpu` | 训练设备 |
| `--workers` | 0 | DataLoader 进程数 |
| `--project` | `runs/train` | 输出根目录 |
| `--name` | `mymodel_coco128` | 实验名称 |
| `--save_period` | 10 | 每隔 N epoch 保存 checkpoint |
| `--patience` | 50 | 早停耐心值 |
| `--half` | False | FP16 混合精度 |
| `--subset` | 0 | 取 COCO128 前 N 张快速测试（0=禁用） |

## 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weights` | 必填 | 训练好的权重路径 (.pt) |
| `--img` | `test.jpg` | 输入图片路径 |
| `--imgsz` | 320 | 推理输入尺寸 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.5 | NMS IoU 阈值 |
| `--cfg` | `pram/cfg/model_0.yaml` | 模型 YAML（需与权重匹配） |
| `--device` | 自动 | 推理设备 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch CUDA 内存分配器 | `expandable_segments:True` |

## 测试

```bash
# 全部测试
python -m pytest tests/ -v

# 单独运行
python -m pytest tests/test_new_block.py -v
python -m pytest tests/test_ceasc.py -v
python -m pytest tests/test_parse_model.py -v
```

覆盖范围：
- **test_new_block.py**：SPD、SCConv、DySample、BiFPN 等全部自定义 Block 的输入输出形状
- **test_ceasc.py**：CEASC 前向（训练/推理模式）、FCOS loss（有/无 GT）、单图/批量推理
- **test_parse_model.py**：全部 `model_*.yaml` 可解析为 `torch.nn.Sequential` 并完成一次 forward

## 常见问题

### 稀疏卷积导入失败

**报错**：`ImportError: 请确保 sparse_conv 扩展已正确编译`

训练和 CPU 推理会走纯 Python 回退路径（`_slow_forward`），不依赖此扩展。如需 GPU 推理加速，编译 CUDA 扩展：

```bash
cd Sparse_conv
python setup.py build_ext --inplace
```

### CUDA 显存不足

1. 减小 batch：`--batch 1`
2. 减小输入尺寸：`--imgsz 320`
3. 启用半精度：`--half`
4. 环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已默认设置

### Ultralytics 版本兼容

自定义模块注册（`module/registrar.py`）通过 monkey-patch 注入 Ultralytics 内部。升级 `ultralytics` 后如遇导入错误，可能需要同步更新注册代码。

## 非项目文件（仅供参考/备用）

以下文件**不在当前项目中实际使用**：

| 路径 | 说明 | 状态 |
|------|------|:---:|
| `yolo11n.pt` | YOLO11n 预训练权重，结构不兼容 | 仅供 `tools/weight_extract.py` 提取 |
| `CEASC-main/` | 原始 CEASC 参考实现（MMDetection） | 本项目已用 Ultralytics 重写，**不依赖它** |
| `Sparse_conv/sparse_conv.cp38-win_amd64.pyd` | CUDA 稀疏卷积（Py3.8/Win） | 与 `module/` 下同名文件重复 |
| `module/my_sparse_conv_cpu.cp311-win_amd64.pyd` | CPU 稀疏卷积（Py3.11/Win） | 代码已断言 `not cpu`，未被调用 |
| `datasets/coco128_10/` | `--subset 10` 自动生成 | 每次运行重新生成，可随时删除 |
| `cfg/default.yaml` | Ultralytics 全局配置模板 | 训练时自动加载，无需手动修改 |
| `CEASC-main/demo/` | 原始仓库演示素材 | MMDetection 框架下的 demo，本项目用 `test.py` |

清理命令：
```bash
rm -rf CEASC-main/ datasets/coco128_10/ \
       Sparse_conv/sparse_conv.cp38-win_amd64.pyd \
       module/my_sparse_conv_cpu.cp311-win_amd64.pyd
```
