# CLAUDE[.md](http://CONTEXT.md)

## Project Overview

Project Name:  
Fundus Lesion Segmentation using U-Net

Goal:  
Train a semantic segmentation model for retinal lesion segmentation on fundus images.

Current baseline model:  
U-Net

Framework:  
PyTorch

Development workflow:

- Local development: MacBook Air M4 + VS Code
- Remote training: AutoDL server (A100-PCIE-40GB × 1)
- SSH connected to VS Code

---

## Environment & Infrastructure

### Local Machine

| 项目 | 路径 |
|------|------|
| 项目根目录 | `/Users/sonder/Desktop/fundus-lesion-segmentation` |
| 数据集根目录 | `/Users/sonder/Desktop/fundus-lesion-segmentation/dataset/Segmentation` |
| outputs 目录 | `/Users/sonder/Desktop/fundus-lesion-segmentation/outputs` |

### AutoDL 服务器

| 项目 | 值 |
|------|-----|
| SSH 连接指令 | `ssh -p 32246 root@region-9.autodl.pro` |
| GPU | A100-PCIE-40GB × 1 |
| 项目根目录 | `/root/fundus-lesion-segmentation` |
| 数据集根目录 | `/root/fundus-lesion-segmentation/dataset/Segmentation` |
| outputs 目录 | `/root/fundus-lesion-segmentation/outputs` |

### GitHub 仓库

```
https://github.com/sonderhyr-cyber/fundus-lesion-segmentation
```

### 标准工作流（每次改代码后）

```bash
# 本地：提交并推送
git add <改动的文件>
git commit -m "描述"
git push

# AutoDL：拉取并执行
# 先 SSH 登录：ssh -p 32246 root@region-9.autodl.pro
cd /root/fundus-lesion-segmentation
git pull
python <脚本路径>
```

### Terminal 指令规范

**所有给用户的终端指令必须满足以下要求：**

1. 本地指令从项目根目录 `/Users/sonder/Desktop/fundus-lesion-segmentation` 出发
2. AutoDL 指令从 `/root/fundus-lesion-segmentation` 出发
3. 指令必须可以直接复制粘贴执行，不需要用户手动替换任何变量
4. 如果需要先 SSH 登录，第一行必须写明：`ssh -p 32246 root@region-9.autodl.pro`
5. 后台运行时 log 统一输出到 `outputs/<模块名>/logs/stdout.log`

---

## Dataset Information

Dataset root:

dataset/Segmentation

Structure:

dataset/  
└── Segmentation/  
├── images/  
│ ├── train/  
│ ├── valid/  
│ └── test/  
└── labels/  
├── train/  
├── valid/  
└── test/

YOLO segmentation format:

class_id x1 y1 x2 y2 ...

Coordinates are normalized.

Original classes:

MA -> 0  
HE -> 1  
EX -> 2  
SE -> 3

Semantic segmentation mapping:

background -> 0  
MA -> 1  
HE -> 2  
EX -> 3  
SE -> 4

Number of classes:

NUM_CLASSES = 5

---

## Completed Work

### 1. Data parsing

Implemented:

parse_yolo_segmentation()

Successfully converts YOLO polygon annotations into polygon coordinates.

Verified.

---

### 2. Mask generation

Implemented:

yolo_polygons_to_semantic_mask()

Uses OpenCV fillPoly().

Verified.

Mask labels:

0 background  
1 MA  
2 HE  
3 EX  
4 SE

---

### 3. Visualization

Completed:

- image visualization
- mask visualization
- overlay visualization

Manual inspection passed.

Labels align correctly with lesions.

---

### 4. Dataset

Implemented:

FundusSegmentationDataset

Returns:

image tensor:  
[3, 512, 512]

mask tensor:  
[512, 512]

Verified.

---

### 5. DataLoader

Verified.

Current configuration:

batch_size = 4

Output:

images:  
[B,3,512,512]

masks:  
[B,512,512]

Works correctly.

---

### 6. Image resizing

Important:

Original images have inconsistent sizes.

Resolved by resizing all samples to:

512 x 512

Image interpolation:

cv2.INTER_LINEAR

Mask interpolation:

cv2.INTER_NEAREST

Do NOT change mask interpolation.

---

### 7. U-Net

Implemented.

Current architecture:

Input:  
[3,512,512]

Encoder:  
64 -> 128 -> 256 -> 512 -> 1024

Decoder:  
1024 -> 512 -> 256 -> 128 -> 64

Output:  
[5,512,512]

Approximate parameters:

31M

Potential future optimization:

base_channels = 32

to reduce model size.

---

## Current Status

Training pipeline exists.

[train.py](http://train.py) implemented.

Configuration:

Optimizer:  
Adam

Learning rate:  
1e-3

Loss:  
CrossEntropyLoss

Epochs:  
5

Batch size:  
4

Model checkpoint:

outputs/best_model.pth

Not fully trained yet.

---

## Known Issues

### Class imbalance

Dataset is highly imbalanced.

Approximate lesion counts:

MA: 9869  
HE: 11443  
EX: 21566  
SE: 1207

Future work:

- Dice Loss
- Weighted CrossEntropy
- Focal Loss
- Dice + CE hybrid loss

Need experimentation.

---

## Important Project Rules

1. Always use relative paths.

Never hardcode:

/Users/...

D:/...

C:/...

1. Use pathlib.Path.
2. Keep AutoDL compatibility.
3. Save outputs into:

outputs/

1. Save checkpoints into:

outputs/checkpoints/

---

## Immediate Next Tasks

Priority 1

Verify training on AutoDL.

Tasks:

- upload dataset
- install requirements
- run [train.py](http://train.py)
- confirm loss decreases

Priority 2

Implement metrics:

- Dice
- IoU

Priority 3

Implement [predict.py](http://predict.py)

Generate:

- original image
- ground truth mask
- predicted mask
- overlay

Priority 4

Integrate Weights & Biases (wandb)

Track:

- train loss
- validation loss
- Dice
- IoU

---

## What Has Been Verified

Dataset: PASS

DataLoader: PASS

Mask generation: PASS

Visualization: PASS

U-Net forward pass: PASS

Training loop:  
NOT YET VERIFIED ON AUTODL

---

## Preferred Development Style

- Keep code simple.
- Make minimal changes.
- Verify each step before adding complexity.
- Prioritize reproducibility.
- Avoid introducing new architectures before baseline U-Net works.

---

## Unified Data Pipeline (ENFORCE STRICTLY)

本项目已经存在统一数据管线。

### 禁止

- 新建 dataset loader
- 新建 train.py
- 新建 evaluate.py
- 修改 dataset 目录结构
- 修改标签格式

### 必须

- 使用 src/dataset.py
- 使用现有 train pipeline（src/model/train.py）
- 使用现有 evaluate pipeline（src/model/evaluate.py）

### 允许

- 在 src/model/ 下新增模型文件
- 在 src/model/loss.py 中新增损失函数
- 在 requirements.txt 中增加缺失依赖

### 目标

所有模型共享同一套数据集和评价指标，保证实验结果可对比。

