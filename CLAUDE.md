# CLAUDE.md

> 最后更新：2026-07-21（暑期冲刺启动）

## Project Overview

**项目名称：** Fundus Lesion Segmentation（眼底病灶分割）

**目标：** 一个月内做出**可发表成果**。以 HRDecoder（MICCAI 2024）为骨干，针对微小病灶（MA/SE）提出改进模块与尺寸感知损失，在 DDR + IDRiD 上做完整对比与消融。

**框架：** PyTorch

**主指标：** **mAUPR**（逐病灶 PR 曲线下面积）—— 本领域标准，**不是 IoU**

**开发流程：**
- 本地开发：MacBook Air M4 + VS Code
- 远程训练：AutoDL（A100-PCIE-40GB × 1）
- VS Code 通过 SSH 连接

**角色分工：**

| 角色 | 负责内容 |
|---|---|
| **B（用户，同时是负责人）** | 统一评测脚本、M2MRF 复现、重算旧基线、产出【表 1】；并独立验收 A 的交付 |
| **A（另一位同学）** | 官方数据划分、HRDecoder 复现、实验框架 |

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

> 注：曾创建过 `fundus-seg-2026` 备用仓库，**现已弃用**，一切以本仓库为准。

### 标准工作流（每次改代码后）

```bash
# 本地：提交并推送
cd /Users/sonder/Desktop/fundus-lesion-segmentation
git add <改动的文件>
git commit -m "描述"
git push

# AutoDL：拉取并执行
ssh -p 32246 root@region-9.autodl.pro
cd /root/fundus-lesion-segmentation
git pull
python <脚本路径>
```

### Terminal 指令规范

**所有给用户的终端指令必须满足：**

1. 本地指令从 `/Users/sonder/Desktop/fundus-lesion-segmentation` 出发
2. AutoDL 指令从 `/root/fundus-lesion-segmentation` 出发
3. 可直接复制粘贴执行，不需手动替换变量
4. 需要 SSH 时第一行必须写明：`ssh -p 32246 root@region-9.autodl.pro`
5. 后台运行 log 统一输出到 `outputs/<模块名>/logs/stdout.log`

---

## Dataset Information

**数据集：** DDR（Diabetic Retinopathy Dataset），分割子集共 757 张。

```
dataset/
├── Segmentation/          ← 本月唯一使用
│   ├── images/{train,valid,test}/
│   └── labels/{train,valid,test}/
├── Grading/               （分类任务，本月冻结）
├── Detection/             （本月冻结）
└── best_segmentation/     （IDRiD 类数据，第 3 周泛化实验可能用）
```

**标签格式：** YOLO segmentation，`class_id x1 y1 x2 y2 ...`，坐标已归一化。

| 原始类别 | MA | HE | EX | SE |
|---|---|---|---|---|
| class_id | 0 | 1 | 2 | 3 |

**语义分割映射：**

| 标签 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 含义 | background | MA | HE | EX | SE |

`NUM_CLASSES = 5`

### ⚠️ 数据划分现状（第 1 周必须解决）

| | 本地现状 | 官方标准（文献口径） |
|---|---|---|
| train | **338** | **383** |
| valid | 149 | 149 |
| test | **208** | **225** |

**缺失的 45 张训练图 + 17 张测试图本地不存在**（已解压的 `dataset/` 与曾经的 `dataset.zip` 均为 338/149/208），**必须重新从 DDR 官方源下载补齐**。

> 用非官方划分 = 结果无法与任何论文对比 = 不可发表。这是第 1 周的头号前置任务。

### 类别不均衡（核心难点）

各病灶多边形标注数量（训练集）：MA 9869 / HE 11443 / EX 21566 / **SE 1207**

MA 最难：仅占训练像素约 0.037%，背景:MA ≈ 2712:1。

---

## Completed Work

1. **数据解析** `parse_yolo_segmentation()` — YOLO 多边形 → 坐标，已验证
2. **掩膜生成** `yolo_polygons_to_semantic_mask()` — OpenCV `fillPoly()`，已验证
3. **可视化** — 原图 / mask / overlay，人工检查通过，标注与病灶对齐
4. **Dataset** `FundusSegmentationDataset` — 返回 `[3,512,512]` 图像与 `[512,512]` 掩膜
5. **DataLoader** — batch_size=4，输出 `[B,3,512,512]` / `[B,512,512]`
6. **图像缩放** — 统一 512×512；图像 `cv2.INTER_LINEAR`，**掩膜 `cv2.INTER_NEAREST`（严禁修改）**
7. **U-Net** — 编码 64→1024，解码 1024→64，输出 `[5,512,512]`，约 31M 参数
8. **损失函数** — `src/model/loss.py` 已实现 FocalTverskyLoss / DiceLoss / DiceCELoss
9. **前景偏置 Patch 采样** — 每 epoch 1500 个 patch，80% 前景 / 20% 随机背景
10. **SMP 基线** — `src/baselines/smp/`，ResNet34 + ImageNet 预训练
11. **M2MRF 复现** — `src/models/m2mrf/`，配置 `configs/m2mrf.yaml`
12. **HRDecoder 复现** — `src/baselines/hrdecoder/`，配置 `configs/hrdecoder.yaml`

---

## Current Status（真实结果）

| 模型 | 验证集 mIoU | 测试集 mIoU | 备注 |
|---|---|---|---|
| 自建 U-Net（从零） | 0.057 | 0.096 | SE 类 IoU=0，仅作最低基线 |
| SMP U-Net (ResNet34) | 0.107 | 0.121 | 自建线最好 |
| MMSeg 复现线 | — | mIoU 28.51 / **mAUPR 44.23** | **尚未复现到已发表水平** |

**已有权重：** `outputs/checkpoints/best_model.pth`（119M，自建 U-Net）

### SOTA 坐标（DDR 测试集，官方划分）

> **2026-07-21 修正：** 原表格中 M2MRF / HRDecoder 的数字有误（疑似与 MLNet 自身结果及
> HRDecoder 论文 IDRiD 列混淆），已用三个独立信源交叉核实改正，见下方来源说明。

| 方法 | mAUPR | mDice/mF | mIoU | 来源 |
|---|---|---|---|---|
| M2MRF-A (PR 2022) | 49.56 | — | 31.47 | [官方 GitHub](https://github.com/CVIU-CSU/M2MRF-Lesion-Segmentation) 自报（DDR） |
| M2MRF (HRDecoder 复现) | 49.12 | 46.11 | 30.72 | HRDecoder 论文 Table 1，相同 protocol 独立复现 |
| HRDecoder (MICCAI 2024) | 49.27 | 48.21 | 32.25 | [论文](https://arxiv.org/abs/2411.03976) Table 1/2，DDR test，三次重复均值 |
| MLNet (Algorithms 2024) | 51.81 | 49.85 | 37.19 | [论文](https://doi.org/10.3390/a17040164) 摘要原话，DDR |

三个来源互相印证：M2MRF 官方仓库（49.56/31.47）、HRDecoder 论文独立复现的 M2MRF
（49.12/30.72）、以及 MLNet 论文正文按"比次优方法高 1.87% mAUPR / 6.92% mIoU"反推出的
M2MRF（≈49.9/≈30.3）——三者高度吻合。**HRDecoder 在 DDR 上的真实已发表成绩是
mAUPR 49.27 / mIoU 32.25，不是之前记录的 52–53。** MLNet 是唯一站上 51.81% 的方法，且它
发表早于 HRDecoder，双方未直接对比过。

**差距 = 我们 44.23% vs HRDecoder 已发表 49.27%，第 1 周要补齐的是 ~5 个点，不是 ~8 个。**
**T1.3 的 ±1% 红线目标相应改为 49.27% mAUPR（对齐 HRDecoder，而非 MLNet 的 51.81%）。**

---

## Important Project Rules（铁律）

1. **只做 DDR 分割这一条线**，本月冻结 YOLO 分类/分级（那是另一篇论文）。
2. **数据必须用官方划分 383/149/225。**
3. **主指标 mAUPR**（附 mDice、mIoU），口径对齐 M2MRF/HRDecoder 论文。
4. **AUPR 必须用 softmax 概率图计算，严禁用 argmax。**
5. **先复现、再创新。** 复现未达标前不引入任何新模型/新模块。
6. **掩膜插值 `cv2.INTER_NEAREST` 严禁修改。**
7. 一律使用相对路径 + `pathlib.Path`，**禁止硬编码** `/Users/...`、`D:/`、`C:/`。
8. 保持 AutoDL 兼容；输出进 `outputs/`，权重进 `outputs/checkpoints/`。
9. 完整训练放 AutoDL（A100），本地只做冒烟测试。
10. **每次实验必须登记实验台账**（`docs/experiments.csv`）。

---

## Unified Data Pipeline（ENFORCE STRICTLY）

本项目已有统一数据管线，所有模型共享同一套数据与评价指标，保证实验可对比。

**禁止：** 新建 dataset loader / 新建 `train.py` / 新建 `evaluate.py` / 修改 dataset 目录结构 / 修改标签格式

**必须：** 使用 `src/dataset.py`、`src/model/train.py`、`src/model/evaluate.py`

**允许：** 在 `src/model/`、`src/models/` 下新增模型文件；在 `src/model/loss.py` 新增损失；在 `requirements.txt` 增补依赖

---

## Immediate Next Tasks（第 0–1 周）

详见 `docs/新Session工作指南-第0与第1周.md`。

**第 0 周（开工准备）**
- 提交当前未跟踪的代码（configs/、src/models/、src/baselines/hrdecoder/ 等）
- AutoDL 环境自检；建实验台账；建 `sprint-week1` 分支
- 与 A 对齐分工与红线

**第 1 周（对齐口径 + 复现基线）**
- **T1.0** 重新下载 DDR 官方数据，补齐 383/149/225 ← 头号前置
- **T1.1**【A】固化官方划分，`dataset.py` 读取划分清单
- **T1.2**【B】升级 `evaluate.py`：mAUPR（概率图）+ mDice + mIoU
- **T1.3**【A】HRDecoder 复现到已发表 ±1% ← 本周锚点
- **T1.4**【B】M2MRF 第二基线
- **T1.5**【B】用新脚本重算旧基线
- **T1.6**【B】产出【表 1】统一口径对比表

**红线：** HRDecoder 未复现到 ±1%，不得进入第 2 周。

---

## Preferred Development Style

- 代码保持简单，改动最小化
- 每一步验证通过再加复杂度
- 优先保证可复现性
- 基线未跑通前不引入新架构
- 不轻信口头数字，一切以负责人用统一脚本复算的结果为准
