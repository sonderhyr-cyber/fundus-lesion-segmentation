# HRDecoder SE 专项实验报告（板块二 + 板块三）

实验框架：HRDecoder（MICCAI 2024, arXiv:2411.03976），MMSegmentation 0.16 / mmcv 1.7.1 /
PyTorch 2.0.1，DDR 数据集固定 6:1:3 划分（train 454 / val 76 / test 227，seed 42，板块一产出）。
板块二（SE copy-paste / 滑窗过采样）与板块三（Loss 类别加权）均为在论文框架基础上
**自行设计的扩展实验，非论文原文方法**。SE = Soft Exudate（软性渗出），四类病灶标签之一。

所有实验：单次训练（seed=42）、40000 iter、val 选 checkpoint、test 集统一评估（227 张，无 TTA）。
测试集结果（%）：**IoU / F1 / AUPR**。

## 板块二：SE 专项数据增强

| 实验 | mIoU | mF1 | mAUPR | EX IoU | HE IoU | SE IoU | MA IoU | SE F1 | SE AUPR |
|---|---|---|---|---|---|---|---|---|---|
| ① baseline（复用板块一） | **42.20** | **58.45** | **60.59** | 51.56 | 43.13 | **49.98** | 24.11 | 66.65 | 69.60 |
| ② +SE copy-paste | 39.67 | 56.08 | 58.77 | 50.54 | 39.62 | 44.22 | 24.30 | 61.32 | 67.67 |
| ③ +SE 滑窗过采样 | 40.07 | 56.38 | 58.39 | 51.77 | 38.22 | 46.30 | 24.01 | 63.29 | 66.42 |
| ④ ②+③ 组合 | 40.79 | 57.16 | 60.13 | 51.46 | 42.29 | 45.28 | 24.15 | 62.34 | 68.43 |

val 轨迹（mIoU best）：baseline 39.4%@32k；② 39.7%@32k；③ 39.6%@20k；④ 40.4%@40k。

**结论（如实报告）**：
- 测试集上没有任何 SE 增强组超过 baseline 的 SE IoU（49.98），整体 mIoU 也略降（-1.4 ~ -2.5pp）。
- ②④（copy-paste 系）显著提升 SE 精确率 PPV（83.9% / 81.5% vs baseline 72.8%），但召回率下降
  （48.3% / 50.5% vs 61.5%），净效果 IoU 降低；copy-paste 让模型预测 SE 更"保守"。
- ③ 滑窗过采样在 val 上提升明显（36.9%@16k、39.6%@20k），但测试集未转化（SE 46.3 vs 50.0），
  存在 val/test 泛化差距。
- 板块二三种方案在本实验设置（单次种子、HRDecoder 本身已含 HR 分支强监督）下均为负结果/持平，
  作为可复现的对照数据保留，不做美化。

## 板块三：Loss 类别加权（L_Seg 中 SE 项加权）

权重定义：class_weight = [EX, HE, SE, MA]，作用于 BinaryLoss(Dice) 逐类求和；
自适应组 = 训练集像素频率倒数、均值归一化为 1（[0.236, 0.154, 1.027, 2.584]）。

| SE权重 | mIoU | mF1 | mAUPR | EX IoU | HE IoU | SE IoU | MA IoU |
|---|---|---|---|---|---|---|---|
| 1x（baseline） | **42.20** | **58.45** | **60.59** | 51.56 | 43.13 | **49.98** | 24.11 |
| 1.5x | 40.90 | 57.33 | 59.13 | 49.47 | **43.96** | 45.68 | 24.48 |
| 2x | 40.31 | 56.64 | 58.04 | 51.00 | 42.02 | 44.77 | 23.45 |
| 3x | 38.50 | 54.89 | 55.63 | 49.79 | 37.81 | 42.61 | 23.82 |
| 自适应（逆频率） | 38.76 | 54.88 | 56.90 | 50.62 | 32.58 | 48.52 | 23.31 |

曲线见 `se_weight_vs_miou.png`。

**结论（如实报告）**：
- SE 权重越大，整体 mIoU 单调下降（42.20 → 40.90 → 40.31 → 38.50），SE 自身 IoU 也下降，
  与"加权重提升 SE"的直觉相反；推测原因：HRDecoder 的 L_Seg 已包含 fuse 分支的类别间耦合，
  提高 SE 权重同时降低了 EX/HE 的学习，而 SE 本身标注稀疏，权重提升主要放大噪声/漏标影响。
- 1.5x 是相对最优（mIoU 40.90，HE 43.96 甚至超过 baseline），但 SE 仍低于 baseline。
- 自适应逆频率权重主要牺牲 HE（32.58 vs 43.13，-10.5pp），SE 为 48.52（接近 baseline），
  符合"压低高频类"的设计预期，但整体不划算。

## 交付物与文件位置

- 服务器（已关机，数据保留在实例磁盘）：`/root/HRDecoder/data/DDR_split_613/`（划分+报告）、
  `/root/HRDecoder/work_dirs/`（各实验日志/checkpoint）、`/root/HRDecoder/ddr_split.py`、
  `build_se_library.py`、`mmseg/datasets/pipelines/se_aug.py`、`mmseg/models/segmentors/HRDecoder.py`、
  `configs/lesion/*split613*`
- 本地备份：`D:\dataset\lesion_segmentation\server_checkpoints\`（baseline/②③④/1.5x/2x/3x best
  checkpoint 共 7 个 + 曲线图）；自适应组 checkpoint 在 S3 实例磁盘（已关机，如需可重启取回）

## 说明

- 本报告为单种子（42）单次训练结果，未做 3 次重复取平均；如需交叉验证可对关键对比组
  （②④、1.5x）补跑种子 123/2024（重启任一实例即可）。
- test 集仅在最终统一评估时使用，全程未参与调参/选点（选点仅用 val）。
