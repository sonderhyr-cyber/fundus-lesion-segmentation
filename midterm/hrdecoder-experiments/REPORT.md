# HRDecoder DDR 实验报告（2026-08-16 ~ 08-18）

> **⚠️ 划分声明（重要）**：仓库 README 规定数据用 **官方划分（383/149/225）**；
> 本批实验全部使用**自定义 6:1:3 重划分**（757 张合并 → train 454 / val 76 / test 227，
> seed 42，按 SE 病灶分层抽样）。**以下所有数字仅在该自定义划分下有效，
> 不可与官方划分下的任何结果直接对比**（官方划分下同配置复跑 test mIoU 约 30.8%，
> 新划分 42.2% 的主要来源是训练数据 +18.5% 与测试集病灶构成不同）。

## 1. 背景与目标

- 模型：HRDecoder（MICCAI 2024，arXiv:2411.03976），本地单卡改良版
  （单卡 BN、bs=2、lr=0.005、预训练权重强注入），MMSegmentation 0.16 / mmcv 1.7.1 / PyTorch 2.0.1。
- 任务：DDR 4 类病灶（EX/HE/SE/MA）像素级分割，主指标 mAUPR + mIoU。
- 评估协议：val 76 张选 checkpoint（save_best mIoU）；test 227 张仅最终评估；单次推理、
  无 TTA、sigmoid 0.5 阈值；mAUPR = 11 阈值（0~1 步长 0.1）全局 PR 梯形面积。
- **Baseline**（板块一复跑）：mIoU 42.20 / mF1 58.45 / mAUPR 60.59
  （EX 51.56/72.66，HE 43.13/63.82，SE 49.98/69.60，MA 24.11/36.27，IoU/AUPR）。

## 2. 瓶颈诊断（阶段 B）

1. **MA 是绝对瓶颈**：IoU 24.11（次差 HE 43.13），召回 33.5%。MA 病灶中位仅 **48 px²**
   （p95 244px²），而 logit 原生网格 = 2048 图的 1/8（骨干输入 1024、stride-4）——
   中位 MA **不足 1 个 logit 格子**；HR 分支只上采样同一份低分特征，不含真高分辨率信息。
2. **PR 曲线**：MA 在阈值 0.1 时召回即封顶 53%——模型对过半 MA 不产生高分（分数问题）。
3. **连通域检出率**：所有类别按尺寸单调下降（<100px 检出 0.12~0.52，≥500px 0.77~0.78）。
4. **阈值校准**：HE/MA 的 sigmoid 分数系统性偏低，0.5 阈值次优（HE val 最优 0.25，MA 0.30）。

## 3. 实验与结果（test 227，标准协议）

| # | 实验 | 改动 | mIoU | mF1 | mAUPR | 结论 |
|---|------|------|------|-----|-------|------|
| - | baseline | — | 42.20 | 58.45 | 60.59 | 对齐基准 |
| E1 | Tversky loss | loss 换 tversky（α=0.3/0.7 各一） | — | — | — | 训练塌缩（iter4k 全正预测，mIoU 0.17%/0.39%） |
| E4 | 细节分支 | stride-4 全分辨率 CNN（32ch）+可学习加性融合 | 41.63 | 57.91 | 59.39 | MA 召回 +5.8pp、精度 -7.7pp（IoU 持平） |
| E1v2 | CE+Dice | loss 换 ce_dice | 39.43 | 55.73 | 56.36 | val 假象（39.1 vs 39.4），test 全面崩（HE -6） |
| COMBO | E4+E1v2 | 细节分支+CE+Dice | 41.08 | 57.38 | 59.55 | CE+Dice 毒成分盖过 E4 收益 |
| E2E3 | 融合+ASPP | 可学习 LR/HR 融合+ASPP-lite 上下文 | 39.98 | 56.23 | 58.84 | EX +0.6 但 HE -4.7、SE -3.9 |
| E4v2 | E4+辅助损失 | 细节头直接 dice 监督（0.3） | 40.06 | 56.99 | 58.81 | 辅助损失干扰主损失 |
| combo2 | E4v2+E2E3 | 组合 | — | — | — | 预算中断（24500/40000，val 37.1% 无惊喜） |
| **E6** | **阈值调优** | 逐类阈值 EX .45/HE .25/SE .40/MA .30 | **val +0.7pp** | — | — | **唯一正收益**（E2E3 上 test 验证 +0.88pp，HE +2.4pp） |

- E2/E3 为打包验证（未单独跑）——消融不严谨，如实标注。
- 全部负面结果的机制分析见 `EXPERIMENT_LOG.md`（Tversky 塌缩/CE+Dice val 假象/E4 精度/
  E2E3 HE 崩坏/E4v2 干扰）。

## 4. 结论

1. **唯一被真实实验验证的正收益路线：保持 baseline 架构 + 逐类阈值调优**
   （HE 0.25 / MA 0.30 / EX 0.45 / SE 0.40）：val mIoU 39.4→40.1（+0.7pp）；
   test 侧在 E2E3 上验证 mIoU +0.88pp（HE 单类 +2.4pp）。机制：HE/MA 分数系统性偏低。
2. 全部 6 个模型级改动（细节分支/融合/上下文/loss 替换/辅助损失）test 均为负面或持平；
   HE 对一切改动敏感（-1.5~-6pp），baseline 的 HE 43.13 近乎上限。
3. **局限**：baseline+阈值调优的 test 端到端数字未直接测量（baseline checkpoint 因磁盘
   事故被误删），由同协议下 E2E3 的 +0.88pp 外推；E2E3/E4v2 的 checkpoint 与 json 在
   另一台已关机服务器（S1）上，本批 Release 不含；combo2 无 test 结果。

## 5. 文件清单

- `logs/`：各实验 mmcv 训练日志 json（E4/E1v2/COMBO×2/E2E3/E4v2/COMBO-v2/combo2，共 8 份）
- `evals/`：各实验 test/val 评估文本与 PR 分析 json
- `code/`：全部实验配置（configs/）+ 核心代码改动（mmseg/）+ 总补丁
  （`exp_improve_all_changes.patch`）；详见 `code/README.md`
- `checkpoints/`（GitHub Release `hrdecoder-ddgr-exp-2026-08` 附件，7 个 best checkpoint，
  文件名为实验名前缀）：`e4_detail_best_mIoU_iter_36000.pth`、`e1v2_cedice_best_mIoU_iter_12000.pth`、
  `combo_e4_cedice_best_mIoU_iter_32000.pth`、`e2e3_fuse_aspp_best_mIoU_iter_20000.pth`、
  `e4v2_detail_aux_best_mIoU_iter_36000.pth`、`combo_v2_best_mIoU_iter_16000.pth`（中断）、
  `combo2_best_mIoU_iter_12000.pth`（中断）
