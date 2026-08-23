# HRDecoder 提升实验仓库（hrdecoder-experiments）

> ⚠️ **划分声明**：本目录所有实验使用**自定义 6:1:3 重划分**（757 张 → train 454 / val 76 /
> test 227，seed 42，SE 分层），**不是官方划分（383/149/225）**——数字不可与官方划分下的
> 任何结果直接对比（官方划分同配置复跑 test mIoU ≈ 30.8%，6:1:3 为 42.2%，差异来源是
> 训练数据 +18.5% 与测试集病灶构成不同）。

## checkpoint 下载

**Google Drive（主）**：https://drive.google.com/file/d/1dPWfCVqdX-pa9-tR6TZXwdV4rL_cG6lq/view?usp=sharing
（7 个 best checkpoint 打包，文件名带实验名前缀：e4_detail / e1v2_cedice / combo_e4_cedice /
e2e3_fuse_aspp / e4v2_detail_aux / combo_v2 / combo2）

**GitHub Release（备选）**：https://github.com/a1553846342-dotcom/fundus-seg-2026/releases/tag/hrdecoder-ddgr-exp-2026-08

## 目录结构

```
hrdecoder-experiments/
├── README.md                本文件
├── REPORT.md                实验报告（全实验对比表、结论、机制分析）
├── week1/                   第一周：板块一（划分+baseline）/ 板块二（SE 增强）/ 板块三（loss 加权）
│   ├── baseline_report.md   板块一：6:1:3 划分统计 + baseline 复现（mIoU 42.20 / mAUPR 60.59）
│   ├── module2_3_report.md  板块二/三：SE copy-paste、滑窗过采样、loss 权重的全部负结果
│   ├── scripts/             划分脚本 ddr_split.py、baseline 配置、se_aug.py、滑窗 HRDecoder、
│   │                        各 loss 加权配置、报告生成脚本
│   └── logs/                9 份第一周训练日志 json（baseline/②③④/1.5x/2x/3x/自适应）
├── logs/                    本批实验（E1~E4v2 系列）8 份 mmcv 训练日志 json
├── evals/                   各实验 test/val 评估文本与 PR/阈值分析 json
└── code/                    全部实验配置（configs/）+ 核心代码改动（mmseg/）+ 总补丁
                             （exp_improve_all_changes.patch），详见 code/README.md
```

## 实验速览（test 227，标准协议，基于 6:1:3 划分）

| 实验 | 改动 | mIoU / mAUPR | 结论 |
|---|---|---|---|
| baseline | — | 42.20 / 60.59 | 对齐基准 |
| E4 | 全分辨率细节分支 | 41.63 / 59.39 | MA 召回+5.8pp、精度-7.7pp |
| E1v2 | CE+Dice loss | 39.43 / 56.36 | val 假象，test 崩 |
| COMBO | E4+E1v2 | 41.08 / 59.55 | CE+Dice 毒成分 |
| E2E3 | 可学习融合+ASPP | 39.98 / 58.84 | EX +0.6，HE -4.7 |
| E4v2 | E4+辅助损失 | 40.06 / 58.81 | 干扰主损失 |
| **E6** | **逐类阈值调优** | **val +0.7pp；E2E3 test +0.88pp** | **唯一正收益** |

## 复现

```bash
git apply code/exp_improve_all_changes.patch   # 在 HRDecoder 仓库（mmseg 0.16/mmcv 1.7.1/torch 2.0.1）
python tools/train.py code/configs/<实验名>_split613.py --work-dir <work_dir>   # 单卡 4090, bs=2, lr=0.005
python tools/test.py  code/configs/<实验名>_split613.py <ckpt> --eval mIoU
```
