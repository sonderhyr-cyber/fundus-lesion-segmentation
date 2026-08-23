# code/ —— 实验代码说明

本目录包含 HRDecoder 全部提升实验的**代码与配置**（与 6:1:3 自定义划分配套）。

## 文件清单

### 配置（configs/，文件名即实验名，均继承 `baseline_split613.py`）
| 文件 | 对应实验 | 说明 |
|------|----------|------|
| baseline_split613.py | baseline | 板块一 baseline 复现配置（自定义 6:1:3 划分） |
| e1_tversky_split613.py | E1 | Tversky loss（alpha=0.7 为最终版；0.3 版已废弃——塌缩） |
| e1_cedice_split613.py | E1v2 | CE+Dice loss（负面） |
| e2_fuse_split613.py | E2 | 可学习 LR/HR logit 融合（未单独运行，见 E2E3） |
| e3_context_split613.py | E3 | ASPP-lite 上下文（未单独运行，见 E2E3） |
| e2e3_split613.py | E2E3 | E2+E3 打包（负面，无法归因） |
| e4_detail_split613.py | E4 | 全分辨率细节分支（MA 召回+5.8pp） |
| e4_detail_split613_bs1.py | E4 备用 | bs=1 回退配置（未用） |
| e4v2_detail_split613.py | E4v2 | E4+细节头直接监督（负面） |
| final_combo_split613.py | COMBO | E4+CE+Dice（负面） |
| combo_v2_split613.py | COMBO-v2 | COMBO+细节辅助损失（中途杀停） |
| combo2_split613.py | combo2 | E4v2+E2E3（预算中断） |
| e1e4_split613.py | （未运行） | E1+E4 组合草稿 |
| final_combo_split613_seed123/2024.py | （未运行） | 种子变体草稿 |

### 核心代码（mmseg/）
| 文件 | 实验 | 关键改动 |
|------|------|----------|
| HRDecoder.py | E2/E4/E4v2 | `_fuse_logits` 可学习融合；`DetailBranch/DetailHead` 全分辨率路径（原图 2048 输入，stride-4，32 通道）+ `detail_scale` 加性融合 + `detail_loss_weight` 辅助监督 |
| fcn_head.py | E3 | `context_cfg` 参数：局部卷积后插入 ASPP 上下文模块 |
| aspp_context.py | E3 | ASPP-lite：1×1 + 3×3 膨胀(6/12/18) + 全局池化（池化分支无 BN） |
| detail_branch.py | E4 | `DetailBranch`（3×3×3, stride 2/2/1）与 `DetailHead`（2×3×3 + 1×1） |

### 补丁
- `exp_improve_all_changes.patch`：exp/improve 分支相对用户基线（b50d21c）的全部代码改动
  （git format-patch 生成，可 `git apply`）。

## 使用方式

```bash
# 1. 准备 HRDecoder 仓库（MMSegmentation 0.16 / mmcv 1.7.1 / PyTorch 2.0.1, py3.8）
git apply exp_improve_all_changes.patch
# 2. 数据：DDR_split_613（自定义 6:1:3 划分, seed 42）
# 3. 训练（单卡 4090, bs=2, lr=0.005）
python tools/train.py configs/lesion/exp/<实验名>_split613.py --work-dir <work_dir>
# 4. 评估（test 227，标准协议）
python tools/test.py configs/lesion/exp/<实验名>_split613.py <ckpt> --eval mIoU
```
