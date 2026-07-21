# 新 Session 工作指南 · 第 0 周 + 第 1 周

> **给谁看：** 一个全新的 Claude session（冷启动、无上下文）。
> **你在为谁工作：** 用户是**项目负责人**，同时承担技术分工中的 **B 角色**。你既要帮他完成 B 的活，也要帮他**监督 A 的交付**。
> **使命：** 完成**第 0 周（开工准备）**与**第 1 周（对齐口径 + 复现基线）**。
> **一句话目标：** 补齐官方数据划分、把评测统一到 mAUPR、并把 HRDecoder 复现到已发表水平——为后续创新打好可信地基。
> **最后更新：** 2026-07-21

---

## 0. 开始前请先读这些（30 分钟上手）

按顺序读，读完就有足够上下文开工：

1. `CLAUDE.md` —— 项目铁律、现状、目录结构、AutoDL/GitHub 工作流（**最重要，已于 07-21 更新**）
2. `src/dataset.py` —— 统一数据管线（禁止另建 loader）
3. `src/model/train.py`、`src/model/evaluate.py`、`src/model/loss.py` —— 现有训练/评测/损失
4. `configs/hrdecoder.yaml`、`configs/m2mrf.yaml` —— 复现基线配置
5. `src/baselines/hrdecoder/`、`src/models/m2mrf/` —— 两个 SOTA 复现代码
6. `暑期一个月冲刺-研究路线与目标.md`、`暑期冲刺-每周TodoList.md` —— 总路线与周计划

---

## 1. 角色与分工（**重要**）

| 角色 | 谁 | 负责内容 |
|---|---|---|
| **B（执行）** | **用户（我）** | 统一评测脚本、M2MRF 复现、重算旧基线、产出【表 1】 |
| **A（执行）** | 另一位同学 | 官方数据划分、HRDecoder 复现、实验框架 |
| **负责人（监督）** | **用户（我）** | 盯全局进度、独立验收 A 的交付、守红线、管台账 |

**你（新 session）的定位：** 用户的技术搭档。
- 标 **【B·我做】** 的任务 → 你直接动手实现。
- 标 **【A 做·我监督】** 的任务 → 你不代替 A 干活，但要帮用户**准备验收清单、独立复算 A 的数字、发现问题及时预警**。

> 关键杠杆：**评测脚本掌握在 B 手里**。用户可以用自己的脚本独立复评 A 的 checkpoint，不必轻信 A 报的数字。这是本项目最重要的质量闸门。

---

## 2. 项目背景速览

**任务：** 糖尿病视网膜病变（DR）眼底图像**多病灶语义分割**。
**数据集：** DDR（分割子集 757 张），4 类病灶 + 背景，共 5 类。

| 语义标签 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 含义 | background | MA(微动脉瘤) | HE(出血) | EX(硬性渗出) | SE(软性渗出) |

- 类别极度不均衡，MA 最难（约占训练像素 0.037%，背景:MA ≈ 2712:1）。
- 图像原始分辨率 1500×1100 ~ 3200×2400，尺寸不一，统一缩放到 512×512。
- **本领域主指标是 mAUPR**（逐病灶 PR 曲线下面积），不是 IoU。

**当前进度：**

| 模型 | 验证 mIoU | 测试 mIoU | 备注 |
|---|---|---|---|
| 自建 U-Net | 0.057 | 0.096 | SE=0，最低基线 |
| SMP U-Net (ResNet34) | 0.107 | 0.121 | 自建线最好 |
| MMSeg 复现线 | — | 28.51 / **mAUPR 44.23** | 未达已发表水平 |

**SOTA 坐标（DDR 测试集，官方划分，2026-07-21 修正版——原 51–53% 系误记，见 CLAUDE.md 说明）：**

| 方法 | mAUPR | mDice/mF | mIoU | 来源 |
|---|---|---|---|---|
| M2MRF-A (PR 2022) | 49.56 | — | 31.47 | 官方 GitHub 自报 |
| HRDecoder (MICCAI 2024) | 49.27 | 48.21 | 32.25 | 论文 Table 1/2，DDR test |
| MLNet (Algorithms 2024) | 51.81 | 49.85 | 37.19 | 论文摘要，DDR |

**差距 = 44.23% vs HRDecoder 已发表 49.27%，第 1 周要补齐的是 ~5 个点。**
**T1.3 的 ±1% 红线以 49.27%（HRDecoder）为准，不是 MLNet 的 51.81%。**

---

## 3. 环境与基础设施

| 项目 | 值 |
|---|---|
| **GitHub 仓库** | **https://github.com/sonderhyr-cyber/fundus-lesion-segmentation** |
| 本地项目根目录 | `/Users/sonder/Desktop/fundus-lesion-segmentation` |
| AutoDL SSH | `ssh -p 32246 root@region-9.autodl.pro` |
| AutoDL 项目根 | `/root/fundus-lesion-segmentation` |
| GPU | A100-PCIE-40GB × 1 |
| 数据集根 | `dataset/Segmentation/{images,labels}/{train,valid,test}` |
| 已有权重 | `outputs/checkpoints/best_model.pth`（119M，自建 U-Net） |
| 输出目录 | `outputs/`；日志统一 `outputs/<模块名>/logs/stdout.log` |

> 曾创建的 `fundus-seg-2026` 备用仓库**已弃用**，一切以本仓库为准。
> 仓库已于 07-21 清理垃圾文件（释放 8.3GB）：删除了冗余的 `dataset.zip`、重复报告、`src.zip`、`.DS_Store`、`__pycache__`。

**标准工作流：**
```bash
cd /Users/sonder/Desktop/fundus-lesion-segmentation
git add <改动文件> && git commit -m "描述" && git push

ssh -p 32246 root@region-9.autodl.pro
cd /root/fundus-lesion-segmentation && git pull && python <脚本>
```

---

## 4. 铁律（违反 = 结果不可用）

1. **只做 DDR 分割这一条线。** 不碰 YOLO 分类/分级。
2. **数据必须用官方划分 383/149/225。**
3. **主指标 mAUPR**（附 mDice、mIoU），口径对齐 M2MRF/HRDecoder。
4. **AUPR 必须用 softmax 概率图算，严禁 argmax。**
5. **先复现、再创新。** 第 1 周不引入任何新模型/新模块。
6. **不新建 `train.py` / `dataset.py` / `evaluate.py`。** 只在现有文件内扩展。
7. **掩膜插值 `cv2.INTER_NEAREST` 严禁修改。**
8. 相对路径 + `pathlib`，不硬编码 `/Users/...`。
9. 完整训练放 AutoDL，本地只冒烟测试。
10. 每次实验必登记台账。

---

## 5. 第 0 周任务：开工准备

### T0.0【最先做】提交未跟踪的代码【B·我做】
当前仓库只跟踪了 24 个文件，**大量核心代码从未提交**，有丢失风险。

```bash
cd /Users/sonder/Desktop/fundus-lesion-segmentation
git add configs/ src/ train.py audit_dataset.py 暑期*.md docs/ CLAUDE.md
git commit -m "Add M2MRF/HRDecoder models, configs, entry point, sprint docs; update CLAUDE.md"
git push
```
- [ ] 确认 `configs/`、`src/models/`、`src/baselines/hrdecoder/`、`src/model/registry.py`、`train.py` 已入库
- [ ] 建议把 `midterm/`（47M）和经费 xlsx 加进 `.gitignore` 而非提交
- **验收：** `git status` 干净，`git ls-files | wc -l` 明显大于 24

### T0.1 环境自检（AutoDL）【B·我做】
```bash
ssh -p 32246 root@region-9.autodl.pro
cd /root/fundus-lesion-segmentation
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import mmcv, mmseg; print(mmcv.__version__, mmseg.__version__)"
git pull
```
- [ ] GPU 可见、`torch.cuda` 为 True、mmcv/mmseg 可导入
- [ ] 缺依赖只在 `requirements.txt` 增补，不改已有版本
- **验收：** 能在 AutoDL 上启动一次训练

### T0.2 建实验台账【B·我做，负责人维护】
- [ ] 建 `docs/experiments.csv`，列：
  `实验ID | 日期 | 负责人 | 模型 | 配置 | 划分 | 关键超参 | mAUPR | mDice | mIoU | EX/HE/SE/MA各类AUPR | checkpoint | 备注`
- [ ] **作为负责人**：要求 A 每次训练也必须登记
- **验收：** 表格存在、列齐全、A/B 都知晓登记义务

### T0.3 分支与目录规范【B·我做】
- [ ] `git checkout -b sprint-week1`
- [ ] 日志落 `outputs/<模块名>/logs/stdout.log`，权重落 `outputs/checkpoints/`
- **验收：** 分支创建成功

### T0.4 砍线与对齐【负责人·我】
- [ ] 向 A 明确本月冻结 YOLO 分类线，只推进 MMSeg 分割线
- [ ] 与 A 确认第 1 周分工与交付时间点
- **验收：** A 已确认分工与红线

**第 0 周红线：** 代码未提交、或 AutoDL 起不了训练 → 不进入第 1 周。

---

## 6. 第 1 周任务：对齐口径 + 复现基线

### T1.0【🔴 头号前置】重新下载 DDR 官方数据补齐划分【B·我做 + A 协助】
**这是本周最大的未知量，必须第一天启动。**

本地现状 **338/149/208**，官方标准 **383/149/225**。经核查，缺失的 45 张训练图 + 17 张测试图**本地完全不存在**（已解压的 `dataset/` 和曾经的 `dataset.zip` 都是 338/149/208）。

- [ ] 从 DDR 官方源重新下载完整分割数据（官方仓库 / 论文提供的下载链接）
- [ ] 核对补齐后计数为 `383/149/225`
- [ ] 与 M2MRF / HRDecoder 仓库提供的划分清单逐文件名比对
- [ ] 补齐后同步到 AutoDL（数据不进 git，单独上传）
- **验收：** 本地与 AutoDL 的 Segmentation 划分均为 383/149/225，且文件名与官方清单一致
- **风险预案：** 若官方源下载受阻超过 2 天，立即上报负责人，考虑改用 IDRiD 作为主数据集（需同步调整论文定位）

### T1.1 固化官方划分【A 做 · 我监督】
A 需交付：
- 划分清单文件 `splits/{train,val,test}.txt`
- `dataset.py` 读取该清单，划分固化不再变
- 三划分的图像数与逐类像素占比打印

**我（负责人）的验收动作：**
- [ ] 独立跑一遍计数，确认 `383/149/225`，不接受口头汇报
- [ ] 抽查文件名是否与官方清单一致
- [ ] 确认 mask 插值仍是 `cv2.INTER_NEAREST`
- [ ] 确认排查结论：原来的 338 是怎么来的、是否已还原

### T1.2【最高优先级】升级统一评测脚本（主指标 mAUPR）【B·我做】
现有 `evaluate.py` 只用 argmax 混淆矩阵算 IoU。**AUPR 必须用 softmax 概率图算。**

在现有 `src/model/evaluate.py` 内扩展（不新建文件）：
- [ ] 前向取 `softmax(logits)`，对每个前景类 c∈{1,2,3,4}：
  - 收集全体像素该类**预测概率** `p_c` 与**二值 GT** `(mask==c)`
  - `AUPR_c = average_precision_score(gt_c.flatten(), p_c.flatten())`
- [ ] 逐类输出 AUPR / Dice / IoU，再求前景均值 mAUPR / mDice / mIoU
- [ ] 背景类（0）不计入均值
- [ ] 结果打印 + 落盘 `outputs/<模型>/eval.txt`
- [ ] 内存注意：逐图流式累计，不要一次性堆全部像素概率
- [ ] 提供"评测任意 checkpoint"的命令行入口，方便复评 A 的模型

```python
from sklearn.metrics import average_precision_score
# 每图: probs [C,H,W] = softmax(logits); mask [H,W]
# 对每个前景类累积 (score, label)，最后统一算 AP
```

- **验收：** 能对任意 checkpoint 输出逐类+均值指标；与 A 各跑一次同一 checkpoint，数字一致（±0.1%）

### T1.3【本周锚点】复现 HRDecoder 到已发表 ±1%【A 做 · 我监督】
A 需交付：`python train.py --model hrdecoder` 训练完成 + 日志 + checkpoint + 官方 test(225) 上的 mAUPR

**我（负责人）的验收动作：**
- [ ] **用我自己的 T1.2 脚本独立复评 A 的 checkpoint**，不采信单方数字
- [ ] 确认落在已发表 ±1%（**49.27% mAUPR，即 HRDecoder 论文 DDR test 官方数字，非此前误记的 51–53%**）
- [ ] 若差距大，一起排查：划分 / 输入分辨率 / iter 数 / 损失 / 学习率调度
- [ ] 确认日志权重已规范落盘、已登记台账

### T1.4 复现 M2MRF 作为第二基线【B·我做】
- [ ] `python train.py --model m2mrf`，用统一脚本评测，登记逐类 AUPR
- [ ] 与论文对齐（DDR mAUPR ~51%）
- **验收：** M2MRF 有一组可信结果

### T1.5 用新脚本重算旧基线【B·我做】
- [ ] 用 T1.2 脚本重评 `outputs/checkpoints/best_model.pth`（自建 U-Net）
- [ ] SMP 基线权重若已丢失，需重训一次
- [ ] 旧 checkpoint 是在旧划分上训的，需在官方划分上重训或明确注明
- **验收：** 四基线在统一官方划分 + 统一口径下都有数

### T1.6 产出【表 1】统一口径对比表【B·我做】
- [ ] 行 = UNet / SMP / M2MRF / HRDecoder，列 = mAUPR / mDice / mIoU + 逐类 AUPR
- [ ] 存 `docs/table1_baselines.md`，登记进台账
- **验收：** 表 1 完成，可直接进论文实验章节初稿

---

## 7. 负责人监督清单（用户专属）

### 7.1 每日快检（5 分钟）
- [ ] A 今天有无提交？commit 是否遵守铁律（没新建 pipeline 文件、没硬编码路径）？
- [ ] 昨天的实验有没有登记台账？
- [ ] 有没有人改了数据划分或 mask 插值？
- [ ] T1.0 数据下载进度如何？（本周最大风险项）

### 7.2 关键闸门（必须亲自把关）
| 闸门 | 验收方式 | 不通过怎么办 |
|---|---|---|
| 官方数据补齐 383/149/225 | 我独立跑计数核对 | 卡住，启动 IDRiD 预案 |
| 评测口径一致 | 同 checkpoint 双人复算 ±0.1% | 先修脚本 |
| **HRDecoder 复现 ±1%** | **我用自己的脚本独立复评** | **不进第 2 周** |

### 7.3 周末对账清单
- [ ] 表 1 是否齐全（四基线 × 全指标）？
- [ ] HRDecoder 是否达标？没达标根因是什么、下周怎么补？
- [ ] 台账是否完整可复盘？
- [ ] 代码是否已 commit + push，AutoDL 是否同步？
- [ ] 第 2 周（创新点实现）前置条件是否就绪？

### 7.4 风险预警（发现即干预）
- 🚨 **DDR 官方数据下载受阻 > 2 天** → 启动 IDRiD 预案，调整论文定位
- 🚨 A 报的数字我复算不出来 → 立刻核对划分与评测口径
- 🚨 HRDecoder 复现卡住 > 3 天 → 降级：先用 M2MRF 当主基线
- 🚨 有人"顺手"做新模块/新架构 → 违反"先复现再创新"，立即叫停
- 🚨 台账断更 > 2 天 → 立即补齐

---

## 8. 第 1 周交付物清单（Definition of Done）

**B（我）交付：**
- [ ] 官方数据补齐到 383/149/225（T1.0）
- [ ] `evaluate.py` 升级：逐类+均值 **mAUPR/mDice/mIoU**，双人复算一致
- [ ] M2MRF 第二基线结果
- [ ] 四基线统一口径重算完成
- [ ] 【表 1】统一口径对比表

**A 交付（我验收）：**
- [ ] 官方划分固化 + 可复现核对
- [ ] **HRDecoder 复现 mAUPR 在已发表 ±1% 内**（本周锚点）

**共同：**
- [ ] 所有实验登记台账，日志/权重规范落盘，代码已 commit + push

---

## 9. 常见坑与提醒

- **T1.0 数据补齐是本周最大未知量**，第一天就要启动，不要拖。
- **AUPR 用概率不用 argmax**：升级评测最容易做错的点。
- **划分不一致会让所有对比作废**。
- **HRDecoder 没复现到位就别进第 2 周**。
- **mask 插值 `INTER_NEAREST` 不可改**。
- **别在本地跑完整训练**：本地只冒烟测试，真训练在 A100。
- **不轻信口头数字**：一切以我用自己脚本复算的结果为准。

---

## 10. 参考文献

- M2MRF (PR 2022) — https://arxiv.org/pdf/2111.00193
- HRDecoder (MICCAI 2024) — https://arxiv.org/abs/2411.03976 · 代码 https://github.com/CVIU-CSU/HRDecoder
- MLNet (2024) — https://doi.org/10.3390/a17040164
- DDR 数据集基准 — https://arxiv.org/pdf/2008.09772

---

**本周结束时向用户汇报：四基线统一口径对比表 + HRDecoder 复现是否达标（±1%）。达标才进入第 2 周。**
