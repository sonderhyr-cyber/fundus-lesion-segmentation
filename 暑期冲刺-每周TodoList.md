# 暑期一个月冲刺 · 每周 TodoList

> 配套文档：`暑期一个月冲刺-研究路线与目标.md`
> 分工：**A = 骨干/训练/复现**；**B = 方法模块/损失/消融**；🤝 = 两人共担
> 主指标：**mAUPR**（附 mDice / mIoU）；数据用**官方划分 383/149/225**

---

## ✅ 第 0 步：开工前（Day 0，半天，🤝）
- [ ] 确认 AutoDL A100 环境可用，HRDecoder/M2MRF 依赖（MMCV 1.7.2 / MMSeg 0.16.0）装好
- [ ] 建实验记录表（Notion/飞书/Excel）：一行 = 一次实验，记 配置 / mAUPR / mDice / mIoU / 逐类 AUPR / 备注
- [ ] git 建分支 `sprint-hrdecoder`，约定"每改代码即 commit + push"
- [ ] **砍线**：暂停 YOLO 分类线与自建 U-Net 调参，本月只跑 MMSeg 分割线

---

## 📅 第 1 周：地基 — 对齐口径 + 复现基线
> 里程碑：一张"统一口径下 U-Net / SMP / M2MRF / HRDecoder"对比表，且 **HRDecoder 复现到已发表 ±1%**

### A（骨干/复现）
- [ ] 数据集划分改回**官方 383/149/225**（排查之前为何变成 338/208，大概率过滤了无病灶图）
- [ ] 核对官方划分文件与文献一致，写进 `dataset.py` / config，锁定不再改
- [ ] 在 A100 上跑通 **HRDecoder 官方配置**，复现 DDR mAUPR 到已发表 ±1%（~51%）
- [ ] 保存复现 checkpoint + 完整日志到 `outputs/hrdecoder/logs/`

### B（评测/第二基线）
- [ ] 升级 `src/model/evaluate.py`：统一评测脚本，主输出 **mAUPR（逐类 PR-AUC）** + mDice + mIoU，口径对齐 M2MRF/HRDecoder 论文
- [ ] 复现 **M2MRF** 作为第二基线，记录逐类 AUPR（EX/HE/SE/MA）
- [ ] 把 自建 U-Net / SMP 结果用**新评测脚本重算**一遍（口径统一才能比）
- [ ] 产出【表 1】统一口径 baseline 对比表

### 🤝 周末对账
- [ ] HRDecoder 复现数字 vs 论文差距 < 1%？否则本周不进入第 2 周
- [ ] 评测脚本双人交叉验证（同一 checkpoint 两人各跑一次，数字一致）

---

## 📅 第 2 周：创新点实现（首选方案：HRDecoder + 微小病灶模块 + 尺寸感知损失）
> 里程碑：新方法能完整训练并出 mAUPR，初步 ≥ baseline

### B（方法/损失）
- [ ] 在 `src/model/loss.py` 实现**尺寸感知加权损失**（小连通域病灶更大权重，MA/SE 优先）
- [ ] 实现**微小病灶增强模块**（频域/边界分支，放大红色小点 MA/HE），接到 HRDecoder 骨干
- [ ] 单元测试：前向 shape 正确、loss 可反传、显存占用可接受
- [ ] 遵守统一管线：**不新建 train.py / dataset.py**，只在 `src/models/`、`loss.py` 扩展

### A（实验框架）
- [ ] 搭实验网格脚本（lr / 损失权重 / patch 前景采样比），支持一键复跑
- [ ] 日志与 checkpoint 规范落盘 `outputs/<模块名>/logs/stdout.log`
- [ ] 跑第一版新方法完整训练，出首个 mAUPR 数字

### 🤝 周末对账
- [ ] 新方法 mAUPR ≥ HRDecoder baseline？记录逐类 AUPR，重点看 MA/SE 是否动起来

---

## 📅 第 3 周：刷点 + 消融 + 泛化
> 里程碑：主表 + 消融表 + IDRiD 表 + 定性图 全部齐活

### A（刷点/泛化）
- [ ] 超参搜索，把 DDR mAUPR 推向冲刺线（~53%+）
- [ ] 跑 **IDRiD 泛化实验**（同方法、同评测），出 IDRiD mAUPR
- [ ] 记录计算量 / 参数量 / 推理速度对比（HRDecoder 卖点之一）

### B（消融/统计）
- [ ] 消融实验：①去掉微小病灶模块 ②去掉尺寸感知损失 ③换回 baseline 损失
- [ ] 每个配置跑多次，报**均值 ± 方差**（可复现性）
- [ ] 产出【表 2】消融表、【表 3】IDRiD 泛化表

### 🤝 定性可视化
- [ ] 生成 原图 / GT / 预测 / overlay 四联图，挑 MA/SE 改善明显的样本
- [ ] 挑 3–5 组代表性对比图（我们 vs baseline）

---

## 📅 第 4 周：成稿
> 里程碑：可投稿初稿 + 可复现代码

### 🤝 写作
- [ ] 定目标会议/期刊（BIBM / ISBI / EMBC 或 BSPC / CBM），下载模板
- [ ] 分工写：Intro + Related（B）/ Method + 结构图（B）/ Experiments + Ablation（A）/ Conclusion（🤝）
- [ ] 画方法结构总图（骨干 + 模块 + 损失）

### A（复现/开源）
- [ ] 复核论文所有数字可一键复现
- [ ] 整理开源 README（环境 / 数据 / 训练 / 评测命令）

### B（引用/润色）
- [ ] 补全对比方法引用（M2MRF / HRDecoder / MLNet 等）
- [ ] 格式检查 + 语言润色 + 图表清晰度

### 🤝 交付
- [ ] 初稿 + 代码给导师过一遍，收集修改意见

---

## 🚦 每周红线（不达标不进入下一周）
1. **第 1 周**：HRDecoder 复现 < 论文 ±1%，且评测脚本双人一致 → 否则卡住先修
2. **第 2 周**：新方法 mAUPR ≥ baseline → 否则回退方法设计
3. **第 3 周**：至少一个指标（总 mAUPR 或 MA 类 AUPR）显著优于 baseline → 这是论文的立身之本
4. **全程**：官方划分 + mAUPR 口径不动摇；先复现后创新
