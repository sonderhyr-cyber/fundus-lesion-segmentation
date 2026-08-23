# S1 恢复说明（2026-08-17 21:56 暂停）

## 暂停时状态
- COMBO（E4 细节分支 + E1v2 CE+Dice）训练至 **iter 6200 / 40000**（val mIoU@4k = 34.8%，健康）
- 最近 checkpoint：`work_dirs/exp_combo_split613/iter_6000.pth`（latest.pth 指向它，526MB 完整）
- best_mIoU_iter_4000.pth 也在
- 训练进程与监控已全部停止，GPU 空闲

## 恢复训练（开机后执行）
```
cd /root/HRDecoder
setsid nohup /root/miniconda3/bin/python tools/train.py configs/lesion/exp/final_combo_split613.py \
    --work-dir work_dirs/exp_combo_split613 \
    --resume-from work_dirs/exp_combo_split613/latest.pth \
    > work_dirs/exp_combo_split613/nohup_train_resume.log 2>&1 < /dev/null &
```
- resume 会从 iter 6000 继续（权重+优化器+iter 计数），训练到 40000 完成
- 监控链（monitor.sh v6）可随后重启；其逻辑：COMBO 完成后自动跑 E1v2 finalize → COMBO-seed123 → E2E3

## 注意
- 若 resume 时 latest.pth 丢失，可从 iter_6000.pth 恢复（同样命令换文件名）
- 磁盘有 20G 空闲；评估脚本会生成 6.4GB npz 缓存，用后即删（finalize_experiment.sh 已自动删）
- 不要同时跑两个 train.py（单 GPU）
