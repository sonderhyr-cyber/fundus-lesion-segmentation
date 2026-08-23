import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, se_weight, mIoU, se_iou, mF1, mAUPR)
data = [
    ("baseline", 1.0, 42.20, 49.98, 58.45, 60.59),
    ("SE 1.5x", 1.5, 40.90, 45.68, 57.33, 59.13),
    ("SE 2x", 2.0, 40.31, 44.77, 56.64, 58.04),
    ("SE 3x", 3.0, 38.50, 42.61, 54.89, 55.63),
]
x = [d[1] for d in data]
m = [d[2] for d in data]
s = [d[3] for d in data]
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, m, "o-", label="mIoU (EX/HE/SE/MA)", color="#1f77b4")
ax.plot(x, s, "s--", label="SE IoU", color="#d62728")
for xi, mi, si in zip(x, m, s):
    ax.annotate(f"{mi:.2f}", (xi, mi), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)
    ax.annotate(f"{si:.2f}", (xi, si), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=9)
ax.axhline(38.76, color="gray", ls=":", lw=1)
ax.text(2.6, 38.95, "adaptive mIoU 38.76", fontsize=8, color="gray")
ax.set_xlabel("SE class weight in L_Seg (Dice)")
ax.set_ylabel("Test IoU (%)")
ax.set_title("HRDecoder DDR 6:1:3 - SE weight vs test IoU")
ax.set_xticks(x)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("/root/HRDecoder/work_dirs/se_weight_vs_miou.png", dpi=150)
print("plot saved")
