# SMP Baseline — Retinal Lesion Segmentation

A clean, standalone training pipeline using
[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
to diagnose whether the dataset or the custom U-Net implementation is the limiting factor.

---

## Diagnostic goal

| Result | Conclusion |
|--------|------------|
| SMP also predicts almost entirely background | Dataset / label pipeline is likely wrong |
| SMP learns foreground classes significantly better | Custom U-Net implementation is the problem |

---

## Folder structure

```
project/
├── dataset/
│   └── Segmentation/
│       ├── images/{train,valid,test}/
│       └── labels/{train,valid,test}/
│
├── src/
│   ├── dataset.py          ← shared — reused unchanged
│   ├── paths.py            ← shared — reused unchanged
│   └── baselines/
│       └── smp/
│           ├── model.py    ← SMP UNet builder + ImageNet normalisation
│           ├── losses.py   ← SoftDiceLoss + CombinedDiceCELoss
│           ├── train.py    ← training loop (AdamW, 20 epochs, best-by-Dice)
│           ├── evaluate.py ← Dice, mIoU, per-class metrics
│           └── predict.py  ← 4-panel visualisation for val samples
│
└── outputs/
    └── smp/
        ├── checkpoints/    ← best_model.pth
        ├── logs/           ← training_log.csv
        └── predictions/    ← pred_01_*.png … pred_08_*.png
```

---

## Installation

```bash
pip install segmentation-models-pytorch
```

All other dependencies (`torch`, `numpy`, `opencv-python`, `matplotlib`) are
already required by the project.

---

## Usage

All scripts are run from the **project root** so that relative paths resolve correctly.

### Train

```bash
python src/baselines/smp/train.py
```

Configuration (edit constants at the top of `train.py`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `BATCH_SIZE` | 4 | |
| `EPOCHS` | 20 | |
| `LEARNING_RATE` | 1e-4 | AdamW |
| `WEIGHT_DECAY` | 1e-4 | AdamW |
| `EPOCH_SIZE` | 1500 | patches per epoch |
| `FG_RATIO` | 0.8 | fraction of foreground-centred patches |
| `EARLY_STOP_PATIENCE` | 10 | epochs without val Dice improvement |

The training log is written to `outputs/smp/logs/training_log.csv`:

```
epoch, train_loss, val_loss, val_dice, val_miou, lr
1, 0.892341, 0.871234, 0.023100, 0.012300, 1.00e-04
…
```

The best checkpoint (highest val Dice) is saved to
`outputs/smp/checkpoints/best_model.pth`.

---

### Evaluate

```bash
python src/baselines/smp/evaluate.py
```

Reports Dice and IoU for every class on the **valid** and **test** splits:

```
=== valid ===
Mean Dice (foreground): 0.4812
Mean IoU  (foreground): 0.3214

  Class         Dice     IoU
  ------------  ------  ------
  background  0.9823  0.9650
  MA          0.3102  0.1838
  HE          0.5412  0.3720
  EX          0.6801  0.5154
  SE          0.3933  0.2441
```

Results are also saved to `outputs/smp/evaluation_results.txt`.

---

### Predict (visualise)

```bash
python src/baselines/smp/predict.py
```

Generates 8 random validation samples, each saved as a 4-panel PNG
(`Image | Ground Truth | Prediction | Overlay`) under
`outputs/smp/predictions/`.

---

## Model details

| Setting | Value |
|---------|-------|
| Architecture | SMP UNet |
| Encoder | ResNet34 |
| Encoder weights | ImageNet |
| Input channels | 3 |
| Output classes | 5 (background, MA, HE, EX, SE) |
| Activation | None (raw logits) |
| Image normalisation | ImageNet mean/std applied before every forward pass |

---

## Loss function

**CombinedDiceCELoss** = 0.5 × SoftDice + 0.5 × CrossEntropy

- SoftDice is averaged over foreground classes (1–4) only; background is excluded.
- CrossEntropy covers all classes including background.

---

## Differences from the custom U-Net pipeline

| Aspect | Custom U-Net | SMP baseline |
|--------|-------------|--------------|
| Encoder | Scratch (31 M params) | ResNet34 + ImageNet weights |
| Optimizer | Adam | AdamW |
| Learning rate | 3e-4 | 1e-4 |
| Loss | FocalTverskyLoss | DiceLoss + CrossEntropyLoss |
| Epochs | 100 + early stop | 20 + early stop |
| Train data | ForegroundPatchDataset | ForegroundPatchDataset (same) |
| Val data | FundusSegmentationDataset | FundusSegmentationDataset (same) |
