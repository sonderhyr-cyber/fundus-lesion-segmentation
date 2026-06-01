# CLAUDE[.md](http://CONTEXT.md)

## Project Overview

Project Name:  
Fundus Lesion Segmentation using U-Net

Goal:  
Train a semantic segmentation model for retinal lesion segmentation on fundus images.

Current baseline model:  
U-Net

Framework:  
PyTorch

Development workflow:

- Local development: MacBook Air M4 + VS Code
- Remote training: AutoDL server
- SSH connected to VS Code

---

## Dataset Information

Dataset root:

dataset/Segmentation

Structure:

dataset/  
└── Segmentation/  
├── images/  
│ ├── train/  
│ ├── valid/  
│ └── test/  
└── labels/  
├── train/  
├── valid/  
└── test/

YOLO segmentation format:

class_id x1 y1 x2 y2 ...

Coordinates are normalized.

Original classes:

MA -> 0  
HE -> 1  
EX -> 2  
SE -> 3

Semantic segmentation mapping:

background -> 0  
MA -> 1  
HE -> 2  
EX -> 3  
SE -> 4

Number of classes:

NUM_CLASSES = 5

---

## Completed Work

### 1. Data parsing

Implemented:

parse_yolo_segmentation()

Successfully converts YOLO polygon annotations into polygon coordinates.

Verified.

---

### 2. Mask generation

Implemented:

yolo_polygons_to_semantic_mask()

Uses OpenCV fillPoly().

Verified.

Mask labels:

0 background  
1 MA  
2 HE  
3 EX  
4 SE

---

### 3. Visualization

Completed:

- image visualization
- mask visualization
- overlay visualization

Manual inspection passed.

Labels align correctly with lesions.

---

### 4. Dataset

Implemented:

FundusSegmentationDataset

Returns:

image tensor:  
[3, 512, 512]

mask tensor:  
[512, 512]

Verified.

---

### 5. DataLoader

Verified.

Current configuration:

batch_size = 4

Output:

images:  
[B,3,512,512]

masks:  
[B,512,512]

Works correctly.

---

### 6. Image resizing

Important:

Original images have inconsistent sizes.

Resolved by resizing all samples to:

512 x 512

Image interpolation:

cv2.INTER_LINEAR

Mask interpolation:

cv2.INTER_NEAREST

Do NOT change mask interpolation.

---

### 7. U-Net

Implemented.

Current architecture:

Input:  
[3,512,512]

Encoder:  
64 -> 128 -> 256 -> 512 -> 1024

Decoder:  
1024 -> 512 -> 256 -> 128 -> 64

Output:  
[5,512,512]

Approximate parameters:

31M

Potential future optimization:

base_channels = 32

to reduce model size.

---

## Current Status

Training pipeline exists.

[train.py](http://train.py) implemented.

Configuration:

Optimizer:  
Adam

Learning rate:  
1e-3

Loss:  
CrossEntropyLoss

Epochs:  
5

Batch size:  
4

Model checkpoint:

outputs/best_model.pth

Not fully trained yet.

---

## Known Issues

### Class imbalance

Dataset is highly imbalanced.

Approximate lesion counts:

MA: 9869  
HE: 11443  
EX: 21566  
SE: 1207

Future work:

- Dice Loss
- Weighted CrossEntropy
- Focal Loss
- Dice + CE hybrid loss

Need experimentation.

---

## Important Project Rules

1. Always use relative paths.

Never hardcode:

/Users/...

D:/...

C:/...

1. Use pathlib.Path.
2. Keep AutoDL compatibility.
3. Save outputs into:

outputs/

1. Save checkpoints into:

outputs/checkpoints/

---

## Immediate Next Tasks

Priority 1

Verify training on AutoDL.

Tasks:

- upload dataset
- install requirements
- run [train.py](http://train.py)
- confirm loss decreases

Priority 2

Implement metrics:

- Dice
- IoU

Priority 3

Implement [predict.py](http://predict.py)

Generate:

- original image
- ground truth mask
- predicted mask
- overlay

Priority 4

Integrate Weights & Biases (wandb)

Track:

- train loss
- validation loss
- Dice
- IoU

---

## What Has Been Verified

Dataset: PASS

DataLoader: PASS

Mask generation: PASS

Visualization: PASS

U-Net forward pass: PASS

Training loop:  
NOT YET VERIFIED ON AUTODL

---

## Preferred Development Style

- Keep code simple.
- Make minimal changes.
- Verify each step before adding complexity.
- Prioritize reproducibility.
- Avoid introducing new architectures before baseline U-Net works.
- 

