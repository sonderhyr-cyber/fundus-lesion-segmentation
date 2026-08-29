"""
Unified evaluation script — the single source of truth for all reported numbers.

Protocol (aligned with M2MRF / HRDecoder / MLNet so our numbers are comparable
to the published DDR leaderboard):

  1. The network sees a 512x512 input, but its logits are bilinearly upsampled
     back to the image's ORIGINAL resolution before any metric is computed.
     Ground truth is never downsampled. This matters enormously for MA, whose
     lesions are 5-20 px wide and are destroyed by a 512x512 resize.
  2. AUPR is computed from SOFTMAX PROBABILITIES, never from argmax.
  3. AUPR is a single GLOBAL PR curve per class over all pixels of all images
     in the split (not a per-image AP that is then averaged) — this is the
     convention used by the papers we compare against.
  4. Background (class 0) is excluded from every mean.

Primary metric is mAUPR; mDice and mIoU are reported alongside it.

Usage
-----
    python src/model/evaluate.py --model unet
    python src/model/evaluate.py --model hrdecoder \
        --checkpoint outputs/checkpoints/hrdecoder_best.pth --split test

The --checkpoint flag makes this script able to re-score ANY checkpoint,
including ones produced by a collaborator, which is how we independently
verify reported numbers rather than taking them on trust.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import CHECKPOINT_PATH, OUTPUT_DIR
from dataset import CLASS_NAMES, NUM_CLASSES, FundusSegmentationDataset

FOREGROUND_CLASSES = tuple(range(1, NUM_CLASSES))  # 1..4 -> MA, HE, EX, SE

# ---------------------------------------------------------------------------
# Probability quantisation for memory-bounded AUPR
# ---------------------------------------------------------------------------
# A full-resolution test split is ~1e9 pixels. Holding every pixel's
# probability in memory to hand to sklearn would need tens of GB, so instead we
# accumulate a histogram of (probability bin -> positive/negative counts) and
# derive the exact PR curve from the cumulative counts at the end.
#
# Bins are LOG-spaced, not linear. Average precision is invariant to any
# monotonic transform of the scores, so log spacing is exact in principle and
# far better behaved in practice: rare classes like MA produce probabilities
# around 1e-4, which linear binning would collapse into a single bin and
# corrupt the high-precision end of the curve.
#
# LOG_FLOOR must sit far below any probability we expect to see. An
# under-trained MA channel can output ~1e-10 across the WHOLE image; if the
# floor were near that value every score would tie in bin 0 and the reported
# AUPR would collapse to the class prevalence (verified: a floor of 1e-9 turned
# a true AP of 0.77 into 0.02). 1e-30 is comfortably below float32 softmax
# outputs while still leaving ~16k bins per decade.
N_BINS = 500_000
LOG_FLOOR = 1e-30         # probabilities below this are treated as this
_LOG_FLOOR_EXP = -30.0    # log10(LOG_FLOOR)


def _prob_to_bin(probs: torch.Tensor) -> torch.Tensor:
    """Map probabilities in [0,1] to log-spaced integer bins in [0, N_BINS-1]."""
    clamped = probs.clamp(min=LOG_FLOOR, max=1.0)
    # log10(p) spans [_LOG_FLOOR_EXP, 0] -> normalise to [0, 1]
    normalised = (torch.log10(clamped) - _LOG_FLOOR_EXP) / (-_LOG_FLOOR_EXP)
    bins = (normalised * (N_BINS - 1)).long()
    return bins.clamp_(0, N_BINS - 1)


class PRHistogram:
    """
    Streaming accumulator for one class's global precision-recall curve.

    Stores two int64 histograms over quantised probability, so memory is O(N_BINS)
    regardless of how many pixels are streamed through it.
    """

    def __init__(self, device: torch.device) -> None:
        self.pos = torch.zeros(N_BINS, dtype=torch.int64, device=device)
        self.neg = torch.zeros(N_BINS, dtype=torch.int64, device=device)

    def update(self, probs: torch.Tensor, is_positive: torch.Tensor) -> None:
        """probs: [N] float scores. is_positive: [N] bool ground-truth labels."""
        bins = _prob_to_bin(probs.reshape(-1))
        is_positive = is_positive.reshape(-1)
        self.pos += torch.bincount(bins[is_positive], minlength=N_BINS)
        self.neg += torch.bincount(bins[~is_positive], minlength=N_BINS)

    def average_precision(self) -> float:
        """
        AP = sum_k (R_k - R_{k-1}) * P_k, sweeping the threshold from high score
        to low. This is the same step-wise definition sklearn's
        average_precision_score uses (no trapezoidal interpolation).
        """
        # Move to CPU before the float64 reduction: Apple MPS has no float64
        # support, and float32 is not enough precision for a cumulative sum over
        # ~1e9 pixels (it starts losing counts around 1.7e7).
        pos_hist = self.pos.cpu()
        neg_hist = self.neg.cpu()

        pos = pos_hist.flip(0).cumsum(0).double()   # TP at each threshold
        neg = neg_hist.flip(0).cumsum(0).double()   # FP at each threshold
        total_pos = float(pos_hist.sum().item())
        if total_pos == 0:
            return float("nan")  # class absent from this split — undefined, not 0

        # Guard against the floor-saturation failure mode described above: if
        # most positives landed in the bottom bin, scores are being quantised
        # into a tie and the AP is no longer trustworthy.
        saturated = float(pos_hist[0].item()) / total_pos
        if saturated > 0.01:
            print(
                f"  [WARN] {saturated:.1%} of positive pixels have probability "
                f"<= {LOG_FLOOR:g} and are tied in the bottom bin. AUPR for this "
                f"class is likely UNDERESTIMATED — lower LOG_FLOOR before trusting it."
            )

        predicted_pos = pos + neg
        precision = torch.where(
            predicted_pos > 0, pos / predicted_pos.clamp(min=1), torch.zeros_like(pos)
        )
        recall = pos / total_pos

        prev_recall = torch.cat([torch.zeros(1, dtype=torch.float64, device=recall.device), recall[:-1]])
        return float(((recall - prev_recall) * precision).sum().item())


# ---------------------------------------------------------------------------
# Model loading (architecture-agnostic)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_name: str,
    checkpoint_path: Path | None,
    device: torch.device,
    cfg: dict | None = None,
) -> nn.Module:
    """
    Build any supported architecture and load weights into it.

    Reuses train.build_model so that evaluation can never silently drift from
    the architecture that was actually trained.
    """
    from model.train import build_model
    from model.loss import build_loss

    # HRDecoder's constructor needs a criterion for its auxiliary HR loss; it is
    # unused at inference time but must be supplied to build the module.
    criterion = build_loss((cfg or {}).get("loss"), num_classes=NUM_CLASSES).to(device)
    model, default_ckpt = build_model(model_name, criterion, device, cfg=cfg)

    path = Path(checkpoint_path) if checkpoint_path is not None else default_ckpt
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    model.eval()
    print(f"  Loaded {model_name} from {path}")
    return model


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Stream the split once, accumulating everything needed for AUPR/Dice/IoU
    at the images' original resolution.
    """
    histograms = {c: PRHistogram(device) for c in FOREGROUND_CLASSES}
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)

    for i, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)          # [1, H, W] at ORIGINAL resolution

        outputs = model(images)
        if isinstance(outputs, tuple):    # HRDecoder returns (logits, aux_loss)
            outputs = outputs[0]

        # Upsample logits to the ground truth's original resolution, THEN softmax.
        native_hw = masks.shape[-2:]
        logits = F.interpolate(outputs, size=native_hw, mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[0]     # [C, H, W]

        target = masks[0]                            # [H, W]

        for c in FOREGROUND_CLASSES:
            histograms[c].update(probs[c], target == c)

        preds = probs.argmax(dim=0)
        valid = (target >= 0) & (target < NUM_CLASSES)
        idx = NUM_CLASSES * target[valid] + preds[valid]
        confusion += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(
            NUM_CLASSES, NUM_CLASSES
        )

        if i % 25 == 0:
            print(f"    {i}/{len(loader)} images")

    return _summarise(histograms, confusion.cpu().numpy())


def _summarise(histograms: dict[int, PRHistogram], confusion: np.ndarray) -> dict:
    aupr, dice, iou = {}, {}, {}

    for c in FOREGROUND_CLASSES:
        aupr[c] = histograms[c].average_precision()

        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - tp)
        fn = float(confusion[c, :].sum() - tp)
        dice[c] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou[c] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    def _mean(d: dict[int, float]) -> float:
        values = [v for v in d.values() if not np.isnan(v)]
        return float(np.mean(values)) if values else 0.0

    total = confusion.sum()
    return {
        "aupr_per_class": aupr,
        "dice_per_class": dice,
        "iou_per_class": iou,
        "mAUPR": _mean(aupr),
        "mDice": _mean(dice),
        "mIoU": _mean(iou),
        "pixel_accuracy": float(np.trace(confusion) / total) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(
    model_name: str,
    checkpoint: Path | str,
    split: str,
    m: dict,
    remove_black_border: bool = False,
) -> str:
    lines = [
        "=" * 62,
        f"Model      : {model_name}",
        f"Checkpoint : {checkpoint}",
        f"Split      : {split}",
        "Protocol   : native-resolution, softmax probabilities, global PR curve",
        f"Black crop : {remove_black_border}",
        "=" * 62,
        f"{'class':<12}{'AUPR':>12}{'Dice':>12}{'IoU':>12}",
        "-" * 62,
    ]
    for c in FOREGROUND_CLASSES:
        lines.append(
            f"{CLASS_NAMES[c]:<12}"
            f"{m['aupr_per_class'][c] * 100:>11.2f}%"
            f"{m['dice_per_class'][c] * 100:>11.2f}%"
            f"{m['iou_per_class'][c] * 100:>11.2f}%"
        )
    lines += [
        "-" * 62,
        f"{'MEAN':<12}"
        f"{m['mAUPR'] * 100:>11.2f}%"
        f"{m['mDice'] * 100:>11.2f}%"
        f"{m['mIoU'] * 100:>11.2f}%",
        "=" * 62,
        f"Pixel accuracy: {m['pixel_accuracy'] * 100:.2f}%",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified mAUPR/mDice/mIoU evaluation at native resolution"
    )
    parser.add_argument("--model", default=None, choices=["unet", "hrdecoder", "m2mrf"])
    parser.add_argument(
        "--config", default=None,
        help="Training experiment YAML; required for non-default architecture/preprocessing",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint (default: the model's configured checkpoint)",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "valid", "val", "test"],
        help="Split to evaluate (default: test)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N images — for local smoke tests only",
    )
    args = parser.parse_args()

    from model.train import PROJECT_ROOT as TRAIN_PROJECT_ROOT, _load_cfg

    preliminary_model = args.model or "unet"
    cfg = _load_cfg(preliminary_model, args.config)
    model_name = args.model or cfg.get("model", {}).get("name", "unet")
    if args.model is not None and cfg.get("model", {}).get("name", args.model) != args.model:
        raise ValueError(
            f"--model {args.model} conflicts with model.name={cfg['model']['name']}"
        )
    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg.get("root", "dataset/Segmentation"))
    if not data_root.is_absolute():
        data_root = TRAIN_PROJECT_ROOT / data_root
    preprocess_kwargs = {
        "remove_black_border": bool(data_cfg.get("remove_black_border", False)),
        "black_border_threshold": int(data_cfg.get("black_border_threshold", 10)),
        "black_border_padding_ratio": float(data_cfg.get("black_border_padding_ratio", 0.02)),
    }

    device = get_device()
    print(f"Device: {device}")

    model = load_model(model_name, args.checkpoint, device, cfg=cfg)

    dataset = FundusSegmentationDataset(
        root=data_root,
        split="valid" if args.split == "val" else args.split,
        native_mask=True,
        **preprocess_kwargs,
    )
    if args.limit is not None:
        dataset.samples = dataset.samples[: args.limit]
        print(f"  [smoke test] limited to {len(dataset.samples)} images")
    expected_counts = data_cfg.get("expected_counts", {})
    expected = expected_counts.get(args.split)
    if expected is None and args.split in {"valid", "val"}:
        expected = expected_counts.get("valid", expected_counts.get("val"))
    if expected is not None and args.limit is None and len(dataset) != int(expected):
        raise RuntimeError(
            f"{args.split} split has {len(dataset)} images, expected {expected}. "
            f"Refusing to evaluate a mismatched dataset protocol."
        )
    print(f"  {args.split}: {len(dataset)} images from {data_root}")

    # batch_size MUST be 1: native-resolution masks have differing shapes.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    metrics = evaluate(model, loader, device)

    report = format_report(
        model_name,
        args.checkpoint or "(config default)",
        args.split,
        metrics,
        remove_black_border=preprocess_kwargs["remove_black_border"],
    )
    print(report)

    experiment_name = Path(args.config).stem if args.config else model_name
    out_path = OUTPUT_DIR / experiment_name / "eval.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(report)
    print(f"Appended results to {out_path}")


if __name__ == "__main__":
    main()
