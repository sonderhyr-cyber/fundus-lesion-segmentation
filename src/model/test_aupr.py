"""
Correctness test for the histogram-based AUPR in evaluate.py.

Why this exists: our headline metric (mAUPR) is computed from a streaming
histogram rather than by handing every pixel to sklearn, because a
native-resolution test split is ~1e9 pixels. This test proves the histogram
gives the same answer as sklearn's average_precision_score, including in the
regimes that actually broke earlier implementations.

Run:  python src/model/test_aupr.py
Gate: max abs difference < 1e-3 (i.e. well inside the +-0.1% two-person
      reproducibility requirement for cross-checking a collaborator's numbers).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.evaluate import LOG_FLOOR, PRHistogram

TOLERANCE = 1e-3


def _case(name: str, probs: np.ndarray, labels: np.ndarray) -> tuple[str, float, float, float]:
    hist = PRHistogram(torch.device("cpu"))
    hist.update(torch.from_numpy(probs).float(), torch.from_numpy(labels))
    got = hist.average_precision()
    ref = float(average_precision_score(labels.astype(np.int8), probs))
    return name, ref, got, abs(ref - got)


def build_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    cases = []

    n = 300_000
    y = rng.random(n) < 0.5
    cases.append(("balanced, separable", np.clip(rng.normal(np.where(y, 0.7, 0.3), 0.15), 1e-12, 1), y))

    # Mirrors MA: 0.037% prevalence, probabilities around 1e-4.
    y = rng.random(n) < 0.00037
    cases.append(("MA-like imbalance", np.clip(rng.lognormal(np.where(y, -6.0, -9.0), 1.2), 1e-12, 1), y))

    # An under-trained channel: every probability far below 1e-9.
    y = rng.random(n) < 0.001
    cases.append(("all probs ~1e-10", 10.0 ** rng.normal(np.where(y, -10, -12), 0.5), y))

    y = rng.random(n) < 0.001
    cases.append(("all probs ~1e-13", 10.0 ** rng.normal(np.where(y, -12, -15), 0.5), y))

    # Exact zeros from a saturated softmax must not blow up log binning.
    y = rng.random(n) < 0.002
    p = np.where(rng.random(n) < 0.5, 0.0, 10.0 ** rng.normal(np.where(y, -3, -6), 0.5))
    cases.append(("contains exact zeros", p, y))

    y = rng.random(n) < 0.01
    cases.append(("no signal", rng.random(n), y))

    y = rng.random(n) < 0.005
    cases.append(("near-perfect", np.where(y, rng.uniform(0.9, 1.0, n), rng.uniform(0.0, 0.1, n)), y))

    return cases


def main() -> int:
    print(f"LOG_FLOOR = {LOG_FLOOR:g}   tolerance = {TOLERANCE:g}\n")
    print(f"{'case':<24}{'sklearn':>12}{'histogram':>12}{'abs diff':>12}")
    print("-" * 60)

    worst = 0.0
    for name, probs, labels in build_cases():
        name, ref, got, diff = _case(name, probs, labels)
        worst = max(worst, diff)
        print(f"{name:<24}{ref:>12.6f}{got:>12.6f}{diff:>12.2e}")

    print("-" * 60)
    print(f"worst absolute difference: {worst:.3e}")

    # A class with no positive pixels must be NaN (excluded from the mean),
    # never 0.0 — reporting 0 would silently drag mAUPR down.
    empty = PRHistogram(torch.device("cpu"))
    empty.update(torch.rand(1000), torch.zeros(1000, dtype=torch.bool))
    assert np.isnan(empty.average_precision()), "absent class must yield NaN, not 0"
    print("absent-class handling: NaN (correct)")

    ok = worst < TOLERANCE
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
