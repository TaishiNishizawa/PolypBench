"""Per-frame overlap metrics between a binary prediction and a binary ground truth.

Frames whose ground truth is empty (no polyp in view, e.g. after removal) are
scored separately: Dice/IoU are undefined there (0/0), so they get
false-alarm metrics instead. Callers report the two groups separately.
"""

import numpy as np

POLYP_METRICS = ["dice", "iou", "precision", "recall"]
EMPTY_METRICS = ["false_alarm", "fp_area_frac"]


def binarize(prob, threshold):
    return prob >= threshold


def frame_metrics(pred, gt):
    """pred, gt: HxW bool/0-1 arrays. Returns {metric: value} for this frame."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any():
        return {
            "false_alarm": float(pred.any()),       # 1 if anything was predicted at all
            "fp_area_frac": float(pred.mean()),     # fraction of the frame wrongly predicted
        }
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    return {
        "dice": 2 * tp / (2 * tp + fp + fn),
        "iou": tp / (tp + fp + fn),
        "precision": tp / (tp + fp) if tp + fp > 0 else float("nan"),   # undefined if nothing predicted
        "recall": tp / (tp + fn),
    }
