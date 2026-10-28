"""Probability maps -> boxes, for box-level (detection) evaluation on REAL-Colon.

One box per connected component of the map thresholded at the config's `threshold`,
found at the model's output resolution (the stored npz maps) and scaled to the
frame's original pixels. Every box keeps two candidate scores, the peak and the
mean probability over its component, and its area as a fraction of the map, so
the score rule and the minimum area are evaluation choices (metrics/detection.py)
and never need a re-extraction. Configured in configs/boxes/.

Stored per sequence as one table, one row per box (frames with no box have no row):

    <artifacts_root>/boxes/<dataset>/<model_id>/<postproc>/<boxes>/<sequence>.csv.gz
        frame, xmin, ymin, xmax, ymax   original pixels, 0-based, inclusive (as the ground truth)
        area                            component pixels / map pixels
        peak, mean                      probability over the component, in [0, 1]

The file's presence marks the sequence done (written to a temporary name, then renamed).
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import yaml

from polypbench.paths import ARTIFACTS_ROOT, CONFIG_DIR

BOXES_ROOT = ARTIFACTS_ROOT / "boxes"
BOXES_CONFIG_DIR = CONFIG_DIR / "boxes"
COLUMNS = ["frame", "xmin", "ymin", "xmax", "ymax", "area", "peak", "mean"]


def load_boxes_config(name):
    path = BOXES_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in BOXES_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No boxes config for '{name}'. Available: {available}")
    with open(path) as f:
        return yaml.safe_load(f)


def scale_box(x0, y0, x1, y1, map_hw, frame_wh):
    """Inclusive box in map pixels -> inclusive box in frame pixels covering the same area.
    The model saw the whole frame resized to the map size (no letterbox), so each axis scales on its own."""
    (mh, mw), (w, h) = map_hw, frame_wh
    sx, sy = w / mw, h / mh
    return (int(np.floor(x0 * sx)), int(np.floor(y0 * sy)),
            min(w - 1, int(np.ceil((x1 + 1) * sx)) - 1), min(h - 1, int(np.ceil((y1 + 1) * sy)) - 1))


def extract_boxes(prob, frame_wh, threshold, connectivity=8, min_area_px=1):
    """prob: uint8 H x W map (0-255) as stored. Returns [(xmin, ymin, xmax, ymax, area, peak, mean), ...]."""
    mask = prob >= threshold * 255                      # same as prob / 255 >= threshold
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    if n == 1:
        return []
    labs, vals = labels[mask], prob[mask]
    sums = np.bincount(labs, weights=vals, minlength=n)
    peaks = np.zeros(n, np.uint8)
    np.maximum.at(peaks, labs, vals)
    boxes = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area >= min_area_px:
            boxes.append((*scale_box(x, y, x + bw - 1, y + bh - 1, prob.shape, frame_wh),
                          area / prob.size, peaks[k] / 255.0, sums[k] / area / 255.0))
    return boxes


def boxes_dir(dataset, model_id, postproc, boxes):
    return BOXES_ROOT / dataset / model_id / postproc / boxes


def boxes_path(dataset, model_id, postproc, boxes, sequence):
    return boxes_dir(dataset, model_id, postproc, boxes) / f"{sequence}.csv.gz"


def write_sequence_boxes(path, rows):
    """rows: [(frame, xmin, ymin, xmax, ymax, area, peak, mean), ...]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=COLUMNS).round({"area": 6, "peak": 4, "mean": 4})
    tmp = path.with_name(path.name + ".tmp")
    table.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def read_boxes(dataset, model_id, postproc, boxes, sequences):
    """One DataFrame for the given sequences, with a `video` column; raises if any is missing."""
    missing = [s for s in sequences if not boxes_path(dataset, model_id, postproc, boxes, s).exists()]
    if missing:
        raise FileNotFoundError(f"No boxes for {dataset}/{model_id}/{postproc}/{boxes}: {missing} "
                                f"(run scripts/extract_boxes.py)")
    tables = [pd.read_csv(boxes_path(dataset, model_id, postproc, boxes, s)).assign(video=s) for s in sequences]
    return pd.concat(tables, ignore_index=True)


def write_boxes_run_info(dataset, model_id, postproc, boxes, info):
    with open(boxes_dir(dataset, model_id, postproc, boxes) / "run.json", "w") as f:
        json.dump(info, f, indent=2)
