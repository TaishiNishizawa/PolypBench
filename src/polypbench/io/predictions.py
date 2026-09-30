"""The on-disk prediction format shared by every stage.

One single-channel uint8 PNG per frame, holding the foreground probability
scaled to 0-255 at the frame's original resolution:

    <artifacts_root>/predictions/<dataset>/<model_id>/<postproc>/<sequence>/<frame:06d>.png

plus a run.json in <dataset>/<model_id>/<postproc>/, written only once every
sequence is done, so its presence marks a complete run.
"""

import json

import cv2
import numpy as np

from polypbench.paths import ARTIFACTS_ROOT

PREDICTIONS_ROOT = ARTIFACTS_ROOT / "predictions"


def run_dir(dataset, model_id, postproc):
    return PREDICTIONS_ROOT / dataset / model_id / postproc


def frame_path(dataset, model_id, postproc, sequence, index):
    return run_dir(dataset, model_id, postproc) / sequence / f"{index:06d}.png"


def write_prediction(path, prob):
    """prob: float HxW in [0, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Failed to write {path}")


def read_prediction(path):
    """Return the probability map as float32 HxW in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def write_run_info(dataset, model_id, postproc, info):
    path = run_dir(dataset, model_id, postproc) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def is_complete(dataset, model_id, postproc):
    return (run_dir(dataset, model_id, postproc) / "run.json").exists()
