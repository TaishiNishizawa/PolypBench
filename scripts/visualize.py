"""Visualize saved predictions: overlay videos and per-clip comparison sheets.

Videos: one per model x clip. Every frame shows the prediction (green); on
annotated frames the ground truth (red outline) and that frame's Dice are
added.

Sheets: one PNG per clip. Rows are the clip's annotated frames; columns are
the source frame, the ground truth, then each model's prediction with its
Dice (or its false-positive area on frames with no polyp).

Outputs, under <artifacts>/reports/<dataset>/:
    videos/<postproc>/<model_id>/<clip>.mp4
    sheets/<postproc>/<clip>.png

Usage:
    python scripts/visualize.py --dataset inhouse --models all
    python scripts/visualize.py --dataset inhouse --models sacnet-b2@colondb emcad@colondb --no-videos
"""

import argparse

import cv2
import numpy as np

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.metrics.overlap import binarize, frame_metrics
from polypbench.models import MODELS_CONFIG_DIR
from polypbench.paths import ARTIFACTS_ROOT
from polypbench.viz.overlays import GT_COLOR, PRED_COLOR, VideoWriter, header_cell, label, overlay

SHEET_CELL = (320, 240)   # width, height of each sheet cell


def score_text(pred, gt):
    m = frame_metrics(pred, gt)
    if "dice" in m:
        return f"Dice {m['dice']:.3f}"
    return f"no polyp, FP {100 * m['fp_area_frac']:.1f}%"


def render_videos(dataset, seq, model_ids, postproc, threshold, out_root):
    writers = {m: VideoWriter(out_root / "videos" / postproc / m / f"{seq.name}.mp4", seq.fps, *seq_size(seq))
               for m in model_ids}
    try:
        for idx, rgb in seq.frames():
            gt = seq.load_mask(idx) if idx in seq.annotations else None
            for model_id, writer in writers.items():
                prob = predictions.read_prediction(
                    predictions.frame_path(dataset.name, model_id, postproc, seq.name, idx))
                pred = binarize(prob, threshold)
                img = overlay(rgb, pred, PRED_COLOR)
                lines = [f"{model_id} | {postproc}", f"frame {idx}"]
                if gt is not None:
                    img = overlay(img, gt, GT_COLOR, fill_alpha=0, thickness=2)
                    lines.append(f"annotated: {score_text(pred, gt)}")
                writer.write(label(img, lines))
    finally:
        for writer in writers.values():
            writer.close()


def seq_size(seq):
    """(width, height) of the sequence's frames, from its first annotation mask."""
    h, w = seq.load_mask(next(iter(seq.annotations))).shape
    return w, h


def render_sheet(dataset, seq, model_ids, postproc, threshold, out_path):
    cw, ch = SHEET_CELL
    columns = ["source", "ground truth"] + model_ids
    rows = [np.hstack([header_cell(c, cw) for c in columns])]
    for idx, rgb in seq.frames():
        if idx not in seq.annotations:
            continue
        gt = seq.load_mask(idx)
        cells = [
            label(rgb, [f"frame {idx}", seq.annotations[idx].name]),
            label(overlay(rgb, gt, GT_COLOR), ["no polyp" if not gt.any() else "polyp"]),
        ]
        for model_id in model_ids:
            prob = predictions.read_prediction(
                predictions.frame_path(dataset.name, model_id, postproc, seq.name, idx))
            pred = binarize(prob, threshold)
            img = overlay(overlay(rgb, pred, PRED_COLOR), gt, GT_COLOR, fill_alpha=0, thickness=1)
            cells.append(label(img, [score_text(pred, gt)]))
        rows.append(np.hstack([cv2.resize(c, (cw, ch), interpolation=cv2.INTER_AREA) for c in cells]))
    sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model ids, or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none", "temporal_smooth"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sequences", nargs="+", help="only these clips (default: all)")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    model_ids = args.models
    if model_ids == ["all"]:
        model_ids = sorted(p.stem for p in MODELS_CONFIG_DIR.glob("*.yaml"))
    dataset = load_dataset(args.dataset)
    out_root = ARTIFACTS_ROOT / "reports" / dataset.name

    for postproc in args.postprocs:
        ready = [m for m in model_ids if predictions.is_complete(dataset.name, m, postproc)]
        for m in set(model_ids) - set(ready):
            print(f"[skip] {m} / {postproc}: no complete predictions")
        if not ready:
            continue
        for seq in dataset:
            if args.sequences and seq.name not in args.sequences:
                continue
            if not args.no_sheets:
                path = out_root / "sheets" / postproc / f"{seq.name}.png"
                render_sheet(dataset, seq, ready, postproc, args.threshold, path)
                print(f"[sheet] {path}")
            if not args.no_videos:
                render_videos(dataset, seq, ready, postproc, args.threshold, out_root)
                print(f"[videos] {postproc} / {seq.name}: {len(ready)} models")


if __name__ == "__main__":
    main()
