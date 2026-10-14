"""Inter-frame stability of saved predictions: per-transition CSVs + per-clip summaries.

Ported from LambdaDiceEval/log_interframe_iou.py, but computed from the
predictions saved by predict.py / predict_sali.py, so it covers every model
and post-processing variant without rerunning inference. See
polypbench/metrics/stability.py for the measures. Masks are the saved
probabilities binarized at --threshold (no morphological cleanup).

Each clip is read once, and the optical flow between consecutive frames is
computed once and shared by every model.

Writes:
    <artifacts>/stability/<dataset>/<model_id>/<postproc>.csv   one row per transition (frame k-1 -> k)
    <artifacts>/reports/<dataset>/stability/summary.csv         per model x postproc x clip (+ ALL)
    <artifacts>/reports/<dataset>/stability/<measure>_by_clip.csv   rows = model/postproc, cols = clips + ALL

Plots and side-by-side videos: scripts/visualize_stability.py.

Usage:
    python scripts/stability.py --dataset inhouse --models all
"""

import argparse

import cv2
import time

import pandas as pd

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.metrics.overlap import binarize
from polypbench.metrics.stability import TRANSITION_COLUMNS, FrameMotion, summarize_transitions, transition_metrics
from polypbench.models import resolve_models
from polypbench.paths import ARTIFACTS_ROOT

ALL = "ALL"
TABLES = {   # summary column -> file name of its by-clip table
    "iou_warped_mean": "iou_warped_mean",
    "iou_warped_p5": "iou_warped_p5",
    "iou_raw_mean": "iou_raw_mean",
    "n_appear_vanish": "appear_vanish",
}


def stability_dir(dataset_name):
    return ARTIFACTS_ROOT / "stability" / dataset_name


def score_sequence(dataset_name, seq, runs, threshold):
    """runs: list of (model_id, postproc). Returns {run: [row, ...]} for this sequence."""
    rows = {run: [] for run in runs}
    prev_rgb, prev_masks = None, {}
    for idx, rgb in seq.frames():
        masks = {}
        for model_id, postproc in runs:
            prob = predictions.read_prediction(
                predictions.frame_path(dataset_name, model_id, postproc, seq.name, idx))
            masks[(model_id, postproc)] = binarize(prob, threshold)
        if prev_rgb is not None:
            motion = FrameMotion(prev_rgb, rgb)
            for run in runs:
                rows[run].append({"sequence": seq.name, "frame": idx,
                                  **transition_metrics(motion, prev_masks[run], masks[run])})
        prev_rgb, prev_masks = rgb, masks
    return rows


def summarize(transitions, sequence_order):
    groups = []
    for (model_id, postproc), df in transitions.groupby(["model_id", "postproc"]):
        for seq_name, seq_df in df.groupby("sequence"):
            groups.append({"model_id": model_id, "postproc": postproc, "sequence": seq_name,
                           **summarize_transitions(seq_df)})
        groups.append({"model_id": model_id, "postproc": postproc, "sequence": ALL,
                       **summarize_transitions(df)})
    order = {name: i for i, name in enumerate(sequence_order + [ALL])}
    summary = pd.DataFrame(groups)
    return summary.sort_values(["model_id", "postproc", "sequence"],
                               key=lambda c: c.map(order) if c.name == "sequence" else c)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model ids, group names (configs/groups.yaml), or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none", "temporal_smooth"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for OpenCV (default: every core)")
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)

    model_ids = resolve_models(args.models)
    dataset = load_dataset(args.dataset)

    runs = []
    for model_id in model_ids:
        for postproc in args.postprocs:
            if predictions.is_complete(dataset.name, model_id, postproc):
                runs.append((model_id, postproc))
            else:
                print(f"[skip] {model_id} / {postproc}: no complete predictions")
    if runs:
        rows = {run: [] for run in runs}
        for seq in dataset:
            t0 = time.time()
            for run, seq_rows in score_sequence(dataset.name, seq, runs, args.threshold).items():
                rows[run].extend(seq_rows)
            print(f"[{seq.name}] {len(runs)} runs in {time.time() - t0:.1f}s")
        for (model_id, postproc), run_rows in rows.items():
            df = pd.DataFrame(run_rows)
            df.insert(0, "dataset", dataset.name)
            df.insert(3, "model_id", model_id)
            df.insert(4, "postproc", postproc)
            df.insert(5, "threshold", args.threshold)
            out = stability_dir(dataset.name) / model_id / f"{postproc}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)

    # Summarize every saved transition file at this threshold, not just this run's.
    saved = []
    for path in sorted(stability_dir(dataset.name).glob("*/*.csv")):
        df = pd.read_csv(path)
        if (df.threshold == args.threshold).all():
            saved.append(df)
        else:
            print(f"[not in summary] {path.name}: threshold {df.threshold.iloc[0]}, not {args.threshold}")
    if not saved:
        return

    sequence_order = [seq.name for seq in dataset]
    summary = summarize(pd.concat(saved, ignore_index=True), sequence_order)
    report_dir = ARTIFACTS_ROOT / "reports" / dataset.name / "stability"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "summary.csv", index=False)
    for column, name in TABLES.items():
        table = summary.pivot(index=["model_id", "postproc"], columns="sequence", values=column)
        table = table[[c for c in sequence_order + [ALL] if c in table.columns]]
        table.to_csv(report_dir / f"{name}_by_clip.csv")
        if column in ("iou_warped_mean", "n_appear_vanish"):
            print(f"\n{column} by clip (threshold {args.threshold}):")
            print(table.round(3).to_string())
    print(f"\nTransitions: {stability_dir(dataset.name)}\nReports: {report_dir}")


if __name__ == "__main__":
    main()
