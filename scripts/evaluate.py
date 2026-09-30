"""Score saved predictions against ground truth, per frame, then summarize per clip.

Reads predictions written by predict.py, binarizes them at --threshold, and
scores every annotated frame (see polypbench/metrics/overlap.py for how
empty-ground-truth frames are handled). Writes:

    <artifacts>/metrics/<dataset>/<model_id>/<postproc>.csv    one row per frame x metric
    <artifacts>/reports/<dataset>/summary.csv                  mean/std/n per model x postproc x clip x metric
    <artifacts>/reports/<dataset>/<metric>_by_clip.csv         one table per metric: rows = model/postproc, cols = clips + ALL

"ALL" pools every annotated frame across clips (not the mean of the clip means).

Usage:
    python scripts/evaluate.py --dataset inhouse --models all
    python scripts/evaluate.py --dataset inhouse --models sacnet-b2@colondb --postprocs none --threshold 0.5
"""

import argparse

import pandas as pd

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.metrics.overlap import EMPTY_METRICS, POLYP_METRICS, binarize, frame_metrics
from polypbench.models import MODELS_CONFIG_DIR
from polypbench.paths import ARTIFACTS_ROOT

ALL = "ALL"


def score_run(dataset, model_id, postproc, threshold):
    rows = []
    for seq in dataset:
        for idx in sorted(seq.annotations):
            prob = predictions.read_prediction(predictions.frame_path(dataset.name, model_id, postproc, seq.name, idx))
            gt = seq.load_mask(idx)
            for metric, value in frame_metrics(binarize(prob, threshold), gt).items():
                rows.append({
                    "dataset": dataset.name, "sequence": seq.name, "frame": idx,
                    "model_id": model_id, "postproc": postproc, "threshold": threshold,
                    "gt_empty": not gt.any(), "metric": metric, "value": value,
                })
    return pd.DataFrame(rows)


def summarize(frames, sequence_order):
    """Mean/std/n per model x postproc x clip x metric, plus pooled ALL rows."""
    keys = ["model_id", "postproc", "metric"]
    per_clip = frames.groupby(keys + ["sequence"])["value"].agg(["mean", "std", "count"]).reset_index()
    pooled = frames.groupby(keys)["value"].agg(["mean", "std", "count"]).reset_index().assign(sequence=ALL)
    summary = pd.concat([per_clip, pooled], ignore_index=True).rename(columns={"count": "n"})
    order = {name: i for i, name in enumerate(sequence_order + [ALL])}
    return summary.sort_values(keys + ["sequence"], key=lambda c: c.map(order) if c.name == "sequence" else c)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model ids, or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none", "temporal_smooth"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model_ids = args.models
    if model_ids == ["all"]:
        model_ids = sorted(p.stem for p in MODELS_CONFIG_DIR.glob("*.yaml"))

    dataset = load_dataset(args.dataset)
    metrics_dir = ARTIFACTS_ROOT / "metrics" / dataset.name
    all_frames = []
    for model_id in model_ids:
        for postproc in args.postprocs:
            if not predictions.is_complete(dataset.name, model_id, postproc):
                print(f"[skip] {model_id} / {postproc}: no complete predictions (run predict.py first)")
                continue
            frames = score_run(dataset, model_id, postproc, args.threshold)
            out = metrics_dir / model_id / f"{postproc}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            frames.to_csv(out, index=False)
            all_frames.append(frames)
            print(f"[scored] {model_id} / {postproc} -> {out}")
    if not all_frames:
        return

    sequence_order = [seq.name for seq in dataset]
    summary = summarize(pd.concat(all_frames, ignore_index=True), sequence_order)
    report_dir = ARTIFACTS_ROOT / "reports" / dataset.name
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "summary.csv", index=False)

    for metric in POLYP_METRICS + EMPTY_METRICS:
        table = (summary[summary.metric == metric]
                 .pivot(index=["model_id", "postproc"], columns="sequence", values="mean"))
        table = table[[c for c in sequence_order + [ALL] if c in table.columns]]
        table.to_csv(report_dir / f"{metric}_by_clip.csv")
        if metric in ("dice", "false_alarm"):
            what = "frames with a polyp" if metric == "dice" else "frames without a polyp (lower is better)"
            print(f"\n{metric} by clip, threshold {args.threshold}, {what}:")
            print(table.round(3).to_string())
    print(f"\nReports written to {report_dir}")


if __name__ == "__main__":
    main()
