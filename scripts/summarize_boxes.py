"""Print facts about extracted REAL-Colon boxes, per model: a quick look, not the evaluation.

For each model (boxes from scripts/extract_boxes.py), over the given videos:
  1. how many boxes, per frame, and the share of frames with any box, by phase
     (positive, outside, inspection, after_removal, clean; see metrics/detection.py)
  2. area and score (peak, mean) of boxes touching a ground-truth box vs boxes on
     polyp-free frames
  3. for each score: the tau giving --target-fpr on polyp-free frames (all frames and
     in-patient only), with frame TPR and box sensitivity there
  4. per lesion at the in-patient mean-score tau: detected, hit fraction, latency

Use val videos: anything learned here informs evaluation choices.

Usage:
    python scripts/summarize_boxes.py --dataset realcolon_val --models emcad-b2@sunseg-ft-s0 cmsa-net@sunseg-ft-s0 \
        --sequences 004-012
"""

import argparse

import numpy as np
import pandas as pd

from polypbench.data.realcolon import SPLITS, load_realcolon_config
from polypbench.data.realcolon_gt import load_ground_truth
from polypbench.metrics import detection as d
from polypbench.models import resolve_models
from polypbench.postproc.boxes import read_boxes

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def quantiles(values, scale=1.0, digits=3):
    return " ".join(f"{v * scale:.{digits}f}" for v in values.quantile(QUANTILES)) if len(values) else "-"


def summarize(pred, gt, frames, args):
    print(f"boxes: {len(pred):,} on {pred[['video', 'frame']].drop_duplicates().shape[0]:,} of {len(frames):,} frames "
          f"({len(pred) / len(frames):.2f} per frame)")
    per_frame = pred.groupby(["video", "frame"]).size().rename("n_boxes").reset_index()
    f = frames.merge(per_frame, on=["video", "frame"], how="left").fillna({"n_boxes": 0})
    by_phase = f.groupby("phase").agg(frames=("n_boxes", "size"), any_box=("n_boxes", lambda n: (n > 0).mean()),
                                      boxes_per_frame=("n_boxes", "mean"), max_boxes=("n_boxes", "max"))
    print(by_phase.reindex([p for p in d.PHASES if p in by_phase.index]).round(3).to_string())

    pairs = pred.merge(gt.boxes, on=["video", "frame"], suffixes=("", "_gt"))
    touching = pairs[(pairs["xmin"] <= pairs["xmax_gt"]) & (pairs["xmin_gt"] <= pairs["xmax"])
                     & (pairs["ymin"] <= pairs["ymax_gt"]) & (pairs["ymin_gt"] <= pairs["ymax"])]
    touching = touching.drop_duplicates(["video", "frame", "xmin", "ymin", "xmax", "ymax"])
    negative = pred.merge(frames.loc[~frames["positive"], ["video", "frame"]], on=["video", "frame"])
    print(f"\nquantiles {' / '.join(f'{int(q * 100)}' for q in QUANTILES)}   "
          f"(touching GT: {len(touching):,} boxes, on polyp-free frames: {len(negative):,})")
    print(f"  area % of map  touching: {quantiles(touching['area'], 100)}   polyp-free: {quantiles(negative['area'], 100)}")
    for score in d.SCORES:
        print(f"  {score:13} touching: {quantiles(touching[score])}   polyp-free: {quantiles(negative[score])}")
    for score in d.SCORES:
        saturated = (negative[score] >= 1.0).groupby([negative["video"], negative["frame"]]).any()
        print(f"  polyp-free frames with a box of {score} = 1.0: {saturated.sum() / (~frames['positive']).sum():.1%}")

    print(f"\nat FPR <= {args.target_fpr:.0%} (min area {args.min_area:.2%} of map):")
    lesion_tau = None
    for score in d.SCORES:
        frame_scores, lesion_scores = d.score_boxes(pred, gt.boxes, score, args.min_area)
        scored = d.scored_frames(frames, frame_scores)
        lf = d.scored_lesion_frames(gt.boxes, lesion_scores)
        for scope, sub in (("all frames", scored), ("in-patient", scored[scored["in_patient"]])):
            tau = d.threshold_for_fpr(sub, args.target_fpr)
            r = d.frame_rates(sub, [tau], lf).iloc[0]
            print(f"  {score:4} {scope:10}  tau {tau:.4f}  FPR {r['fpr']:.3f}  frame TPR {r['tpr']:.3f}  "
                  f"box sensitivity {r['box_sensitivity']:.3f}")
            if score == "mean" and scope == "in-patient":
                lesion_tau = tau
    m = d.lesion_metrics(lf, gt.lesions, gt.videos, lesion_tau)
    cols = ["unique_id", "n_boxed_frames", "detected", "hit_frac", "persistent", "latency_s", "within_1s",
            "within_3s", "latency_excluded", "size_mm", "histology_class"]
    print(f"\nlesions at mean tau {lesion_tau:.4f} (in-patient):")
    print(m[cols].round(3).to_string(index=False) if len(m) else "  (no lesions in these videos)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(SPLITS))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--postproc", default="none")
    parser.add_argument("--boxes", default="cc-0.5")
    parser.add_argument("--sequences", nargs="+", help="videos (default: all of the split)")
    parser.add_argument("--after-removal-s", type=float, default=60)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--min-area", type=float, default=0.0, help="fraction of the map (default 0: keep all)")
    args = parser.parse_args()
    if args.dataset == "realcolon_test":
        print("WARNING: test videos. Don't choose scores, areas or thresholds from what this prints.\n")

    videos = args.sequences or load_realcolon_config()["splits"][SPLITS[args.dataset]]
    gt = load_ground_truth()
    gt.boxes = gt.boxes[gt.boxes["video"].isin(videos)]
    gt.lesions = gt.lesions[gt.lesions["video"].isin(videos)]
    frames = d.frame_table(gt, args.after_removal_s, videos)
    print(f"{args.dataset} {', '.join(videos)}: {len(frames):,} frames, {frames['positive'].sum():,} with a GT box, "
          f"{len(gt.lesions)} lesions; after_removal window {args.after_removal_s:g} s")
    for model_id in resolve_models(args.models):
        print(f"\n{'=' * 100}\n{model_id} / {args.postproc} / {args.boxes}\n{'=' * 100}")
        summarize(read_boxes(args.dataset, model_id, args.postproc, args.boxes, videos), gt, frames, args)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    np.seterr(invalid="ignore")
    main()
