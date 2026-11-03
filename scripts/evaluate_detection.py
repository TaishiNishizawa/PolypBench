"""Box-level (detection) metrics on REAL-Colon, per model, from extracted boxes.

Reads the box tables written by scripts/extract_boxes.py and scores them with
polypbench/metrics/detection.py (TRUE-Colon's definitions, our phases and lesion
latency, plus TRUE-Colon's lesion variants as tc_* columns). Two frame scopes:
"all" frames and "in_patient" frames only. Thresholds:

    mask      every box counts (the mask binarized at the boxes config's threshold)
    val_fpr   tau giving --target-fpr on the val videos (primary). On realcolon_val it is
              chosen in-sample and saved to thresholds.json; realcolon_test reads that file
    test_fpr  realcolon_test only: tau giving --target-fpr on test (comparison only, as
              TRUE-Colon reported)

Writes, per model / postproc / boxes / score:

    <artifacts>/metrics/<dataset>/<model_id>/<postproc>/<boxes>/<score>/
        summary.csv      one row per scope x threshold: tau, TPR, FPR, box sensitivity,
                         lesion fractions (detected, persistent, within 1/3/5 s) and latency
        lesion.csv       one row per scope x threshold x lesion (size, histology, latency, ...)
        phase.csv        false-alarm rate by phase (outside, inspection, after_removal, clean)
        curve.csv        TPR / FPR / box sensitivity over tau, per scope
        thresholds.json  (val) the val_fpr tau per scope, read when evaluating test
        run.json         parameters, videos, git commit

    <artifacts>/reports/<dataset>/detection_<boxes>_<score>.csv   every model's summary rows

<score> is the box score, with "_area<a>" appended when --min-area is not 0. With
--sequences (a subset of the videos), <dataset> becomes <dataset>_<video>+<video>...,
so subsets never overwrite or stand in for full-split results.

Usage:
    python scripts/evaluate_detection.py --dataset realcolon_val --models emcad-b2@sunseg-ft-s0 cmsa-net@sunseg-ft-s0
    python scripts/evaluate_detection.py --dataset realcolon_val --models emcad-b2@sunseg-ft-s0 --sequences 004-012
"""

import argparse
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from polypbench.data.realcolon import SPLITS, load_realcolon_config
from polypbench.data.realcolon_gt import load_ground_truth
from polypbench.metrics import detection as d
from polypbench.models import resolve_models
from polypbench.paths import ARTIFACTS_ROOT, REPO_ROOT
from polypbench.postproc.boxes import load_boxes_config, read_boxes

METRICS_ROOT = ARTIFACTS_ROOT / "metrics"
REPORTS_ROOT = ARTIFACTS_ROOT / "reports"
SHOW = ["scope", "threshold", "tau", "fpr", "tpr", "box_sensitivity", "n_lesions", "detected", "persistent",
        "median_latency_s", "within_3s"]


def git_commit():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    return commit + ("-dirty" if dirty else "")


def score_name(score, min_area):
    return score if min_area == 0 else f"{score}_area{min_area:g}"


def run_dir(dataset_dir, model_id, postproc, boxes, score_dir):
    return METRICS_ROOT / dataset_dir / model_id / postproc / boxes / score_dir


def load_val_taus(model_id, postproc, boxes, score_dir, target_fpr):
    path = run_dir("realcolon_val", model_id, postproc, boxes, score_dir) / "thresholds.json"
    if not path.exists():
        raise SystemExit(f"No val thresholds for {model_id}/{postproc}/{boxes}/{score_dir}: "
                         f"evaluate realcolon_val (all videos) first")
    with open(path) as f:
        saved = json.load(f)
    if saved["target_fpr"] != target_fpr:
        raise SystemExit(f"{path} was chosen for FPR {saved['target_fpr']}, not {target_fpr}")
    return saved["tau"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["realcolon_val", "realcolon_test"])
    parser.add_argument("--models", nargs="+", required=True, help="model ids, group names, or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none"])
    parser.add_argument("--boxes", default="cc-0.5", help="configs/boxes/<name>.yaml (default cc-0.5)")
    parser.add_argument("--score", default="mean_prob", choices=d.SCORES, help="box score (default mean_prob)")
    parser.add_argument("--min-area", type=float, default=0.0, help="drop boxes below this fraction of the map")
    parser.add_argument("--after-removal-s", type=float, default=60)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--sequences", nargs="+", help="only these videos (results go to a separate folder)")
    parser.add_argument("--curve-points", type=int, default=200)
    args = parser.parse_args()

    all_videos = load_realcolon_config()["splits"][SPLITS[args.dataset]]
    videos = args.sequences or all_videos
    if unknown := sorted(set(videos) - set(all_videos)):
        raise SystemExit(f"Not in {args.dataset}: {unknown}")
    subset = set(videos) != set(all_videos)
    dataset_dir = f"{args.dataset}_{'+'.join(videos)}" if subset else args.dataset
    is_test = args.dataset == "realcolon_test"
    score_dir = score_name(args.score, args.min_area)
    mask_threshold = load_boxes_config(args.boxes)["threshold"]

    gt = load_ground_truth()
    gt.boxes = gt.boxes[gt.boxes["video"].isin(videos)]
    gt.lesions = gt.lesions[gt.lesions["video"].isin(videos)]
    frames = d.frame_table(gt, args.after_removal_s, videos)
    print(f"{dataset_dir}: {len(videos)} videos, {len(frames):,} frames, {frames['positive'].sum():,} with a GT box, "
          f"{len(gt.lesions)} lesions; score {args.score}, min area {args.min_area:g}, FPR target {args.target_fpr:g}")

    started = datetime.now().isoformat(timespec="seconds")
    for model_id in resolve_models(args.models):
        for postproc in args.postprocs:
            val_taus = load_val_taus(model_id, postproc, args.boxes, score_dir, args.target_fpr) if is_test else None
            pred = read_boxes(args.dataset, model_id, postproc, args.boxes, videos)
            res = d.evaluate_boxes(pred, gt, frames, args.score, args.min_area, mask_threshold, args.target_fpr,
                                   val_taus, args.curve_points)
            out = run_dir(dataset_dir, model_id, postproc, args.boxes, score_dir)
            out.mkdir(parents=True, exist_ok=True)
            keys = {"model_id": model_id, "postproc": postproc, "boxes": args.boxes, "score": score_dir}
            for name in ("summary", "lesion", "phase", "curve"):
                res[name].assign(**keys).to_csv(out / f"{name}.csv", index=False)
            if not is_test:
                with open(out / "thresholds.json", "w") as f:
                    json.dump({"target_fpr": args.target_fpr, "tau": res["chosen"], "videos": videos}, f, indent=2)
            with open(out / "run.json", "w") as f:
                json.dump({"dataset": args.dataset, "videos": videos, **keys, "min_area": args.min_area,
                           "mask_threshold": mask_threshold, "after_removal_s": args.after_removal_s,
                           "target_fpr": args.target_fpr,
                           "val_fpr_source": "realcolon_val thresholds.json" if is_test else "chosen in-sample",
                           "git_commit": git_commit(), "started": started,
                           "finished": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
            print(f"\n{model_id} / {postproc} -> {out}")
            print(res["summary"][SHOW].round(4).to_string(index=False))

    found = sorted((METRICS_ROOT / dataset_dir).glob(f"*/*/{args.boxes}/{score_dir}/summary.csv"))
    report = REPORTS_ROOT / dataset_dir / f"detection_{args.boxes}_{score_dir}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_csv(p) for p in found], ignore_index=True).to_csv(report, index=False)
    print(f"\n{len(found)} runs -> {report}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    np.seterr(invalid="ignore")
    main()
