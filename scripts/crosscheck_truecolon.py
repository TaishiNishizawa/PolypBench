"""Cross-check our detection metrics against TRUE-Colon's own evaluator (step 6d).

Writes the REAL-Colon ground truth and one model's boxes as COCO files (original
pixels; inclusive boxes become xywh with w = xmax - xmin + 1, so boxes sharing a
pixel still have IoU > 0), runs scripts/truecolon_eval.py in the `truecolon` env,
and compares, on the frames of one scope, at each --taus and at the matched
operating point:

    frame   TP / FP / TN / FN
    lesion  per lesion: visible and hit frames, detected, persistent (25% / 50%),
            latency in frames, "within 1 / 3 / 5 s" in TRUE-Colon's sense (tc_*);
            their mean latency (frames, without their two excluded lesions)
    tau     the lowest tau with FPR <= --target-fpr

Every row should say "ok". Box score: mean_prob (their thresholds must be > 0).
COCO files go next to the boxes (dm-3): <boxes run>/truecolon_<scope>/.

Usage:
    python scripts/crosscheck_truecolon.py --dataset realcolon_val --model emcad-b2@sunseg-ft-s0 --sequences 004-012
"""

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from polypbench.data.realcolon import SPLITS, load_realcolon_config
from polypbench.data.realcolon_gt import load_ground_truth
from polypbench.metrics import detection as d
from polypbench.paths import REPO_ROOT
from polypbench.postproc.boxes import boxes_dir, read_boxes

TRUECOLON_PYTHON = "/mnt/dm-3/taishi/envs/truecolon/bin/python"
SCORE = "mean_prob"


def xywh(t):
    return np.stack([t["xmin"], t["ymin"], t["xmax"] - t["xmin"] + 1, t["ymax"] - t["ymin"] + 1], axis=1).tolist()


def write_coco(frames, gt_boxes, pred, out_dir):
    """GT (images = the scope's frames, annotations carry unique_id) and predictions on those frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = (frames["video"] + "_" + frames["frame"].astype(str)).tolist()
    image_id = dict(zip(zip(frames["video"], frames["frame"]), range(len(frames))))
    g = gt_boxes[[k in image_id for k in zip(gt_boxes["video"], gt_boxes["frame"])]]
    annotations = [{"id": i, "image_id": image_id[(v, f)], "category_id": 1, "bbox": b, "area": b[2] * b[3],
                    "iscrowd": 0, "unique_id": u}
                   for i, (v, f, u, b) in enumerate(zip(g["video"], g["frame"], g["unique_id"], xywh(g)))]
    gt = {"images": [{"id": i, "file_name": f"{n}.jpg"} for i, n in enumerate(names)],
          "annotations": annotations, "categories": [{"id": 1, "name": "polyp"}]}
    p = pred.merge(frames[["video", "frame"]], on=["video", "frame"])
    preds = [{"image_id": f"{v}_{f}", "category_id": 1, "bbox": b, "score": float(s)}
             for v, f, b, s in zip(p["video"], p["frame"], xywh(p), p[SCORE])]
    for name, obj in (("gt.json", gt), ("pred.json", preds)):
        with open(out_dir / name, "w") as fh:
            json.dump(obj, fh)
    return out_dir / "gt.json", out_dir / "pred.json", len(g) < len(gt_boxes)


def compare(rows, what, ours, theirs):
    same = (ours == theirs) if not isinstance(ours, float) else bool(np.isclose(ours, theirs, equal_nan=True))
    rows.append({"what": what, "ours": ours, "true_colon": theirs, "ok": "ok" if same else "DIFFERENT"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["realcolon_val", "realcolon_test"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--postproc", default="none")
    parser.add_argument("--boxes", default="cc-0.5")
    parser.add_argument("--sequences", nargs="+", help="videos (default: all of the split)")
    parser.add_argument("--scope", default="in_patient", choices=["in_patient", "all"])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.8, 0.9, 0.95, 0.99])
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()

    videos = args.sequences or load_realcolon_config()["splits"][SPLITS[args.dataset]]
    gt = load_ground_truth()
    gt.boxes = gt.boxes[gt.boxes["video"].isin(videos)]
    gt.lesions = gt.lesions[gt.lesions["video"].isin(videos)]
    frames = d.frame_table(gt, 60, videos)
    if args.scope == "in_patient":
        frames = frames[frames["in_patient"]]
    pred = read_boxes(args.dataset, args.model, args.postproc, args.boxes, videos)
    out_dir = boxes_dir(args.dataset, args.model, args.postproc, args.boxes) / f"truecolon_{args.scope}"
    gt_path, pred_path, boxes_dropped = write_coco(frames, gt.boxes, pred, out_dir)
    if boxes_dropped:
        print(f"note: some GT boxes fall outside the {args.scope} frames; their lesions differ by design")

    run = subprocess.run([TRUECOLON_PYTHON, str(REPO_ROOT / "scripts" / "truecolon_eval.py"), "--gt", str(gt_path),
                          "--pred", str(pred_path), "--taus", *map(str, args.taus),
                          "--target-fpr", str(args.target_fpr)], capture_output=True, text=True)
    if run.returncode:
        sys.exit(run.stderr)
    theirs = json.loads(run.stdout)

    frame_scores, lesion_scores = d.score_boxes(pred, gt.boxes, SCORE)
    scored = d.scored_frames(frames, frame_scores)
    lf = d.scored_lesion_frames(gt.boxes, lesion_scores)
    pos, neg = scored[scored["positive"]], scored[~scored["positive"]]
    rows = []
    for at in theirs["at"]:
        tau, f, les = at["tau"], at["frame"], at["lesion"]
        tp, fp = int((pos["hit"] >= tau).sum()), int((neg["alarm"] >= tau).sum())
        for name, value in (("tp", tp), ("fp", fp), ("tn", len(neg) - fp), ("fn", len(pos) - tp)):
            compare(rows, f"tau {tau:g}  frame {name}", value, f[name])
        m = d.lesion_metrics(lf, gt.lesions, gt.videos, tau).set_index("unique_id")
        for r in les["records"]:
            o = m.loc[r["lesion_id"]]
            lat = None if pd.isna(o["latency_frames"]) else int(o["latency_frames"])
            for name, value, their in (
                    ("visible frames", int(o["n_boxed_frames"]), r["n_frames_visible"]),
                    ("hit frames", int(o["n_hit_frames"]), r["n_frames_detected"]),
                    ("detected", bool(o["detected"]), r["detected_any"]),
                    ("persistent 25%", bool(o["tc_persistent"]), r["detected_25pct"]),
                    ("persistent 50%", bool(o["persistent"]), r["detected_50pct"]),
                    ("latency frames", lat, r["latency_frames"]),
                    *((f"within {s}s (tc)", bool(o[f"tc_within_{s}s"]), bool(r[f"detected_within_{s}s"]))
                      for s in d.WITHIN_S)):
                compare(rows, f"tau {tau:g}  {r['lesion_id']} {name}", value, their)
        det = m[m["detected"] & ~m["tc_latency_excluded"]]
        theirs_mean = les["summary"]["mean_latency_frames"]
        compare(rows, f"tau {tau:g}  mean latency frames (tc exclusions)",
                float(det["latency_frames"].mean()), float("nan") if theirs_mean is None else float(theirs_mean))
    compare(rows, f"tau for FPR <= {args.target_fpr:g}", d.threshold_for_fpr(scored, args.target_fpr),
            theirs["matched"]["threshold"])

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(f"{args.model} on {args.dataset} {', '.join(videos)} ({args.scope}: {len(frames):,} frames, "
          f"{len(pos):,} positive), true-colon {theirs['version']}")
    print(table.to_string(index=False))
    bad = table[table["ok"] != "ok"]
    print(f"\n{len(table) - len(bad)} / {len(table)} agree")
    sys.exit(1 if len(bad) else 0)


if __name__ == "__main__":
    main()
