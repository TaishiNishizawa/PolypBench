"""Score SUN-SEG predictions with the standard VPS evaluation toolbox, in parallel.

The toolbox (PNS+, github.com/GewelsJI/VPS; copy shipped in
third_party/SALI/eval) produces the numbers SUN-SEG papers report: S-alpha,
mean E-phi, weighted F, mean F, mean sensitivity, max Dice. It scores clips
one after another on one core (~1.5 s/frame, hours per split). This script
runs the toolbox's own code unchanged but scores clips in parallel:

  pass 1  run eval_engine_vps with the per-clip scorer swapped for a recorder,
          to collect exactly which frames of which clips it would score;
  pool    score those clips in parallel with the toolbox's own evaluator();
  pass 2  rerun eval_engine_vps with the scorer swapped for a lookup, so all
          averaging/max logic is the toolbox's own.

Predictions must follow the toolbox convention: <pred_dir>/<split>/<clip>/<frame>.png,
per-frame min-max normalized as SUN-SEG methods do. Runs in the `sali` env
(needs prettytable).

Usage:
    python scripts/eval_sunseg_toolbox.py --model_id sali@sunseg \\
        --pred_dir third_party/SALI/res/Net_epoch_8_best/SUN-SEG
    python scripts/eval_sunseg_toolbox.py --model_id sali@sunseg --pred_dir ... --splits TestEasyDataset/Seen
"""

import argparse
import os
import sys
import time
from argparse import Namespace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from polypbench.paths import ARTIFACTS_ROOT, DATA_ROOT, REPO_ROOT

TOOLBOX_DIR = REPO_ROOT / "third_party" / "SALI" / "eval"
SUNSEG_ROOT = DATA_ROOT / "SUN-SEG" / "SUN-SEG"
SPLITS = ["TestEasyDataset/Seen", "TestHardDataset/Seen", "TestEasyDataset/Unseen", "TestHardDataset/Unseen"]
METRICS = ["Smeasure", "meanEm", "wFmeasure", "meanFm", "meanSen", "maxDice"]   # as in SUN-SEG papers

sys.path.insert(0, str(TOOLBOX_DIR))
import evaluator as toolbox   # noqa: E402  (the toolbox imports its metrics module by name)

_score_clip = toolbox.evaluator


def score(args):
    gt, pred = args
    return _score_clip(gt_pth_lst=list(gt), pred_pth_lst=list(pred), metrics=METRICS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True, help="name for the results folder")
    parser.add_argument("--pred_dir", required=True, type=Path, help="folder containing <split>/<clip>/*.png")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    missing = [s for s in args.splits if not (pred_dir / s).is_dir()]
    if missing:
        raise FileNotFoundError(f"No predictions for {missing} under {pred_dir}")
    out_dir = ARTIFACTS_ROOT / "reports" / "sunseg" / "toolbox" / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def opt_for(split):
        return Namespace(gt_root=str(SUNSEG_ROOT), pred_root=str(pred_dir.parent), model_lst=[[pred_dir.name]],
                         data_lst=[[split]], metric_list=METRICS)

    os.chdir(TOOLBOX_DIR)   # the toolbox imports `metrics` relative to its own folder

    calls = []
    def recorder(gt_pth_lst, pred_pth_lst, metrics):
        calls.append((tuple(gt_pth_lst), tuple(pred_pth_lst)))
        return {m: (np.zeros(256) if ("max" in m or "mean" in m) else 0.0) for m in metrics}

    toolbox.evaluator = recorder
    dummy_dir = out_dir / ".pass1"
    dummy_dir.mkdir(exist_ok=True)
    for split in args.splits:
        toolbox.eval_engine_vps(opt_for(split), str(dummy_dir) + "/")
    print(f"{len(calls)} clips, {sum(len(c[0]) for c in calls)} frames to score", flush=True)

    t0 = time.time()
    order = sorted(range(len(calls)), key=lambda i: -len(calls[i][0]))   # longest clips first
    results = {}
    with Pool(args.workers) as pool:
        for n, (i, r) in enumerate(zip(order, pool.imap(score, [calls[i] for i in order])), 1):
            results[calls[i][0]] = r
            if n % 10 == 0 or n == len(calls):
                print(f"scored {n}/{len(calls)} clips ({time.time() - t0:.0f}s)", flush=True)

    toolbox.evaluator = lambda gt_pth_lst, pred_pth_lst, metrics: results[tuple(gt_pth_lst)]
    for split in args.splits:
        toolbox.eval_engine_vps(opt_for(split), str(out_dir) + "/")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
