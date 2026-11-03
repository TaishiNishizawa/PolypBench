"""Run TRUE-Colon's own evaluator on COCO files; print its numbers as JSON.

Runs in the separate `truecolon` env (Python >= 3.11, `true-colon` from git; see
Paper_Progress.md), so it imports nothing from polypbench. Called by
scripts/crosscheck_truecolon.py, which writes the COCO files and compares.

Usage:
    /mnt/dm-3/taishi/envs/truecolon/bin/python scripts/truecolon_eval.py --gt gt.json --pred pred.json \
        --taus 0.5 0.9 --target-fpr 0.05
"""

import argparse
import dataclasses
import json

import true_colon as tc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--taus", nargs="+", type=float, required=True)
    parser.add_argument("--target-fpr", type=float, required=True)
    args = parser.parse_args()

    gt = tc._ingest.load_coco_gt(args.gt)
    with open(args.pred) as f:
        preds = json.load(f)
    out = {"version": tc.__version__ if hasattr(tc, "__version__") else None, "at": []}
    for tau in args.taus:
        ev = tc.create_evaluator(dataset="realcolon", id_source="pred_string", conf=tau,
                                 tiers=frozenset({"frame", "lesion"}))
        r = ev.evaluate(gt, preds)
        out["at"].append({"tau": tau, "frame": dataclasses.asdict(r.frame),
                          "lesion": dataclasses.asdict(r.lesion), "provenance": r.provenance})
    ev = tc.create_evaluator(dataset="realcolon", id_source="pred_string")
    out["matched"] = dataclasses.asdict(ev.match_operating_point(gt, preds, target_fpr=args.target_fpr))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
