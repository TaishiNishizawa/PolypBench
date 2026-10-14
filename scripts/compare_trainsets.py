"""Same architecture, different training data: ColonDB-trained vs SUN-SEG-trained.

Reads the reports written by evaluate.py and stability.py (run those first)
for the pairs in configs/groups.yaml (trainset_pairs), and writes two compact
tables to <artifacts>/reports/<dataset>/trainsets/:

    summary.csv        per architecture x postproc: Dice, false-positive area on
                       polyp-free frames, and stability, for each training set + change
    dice_by_clip.csv   per architecture x postproc x training set: Dice per clip + ALL

Usage:
    python scripts/compare_trainsets.py --dataset inhouse
"""

import argparse

import pandas as pd

from polypbench.data import load_dataset
from polypbench.models import load_groups
from polypbench.paths import ARTIFACTS_ROOT

TRAINSETS = ["colondb", "sunseg"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    report_dir = ARTIFACTS_ROOT / "reports" / args.dataset
    acc = pd.read_csv(report_dir / "summary.csv")
    stab = pd.read_csv(report_dir / "stability" / "summary.csv")
    clips = [seq.name for seq in load_dataset(args.dataset)] + ["ALL"]
    pairs = load_groups()["trainset_pairs"]

    def acc_mean(model_id, postproc, metric, clip="ALL"):
        row = acc[(acc.model_id == model_id) & (acc.postproc == postproc) & (acc.metric == metric) & (acc.sequence == clip)]
        return row["mean"].iloc[0] if len(row) else float("nan")

    def stab_mean(model_id, postproc):
        row = stab[(stab.model_id == model_id) & (stab.postproc == postproc) & (stab.sequence == "ALL")]
        return row["iou_warped_mean"].iloc[0] if len(row) else float("nan")

    summary, by_clip = [], []
    for arch, ids in pairs.items():
        for postproc in ["none", "temporal_smooth"]:
            row = {"architecture": arch, "postproc": postproc}
            for name, fn in [("dice", lambda m: acc_mean(m, postproc, "dice")),
                             ("fp_area", lambda m: acc_mean(m, postproc, "fp_area_frac")),
                             ("stability", lambda m: stab_mean(m, postproc))]:
                values = {ts: fn(ids[ts]) for ts in TRAINSETS}
                row.update({f"{name}_{ts}": values[ts] for ts in TRAINSETS})
                row[f"{name}_change"] = values["sunseg"] - values["colondb"]
            summary.append(row)
            for ts in TRAINSETS:
                by_clip.append({"architecture": arch, "postproc": postproc, "trained_on": ts, "model_id": ids[ts],
                                **{clip: acc_mean(ids[ts], postproc, "dice", clip) for clip in clips}})

    out_dir = report_dir / "trainsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(by_clip).to_csv(out_dir / "dice_by_clip.csv", index=False)

    print("Dice (frames with a polyp), FP area (polyp-free frames, lower is better), stability (motion-compensated IoU):")
    print(summary.round(3).to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
