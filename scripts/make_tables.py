"""Presentation tables (Markdown + CSV) from the saved reports.

Reads what evaluate.py, stability.py and eval_sunseg_toolbox.py wrote and
assembles compact tables, one file each, under <artifacts>/reports/tables/:

    inhouse_<group>.md/.csv       per model of a group (configs/groups.yaml):
                                  Dice per clip + overall, precision, recall,
                                  false-positive area on polyp-free frames, stability
    inhouse_trainsets.md/.csv     same architecture, ColonDB- vs SUN-SEG-trained
    sunseg_<metric>.md/.csv       SUN-SEG toolbox metric per model x test split

Missing results (not run yet) show as "-". Safe to rerun at any time.

Usage:
    python scripts/make_tables.py
"""

import argparse
import re

import pandas as pd

from polypbench.data import load_dataset
from polypbench.models import load_groups, resolve_models
from polypbench.paths import ARTIFACTS_ROOT

SPLITS = ["TestEasyDataset-Seen", "TestHardDataset-Seen", "TestEasyDataset-Unseen", "TestHardDataset-Unseen"]
SPLIT_LABELS = {"TestEasyDataset-Seen": "Easy/Seen", "TestHardDataset-Seen": "Hard/Seen",
                "TestEasyDataset-Unseen": "Easy/Unseen", "TestHardDataset-Unseen": "Hard/Unseen"}
TOOLBOX_METRICS = {"maxDice": "max Dice", "Smeasure": "S-alpha", "meanEm": "mean E-phi", "wFmeasure": "weighted F"}


def fmt(x, digits=3):
    return "-" if pd.isna(x) else f"{x:.{digits}f}"


def write(df, stem, title, note, out_dir):
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(fmt)
    lines = [f"### {title}", "", note, "", "| " + " | ".join(shown.columns) + " |",
             "|" + "|".join("---" for _ in shown.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n")


def inhouse_tables(out_dir, postproc):
    reports = ARTIFACTS_ROOT / "reports" / "inhouse"
    acc = pd.read_csv(reports / "summary.csv")
    stab = pd.read_csv(reports / "stability" / "summary.csv")
    clips = [s.name for s in load_dataset("inhouse")]

    def acc_mean(m, metric, clip="ALL"):
        r = acc[(acc.model_id == m) & (acc.postproc == postproc) & (acc.metric == metric) & (acc.sequence == clip)]
        return r["mean"].iloc[0] if len(r) else float("nan")

    def stab_mean(m):
        r = stab[(stab.model_id == m) & (stab.postproc == postproc) & (stab.sequence == "ALL")]
        return r["iou_warped_mean"].iloc[0] if len(r) else float("nan")

    for group in load_groups()["groups"]:
        rows = []
        for m in resolve_models([group]):
            rows.append({"model": m, **{c: acc_mean(m, "dice", c) for c in clips}, "Dice (all)": acc_mean(m, "dice"),
                         "precision": acc_mean(m, "precision"), "recall": acc_mean(m, "recall"),
                         "FP area, no-polyp frames": acc_mean(m, "fp_area_frac"), "stability": stab_mean(m)})
        write(pd.DataFrame(rows), f"inhouse_{group}", f"In-house video, group `{group}` ({postproc})",
              "Dice/precision/recall on the 109 annotated frames with a polyp (threshold 0.5); FP area = mean "
              "fraction of the frame marked on the 14 polyp-free frames (lower is better); stability = mean "
              "motion-compensated IoU between consecutive frames.", out_dir)

    rows = []
    for arch, ids in load_groups()["trainset_pairs"].items():
        for ts in ["colondb", "sunseg"]:
            m = ids[ts]
            rows.append({"architecture": arch, "trained on": {"colondb": "ColonDB", "sunseg": "SUN-SEG"}[ts],
                         "Dice (all)": acc_mean(m, "dice"), "precision": acc_mean(m, "precision"),
                         "recall": acc_mean(m, "recall"), "FP area, no-polyp frames": acc_mean(m, "fp_area_frac"),
                         "stability": stab_mean(m)})
    write(pd.DataFrame(rows), "inhouse_trainsets", f"In-house video: same architecture, different training data ({postproc})",
          "Each architecture trained on ColonDB (previous checkpoints) vs on SUN-SEG (our shared recipe).", out_dir)


def read_toolbox(model_id):
    """{split: {metric: value}} from <reports>/sunseg/toolbox/<model_id>/<split>_eval.txt."""
    out = {}
    for split in SPLITS:
        path = ARTIFACTS_ROOT / "reports" / "sunseg" / "toolbox" / model_id / f"{split}_eval.txt"
        if not path.exists():
            continue
        rows = [l for l in path.read_text().splitlines() if l.startswith("|")]
        header = [c.strip() for c in rows[0].strip("|").split("|")]
        values = [c.strip() for c in rows[1].strip("|").split("|")]
        if all(float(v) == 0 for v in values[2:]):   # an unfinished run's placeholder table
            continue
        out[split] = {h: float(v) for h, v in zip(header[2:], values[2:])}
    return out


def sunseg_tables(out_dir):
    model_ids = resolve_models(["sunseg", "colondb"])
    results = {m: read_toolbox(m) for m in model_ids}
    for metric, label in TOOLBOX_METRICS.items():
        rows = [{"model": m, **{SPLIT_LABELS[s]: results[m].get(s, {}).get(metric, float("nan")) for s in SPLITS}}
                for m in model_ids]
        write(pd.DataFrame(rows), f"sunseg_{metric}", f"SUN-SEG test splits: {label}",
              "Standard SUN-SEG evaluation toolbox (PNS+), per-frame min-max normalized predictions, as in "
              "SUN-SEG papers. Seen = test clips from videos with other clips in training; Unseen = new videos.",
              out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--postproc", default="none", help="in-house tables: none or temporal_smooth")
    args = parser.parse_args()
    out_dir = ARTIFACTS_ROOT / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    inhouse_tables(out_dir, args.postproc)
    sunseg_tables(out_dir)
    print(f"Tables written to {out_dir}")


if __name__ == "__main__":
    main()
