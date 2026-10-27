"""Build the REAL-Colon ground-truth tables (boxes, lesions, videos) for all 60 videos.

Reads every per-frame annotation XML once (from the archives under
<data_root>/REAL-Colon/annotations) plus video_info.csv and lesion_info.csv, and
writes <artifacts>/ground_truth/realcolon/{boxes,lesions,videos}.csv.gz and a
run.json. See polypbench/data/realcolon_gt.py for what each table holds. Fails
loudly on anything unexpected (frame count or size mismatch, empty boxes).
tests/test_realcolon_gt.py checks the result against the published counts.

Takes a few minutes (2.76M small XML files), one video per worker process.

Usage:
    python scripts/build_realcolon_gt.py
"""

import argparse
import json
import subprocess
from datetime import datetime
from multiprocessing import Pool

import pandas as pd

from polypbench.data.realcolon import load_realcolon_config
from polypbench.data.realcolon_gt import GT_DIR, add_lesion_info, build_video, save_ground_truth
from polypbench.paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = load_realcolon_config()
    videos = [v for split in ("train", "val", "test") for v in config["splits"][split]]
    started = datetime.now().isoformat(timespec="seconds")
    boxes, lesions, rows = [], [], []
    with Pool(args.workers) as pool:
        for video_boxes, video_lesions, video_row in pool.imap(build_video, videos):
            boxes += video_boxes
            lesions += video_lesions
            rows.append(video_row)
            print(f"{video_row['video']}: {video_row['n_frames']:,} frames, {video_row['n_boxed_frames']:,} boxed, "
                  f"{video_row['n_lesions']} lesions", flush=True)

    boxes = pd.DataFrame(boxes).sort_values(["video", "frame", "unique_id"], ignore_index=True)
    lesions = add_lesion_info(pd.DataFrame(lesions), config).sort_values("unique_id", ignore_index=True)
    videos = pd.DataFrame(rows)
    save_ground_truth(boxes, lesions, videos)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    summary = videos.groupby("split")[["n_frames", "n_boxed_frames", "n_lesions"]].sum().to_dict("index")
    with open(GT_DIR / "run.json", "w") as f:
        json.dump({"git_commit": commit, "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
                   "n_boxes": len(boxes), "n_clipped": int(boxes["clipped"].sum()), "per_split": summary}, f, indent=2)
    print(f"\nWrote {GT_DIR}: {len(boxes):,} boxes ({int(boxes['clipped'].sum()):,} clipped), "
          f"{len(lesions)} lesions, {len(videos)} videos")
    print(pd.DataFrame(summary).T.to_string())
    unmatched = lesions[~lesions["boxed"] | ~lesions["in_lesion_info"]]
    if len(unmatched):
        print("\nLesions not both boxed and in lesion_info.csv:\n" + unmatched.to_string())


if __name__ == "__main__":
    main()
