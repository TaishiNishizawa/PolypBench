"""REAL-Colon ground truth for evaluation: three tables built once from the annotation
archives and the organizers' metadata (scripts/build_realcolon_gt.py), read with
load_ground_truth().

  boxes    one row per box: video, frame, unique_id, xmin, ymin, xmax, ymax, clipped.
           Original pixels, 0-based, inclusive, clipped to the frame (some annotated
           boxes extend past its edge; `clipped` marks them).
  lesions  one row per lesion: first / last boxed frame, boxed-frame count, longest
           gap, TRUE-Colon latency exclusion, size, site, histology, is_polyp.
  videos   one row per video: split, cohort, endoscope, fps, n_frames, frame size,
           in-patient range, lesion counts.

Facts only. Evaluation choices (phases such as "after removal", thresholds,
minimum areas) are derived from these when evaluating, so changing them never
means rebuilding the ground truth.
"""

from types import SimpleNamespace

import pandas as pd

from polypbench.data.realcolon import iter_annotations, load_realcolon_config, load_video_info
from polypbench.paths import ARTIFACTS_ROOT, DATA_ROOT

GT_DIR = ARTIFACTS_ROOT / "ground_truth" / "realcolon"
TABLES = ("boxes", "lesions", "videos")
SITE_SPELLING = {"caecum": "cecum", "trasnverse": "transverse"}   # variants in lesion_info.csv
# TRUE-Colon (Doerrich et al. 2026): a lesion is left out of the detection-latency mean if
# its annotations have a gap of more than 50 frames within its first 250 frames.
LATENCY_WINDOW, LATENCY_MAX_GAP = 250, 50


def latency_excluded(boxed_frames):
    """TRUE-Colon's rule on a lesion's sorted boxed frames: a gap > LATENCY_MAX_GAP missing
    frames between consecutive boxed frames, starting within the first LATENCY_WINDOW frames."""
    end = boxed_frames[0] + LATENCY_WINDOW
    return any(b - a - 1 > LATENCY_MAX_GAP for a, b in zip(boxed_frames, boxed_frames[1:]) if a < end)


def split_of(video, config):
    return next(split for split, videos in config["splits"].items() if video in videos)


def build_video(video, config=None):
    """(box rows, lesion rows, video row) for one video, from its annotation archive."""
    config = config or load_realcolon_config()
    info = load_video_info(config)[video]
    sizes, n_xml, rows = set(), 0, []
    for frame, (w, h), boxes in iter_annotations(video, config):
        n_xml += 1
        sizes.add((w, h))
        for uid, x0, y0, x1, y1 in boxes:
            c = (max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1))
            if not (c[0] < c[2] and c[1] < c[3]):
                raise ValueError(f"{video} frame {frame}: box {(x0, y0, x1, y1)} is empty inside a {w}x{h} frame")
            rows.append({"video": video, "frame": frame, "unique_id": uid,
                         "xmin": c[0], "ymin": c[1], "xmax": c[2], "ymax": c[3], "clipped": c != (x0, y0, x1, y1)})
    if n_xml != int(info["num_frames"]):
        raise ValueError(f"{video}: {n_xml} annotation files, video_info.csv says {info['num_frames']} frames")
    if len(sizes) != 1:
        raise ValueError(f"{video}: frame size changes within the video: {sorted(sizes)}")
    (width, height), = sizes

    lesions = []
    boxes = pd.DataFrame(rows, columns=["video", "frame", "unique_id", "xmin", "ymin", "xmax", "ymax", "clipped"])
    for uid, group in boxes.groupby("unique_id"):
        frames = sorted(group["frame"].unique())
        gaps = [b - a - 1 for a, b in zip(frames, frames[1:])]
        lesions.append({"unique_id": uid, "video": video, "first_frame": frames[0], "last_frame": frames[-1],
                        "n_boxed_frames": len(frames), "longest_gap": max(gaps, default=0),
                        "latency_excluded": latency_excluded(frames)})
    first, last = config["inside_patient"][video]
    video_row = {"video": video, "split": split_of(video, config), "cohort": video[:3],
                 "endoscope": info["endoscope_brand"], "fps": float(info["fps"]), "n_frames": n_xml,
                 "width": width, "height": height, "inside_first": first, "inside_last": last,
                 "n_lesions": len(lesions), "n_lesions_video_info": int(info["num_lesions"]),
                 "n_boxed_frames": int(boxes["frame"].nunique()) if rows else 0}
    return rows, lesions, video_row


def add_lesion_info(lesions, config=None):
    """Join size, site and histology from lesion_info.csv (outer join: nothing silently dropped)."""
    config = config or load_realcolon_config()
    info = pd.read_csv(DATA_ROOT / config["root"] / "lesion_info.csv", dtype=str).rename(columns={
        "unique_object_id": "unique_id", "unique_video_name": "video_info", "size [mm]": "size_mm"})
    info["size_mm"] = pd.to_numeric(info["size_mm"])
    info["site"] = info["site"].replace(SITE_SPELLING)
    merged = lesions.merge(info.drop(columns="video_info"), on="unique_id", how="outer", indicator=True)
    merged["boxed"] = merged["_merge"] != "right_only"
    merged["in_lesion_info"] = merged["_merge"] != "left_only"
    merged["is_polyp"] = merged["histology_class"].ne("NO POLYP") & merged["in_lesion_info"]
    return merged.drop(columns="_merge")


def save_ground_truth(boxes, lesions, videos):
    GT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in zip(TABLES, (boxes, lesions, videos)):
        table.to_csv(GT_DIR / f"{name}.csv.gz", index=False)


def load_ground_truth():
    """SimpleNamespace(boxes=, lesions=, videos=) DataFrames, as written by scripts/build_realcolon_gt.py."""
    missing = [n for n in TABLES if not (GT_DIR / f"{n}.csv.gz").exists()]
    if missing:
        raise FileNotFoundError(f"No REAL-Colon ground truth in {GT_DIR}: run scripts/build_realcolon_gt.py")
    ids = {"video": str, "unique_id": str, "cohort": str}
    return SimpleNamespace(**{n: pd.read_csv(GT_DIR / f"{n}.csv.gz", dtype=ids) for n in TABLES})
