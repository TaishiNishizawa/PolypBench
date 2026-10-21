"""Loader for REAL-Colon (configs/datasets/realcolon.yaml).

Each official split is its own dataset: realcolon_train, realcolon_val,
realcolon_test. One sequence per video, named by its id (e.g. 001-013),
covering every frame from 0, including frames outside the patient; in-patient
trimming is applied at evaluation (see inside_patient).

Frame paths are built from num_frames in video_info.csv instead of listing
the folders: frame ids are 0-based and contiguous, and listing ~1M files over
NFS is slow. A missing frame raises when it is read (tests/test_realcolon.py
checks the folders match).

Ground truth is boxes, not masks, so sequences have no mask annotations.

Usage:
    from polypbench.data.realcolon import load_realcolon
    dataset = load_realcolon("realcolon_test")
"""

import csv

import yaml

from polypbench.data.base import Dataset, FrameFolderSequence
from polypbench.paths import CONFIG_DIR, DATA_ROOT

CONFIG_PATH = CONFIG_DIR / "datasets" / "realcolon.yaml"
SPLITS = {
    "realcolon_train": "train",
    "realcolon_val": "val",
    "realcolon_test": "test",
}


def load_realcolon_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_video_info(config=None):
    """{video: row of video_info.csv}, e.g. row["fps"], row["num_frames"], row["endoscope_brand"]."""
    config = config or load_realcolon_config()
    with open(DATA_ROOT / config["root"] / config["video_info"], newline="", encoding="utf-8-sig") as f:
        return {row["unique_video_name"]: row for row in csv.DictReader(f)}


def inside_patient(video, config=None):
    """(first, last) in-patient frame ids of `video`, inclusive."""
    config = config or load_realcolon_config()
    first, last = config["inside_patient"][video]
    return first, last


def frame_path(video, index, config=None):
    config = config or load_realcolon_config()
    suffix = ".0" if video in config["float_frame_names"] else ""   # e.g. 001-012_1234.0.jpg
    return DATA_ROOT / config["root"] / config["frames_dir"] / f"{video}_frames" / f"{video}_{index}{suffix}.jpg"


def load_realcolon(name):
    config = load_realcolon_config()
    info = load_video_info(config)
    sequences = []
    for video in config["splits"][SPLITS[name]]:
        n_frames = int(info[video]["num_frames"])
        frames = [frame_path(video, i, config) for i in range(n_frames)]
        sequences.append(FrameFolderSequence(video, frames, float(info[video]["fps"]), annotations={}))
    return Dataset(name, sequences)
