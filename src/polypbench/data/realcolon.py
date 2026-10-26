"""Loader for REAL-Colon (configs/datasets/realcolon.yaml).

Each official split is its own dataset: realcolon_train, realcolon_val,
realcolon_test. One sequence per video, named by its id (e.g. 001-013),
covering every frame from 0, including frames outside the patient; in-patient
trimming is applied at evaluation (see inside_patient).

Frames come from the cache (a smaller copy built by
scripts/build_realcolon_cache.py) or the originals on the NAS: `frames_from`
in the config, or the `source` argument. Loading from the cache fails if a
video isn't fully cached, so a run never mixes the two.

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
import json
import tarfile
from pathlib import PurePosixPath

import xml.etree.ElementTree as ET   # official, MD5-checked archives; expat >= 2.4 blocks entity expansion
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


def load_boxes(video, config=None):
    """{frame: [(unique_id, xmin, ymin, xmax, ymax), ...]} in original pixels, frames with >= 1 box only.

    Read straight from the annotation archive. Coordinates are read by tag name:
    REAL-Colon writes them as xmin, xmax, ymin, ymax, so a positional parse
    scrambles every box. A lesion's boxed frames can have gaps (it leaves the view).
    """
    config = config or load_realcolon_config()
    path = DATA_ROOT / config["root"] / config["annotations_dir"] / f"{video}_annotations.tar.gz"
    boxes = {}
    with tarfile.open(path) as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".xml")):
                continue
            objects = ET.fromstring(tar.extractfile(member).read()).findall("object")
            if not objects:
                continue
            index = int(float(PurePosixPath(member.name).stem.rsplit("_", 1)[1]))   # "004-012_9307" or "_9307.0"
            boxes[index] = [(obj.findtext("unique_id"),
                             *(int(float(obj.find("bndbox").findtext(tag))) for tag in ("xmin", "ymin", "xmax", "ymax")))
                            for obj in objects]
    return boxes


def frame_path(video, index, config=None):
    config = config or load_realcolon_config()
    suffix = ".0" if video in config["float_frame_names"] else ""   # e.g. 001-012_1234.0.jpg
    return DATA_ROOT / config["root"] / config["frames_dir"] / f"{video}_frames" / f"{video}_{index}{suffix}.jpg"


def cache_dir(video, config=None):
    config = config or load_realcolon_config()
    cache = config["cache"]
    return DATA_ROOT / cache["root"] / f"L{cache['long_side']}" / video


def cached_frame_path(video, index, config=None):
    return cache_dir(video, config) / f"{index:06d}.jpg"


def cache_info(video, config=None):
    """The cache's done.json for `video` (n_frames, original_wh, cached_wh, ...), or None if not cached."""
    path = cache_dir(video, config) / "done.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_realcolon(name, source=None):
    """source: "cache" or "original" frames (default: frames_from in the config)."""
    config = load_realcolon_config()
    source = source or config["frames_from"]
    if source not in ("cache", "original"):
        raise ValueError(f"source must be 'cache' or 'original', got '{source}'")
    info = load_video_info(config)
    sequences = []
    for video in config["splits"][SPLITS[name]]:
        n_frames = int(info[video]["num_frames"])
        if source == "cache":
            cached = cache_info(video, config)
            if cached is None or cached["n_frames"] != n_frames:
                raise FileNotFoundError(f"{video} isn't (fully) cached at {cache_dir(video, config)}: "
                                        f"run scripts/build_realcolon_cache.py")
            frames = [cached_frame_path(video, i, config) for i in range(n_frames)]
        else:
            frames = [frame_path(video, i, config) for i in range(n_frames)]
        sequences.append(FrameFolderSequence(video, frames, float(info[video]["fps"]), annotations={}))
    return Dataset(name, sequences)
