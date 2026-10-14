"""Loader for the SUN-SEG test splits (configs/datasets/sunseg.yaml).

Each split is its own dataset, named sunseg_<difficulty>_<seen|unseen>, e.g.
sunseg_easy_seen = TestEasyDataset/Seen. Every frame is annotated (dense).

Frame order: a SUN-SEG clip can be spliced from several segments of the
original video (`..._a<segment>_ayy_image<number>.jpg`), so frames are ordered
by (segment, image number), as the SUN-SEG evaluation toolbox does. Plain
alphabetical order would put a10 before a2.
"""

import re

import yaml

from polypbench.data.base import Dataset, FrameFolderSequence
from polypbench.paths import CONFIG_DIR, DATA_ROOT

SPLITS = {
    "sunseg_easy_seen": "TestEasyDataset/Seen",
    "sunseg_hard_seen": "TestHardDataset/Seen",
    "sunseg_easy_unseen": "TestEasyDataset/Unseen",
    "sunseg_hard_unseen": "TestHardDataset/Unseen",
}
FPS = 30.0   # SUN database videos are recorded at 30 fps
_NAME = re.compile(r"_a(\d+)_.*image(\d+)$")


def load_sunseg_config():
    with open(CONFIG_DIR / "datasets" / "sunseg.yaml") as f:
        return yaml.safe_load(f)


def frame_order(path):
    m = _NAME.search(path.stem)
    if m is None:
        raise ValueError(f"Unexpected SUN-SEG frame name: {path.name}")
    return int(m.group(1)), int(m.group(2))


def load_sunseg(name):
    return load_sunseg_split(SPLITS[name], name)


def load_sunseg_split(split, name, clips=None, exclude=()):
    """Any SUN-SEG split (e.g. TrainDataset) as a Dataset, optionally restricted to/excluding clips."""
    cfg = load_sunseg_config()
    split_root = DATA_ROOT / cfg["root"] / split
    frame_root, mask_root = split_root / cfg["frames_dir"], split_root / cfg["masks_dir"]
    sequences = []
    for clip_dir in sorted(frame_root.iterdir()):
        if (clips is not None and clip_dir.name not in clips) or clip_dir.name in exclude:
            continue
        frames = sorted(clip_dir.glob("*.jpg"), key=frame_order)
        annotations = {}
        for idx, frame in enumerate(frames):
            mask = mask_root / clip_dir.name / f"{frame.stem}.png"
            if not mask.exists():
                raise FileNotFoundError(f"No mask for {frame}")
            annotations[idx] = mask
        sequences.append(FrameFolderSequence(clip_dir.name, frames, FPS, annotations))
    return Dataset(name, sequences)


def load_sunseg_train_val():
    """TrainDataset as (train, val) Datasets: val = the held-out val_clips, train = the rest."""
    cfg = load_sunseg_config()
    val_clips = set(cfg["val_clips"])
    return (load_sunseg_split(cfg["train_split"], "sunseg_train", exclude=val_clips),
            load_sunseg_split(cfg["train_split"], "sunseg_val", clips=val_clips))
