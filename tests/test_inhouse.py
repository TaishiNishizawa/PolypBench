"""Checks for the in-house dataset loader and its generated config."""

import cv2
import numpy as np
import pytest
import yaml

from polypbench.data.inhouse import CONFIG_PATH, load_inhouse
from polypbench.paths import DATA_ROOT

MAX_DIFF = 0.01   # same threshold as build_inhouse_manifest.py


@pytest.fixture(scope="module")
def dataset():
    return load_inhouse()


@pytest.fixture(scope="module")
def images_dir():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return DATA_ROOT / config["root"] / config["images_dir"]


def test_counts(dataset):
    assert len(dataset) == 4
    assert sum(len(seq.annotations) for seq in dataset) == 123


def test_no_duplicate_annotation_files(dataset):
    names = [path.name for seq in dataset for path in seq.annotations.values()]
    assert len(names) == len(set(names))


def test_annotation_indices_in_range(dataset):
    for seq in dataset:
        for idx in seq.annotations:
            assert 0 <= idx < seq.n_frames, f"{seq.name}: annotation index {idx} out of range"


def test_masks_binary_and_sized(dataset):
    for seq in dataset:
        for idx in seq.annotations:
            mask = seq.load_mask(idx)
            where = f"{seq.name} frame {idx}"
            assert mask.shape == (360, 480), where
            assert mask.dtype == np.uint8, where
            assert set(np.unique(mask)) <= {0, 1}, where


def test_empty_masks(dataset):
    """Empty masks (polyp out of view or removed) are valid, but pinned so changes are noticed."""
    empty = {seq.name: sorted(idx for idx in seq.annotations if not seq.load_mask(idx).any())
             for seq in dataset}
    assert empty == {
        "first_polyp_seen": [],
        "first_polyp_removed": [330, 345],
        "second_polyp_seen": [],
        "second_polyp_removed": [255, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450],
    }


def test_missing_annotation_raises(dataset):
    seq = dataset["first_polyp_seen"]
    with pytest.raises(KeyError):
        seq.load_mask(7)          # 7 isn't an annotated frame


def test_frames_align_with_annotations(dataset, images_dir):
    """The frame at each annotated index is the annotated image."""
    for seq in dataset:
        n_read = 0
        for idx, frame in seq.frames():
            n_read += 1
            if idx not in seq.annotations:
                continue
            ann = cv2.imread(str(images_dir / seq.annotations[idx].name))
            ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)     # frames() yields RGB
            diff = np.mean(np.abs(frame.astype(np.float32) - ann.astype(np.float32))) / 255.0
            assert diff < MAX_DIFF, f"{seq.name} frame {idx}: diff {diff:.4f}"
        assert n_read == seq.n_frames, f"{seq.name}: read {n_read} frames, expected {seq.n_frames}"
