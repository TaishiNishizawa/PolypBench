"""Checks for the REAL-Colon loader and configs/datasets/realcolon.yaml.

The frame-folder checks list ~1M files on the NAS and take a minute or two.
They cover val and test only; train is checked once its download finishes
(realcolon_subset.py verify --splits train).
"""

import os

import cv2
import pytest

from polypbench.data import load_dataset
from polypbench.data.realcolon import frame_path, inside_patient, load_realcolon_config, load_video_info

EVAL_SPLITS = ["realcolon_val", "realcolon_test"]


@pytest.fixture(scope="module")
def config():
    return load_realcolon_config()


@pytest.fixture(scope="module")
def info(config):
    return load_video_info(config)


@pytest.fixture(scope="module", params=EVAL_SPLITS)
def dataset(request):
    return load_dataset(request.param)


def test_splits_follow_official_rule(config, info):
    """Per cohort: videos 1-10 train, 11-12 val, 13-15 test; all 60 videos, no overlap."""
    numbers = {"train": range(1, 11), "val": range(11, 13), "test": range(13, 16)}
    seen = []
    for split, videos in config["splits"].items():
        assert sorted(videos) == sorted(f"{c:03d}-{v:03d}" for c in range(1, 5) for v in numbers[split]), split
        seen += videos
    assert len(seen) == len(set(seen)) == 60
    assert set(seen) == set(info)


def test_inside_patient_ranges(config, info):
    assert set(config["inside_patient"]) == set(info)
    for video, row in info.items():
        first, last = inside_patient(video, config)
        assert 0 <= first <= last < int(row["num_frames"]), video


def test_inside_patient_matches_true_colon(config):
    """TRUE-Colon's in-patient test set has 572,085 frames."""
    total = 0
    for video in config["splits"]["test"]:
        first, last = inside_patient(video, config)
        total += last - first + 1
    assert total == 572_085


def test_frame_totals(dataset):
    """All-frame totals of the official splits (REAL-Colon paper)."""
    expected = {"realcolon_val": 371_169, "realcolon_test": 591_647}
    assert sum(seq.n_frames for seq in dataset) == expected[dataset.name]


def test_fps_per_video():
    dataset = load_dataset("realcolon_val")
    assert dataset["001-012"].fps == 60.0          # 001-001 .. 001-012 were recorded at 60 fps
    assert dataset["003-011"].fps == 25.0
    assert dataset["004-011"].fps == 29.97


def test_no_mask_annotations(dataset):
    assert all(seq.annotations == {} for seq in dataset)


def test_frame_folders_match(dataset):
    """Each video folder holds exactly the frames the loader builds paths for."""
    for seq in dataset:
        folder = seq.frame_paths[0].parent
        assert set(os.listdir(folder)) == {p.name for p in seq.frame_paths}, seq.name


def test_first_and_last_frame_readable(dataset):
    for seq in dataset:
        idx, rgb = next(iter(seq.frames()))
        assert idx == 0 and rgb.ndim == 3 and rgb.shape[2] == 3, seq.name
        assert seq.frame_paths[-1] == frame_path(seq.name, seq.n_frames - 1)
        assert cv2.imread(str(seq.frame_paths[-1])) is not None, seq.name
