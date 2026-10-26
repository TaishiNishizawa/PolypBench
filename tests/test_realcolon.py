"""Checks for the REAL-Colon loader and configs/datasets/realcolon.yaml.

The frame-folder checks list ~1M files on the NAS and take a minute or two.
They cover val and test only; train is checked once its download finishes
(realcolon_subset.py verify --splits train).
"""

import os

import cv2
import numpy as np
import pytest

from polypbench.data.realcolon import (cache_dir, cache_info, frame_path, inside_patient, load_boxes, load_realcolon,
                                       load_realcolon_config, load_video_info)

EVAL_SPLITS = ["realcolon_val", "realcolon_test"]


@pytest.fixture(scope="module")
def config():
    return load_realcolon_config()


@pytest.fixture(scope="module")
def info(config):
    return load_video_info(config)


@pytest.fixture(scope="module", params=EVAL_SPLITS)
def dataset(request):
    return load_realcolon(request.param, source="original")


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
    dataset = load_realcolon("realcolon_val", source="original")
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


def cached_videos(config, info):
    videos = [v for s in ("val", "test") for v in config["splits"][s]
              if (cache_info(v, config) or {}).get("n_frames") == int(info[v]["num_frames"])]
    if not videos:
        pytest.skip("no video cached yet (scripts/build_realcolon_cache.py)")
    return videos


def test_cache_complete_and_sized(config, info):
    """Every cached video has all its frames, long side = cache.long_side, original aspect ratio."""
    long_side = config["cache"]["long_side"]
    for video in cached_videos(config, info):
        cached = cache_info(video, config)
        files = [f for f in os.listdir(cache_dir(video, config)) if f.endswith(".jpg")]
        assert len(files) == cached["n_frames"], video
        (w0, h0), (w, h) = cached["original_wh"], cached["cached_wh"]
        assert max(w, h) == min(long_side, max(w0, h0)), video
        assert abs(w / h - w0 / h0) < 0.01, video
        first = cv2.imread(str(cache_dir(video, config) / "000000.jpg"))
        assert first.shape[:2] == (h, w), video


def test_cache_matches_originals(config, info):
    """A cached frame is its original, resized: PSNR >= 35 dB.

    JPEG loss grows with sensor grain: 46-49 dB on clean videos, ~38 dB on the
    grainiest (004-012). The wrong frame (even the next one) scores ~27 dB.
    """
    for video in cached_videos(config, info):
        n = int(info[video]["num_frames"])
        for idx in (0, n // 2, n - 1):
            cached = cv2.imread(str(cache_dir(video, config) / f"{idx:06d}.jpg")).astype("float32")
            original = cv2.imread(str(frame_path(video, idx, config)))
            original = cv2.resize(original, cached.shape[1::-1], interpolation=cv2.INTER_AREA).astype("float32")
            psnr = 10 * np.log10(255.0 ** 2 / np.mean((cached - original) ** 2))
            assert psnr >= 35.0, f"{video} frame {idx}: PSNR {psnr:.1f} dB"


def test_load_boxes_by_tag_name(config):
    """004-012: one lesion, boxed on 3,592 frames between 9,307 and 13,494, with gaps."""
    boxes = load_boxes("004-012", config)
    assert len(boxes) == 3592 and min(boxes) == 9307 and max(boxes) == 13494
    assert 11000 not in boxes                                     # lesion out of view
    assert boxes[10000] == [("004-012_1", 359, 193, 857, 773)]    # XML order is xmin, xmax, ymin, ymax
    for frame_boxes in boxes.values():
        for _, x0, y0, x1, y1 in frame_boxes:
            assert x0 < x1 and y0 < y1
