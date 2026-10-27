"""Checks for the REAL-Colon ground-truth tables (scripts/build_realcolon_gt.py) against published counts."""

import pytest

from polypbench.data.realcolon import cache_info, load_boxes, load_realcolon_config
from polypbench.data.realcolon_gt import latency_excluded, load_ground_truth


@pytest.fixture(scope="module")
def gt():
    try:
        return load_ground_truth()
    except FileNotFoundError:
        pytest.skip("ground truth not built yet (scripts/build_realcolon_gt.py)")


def test_latency_rule():
    """TRUE-Colon: excluded if a gap of > 50 missing frames starts within the first 250 frames."""
    assert not latency_excluded(list(range(100)))                # no gap
    assert latency_excluded([0, 1, 53])                          # frames 2-52 missing: 51 > 50
    assert not latency_excluded([0, 1, 52])                      # frames 2-51 missing: exactly 50
    assert latency_excluded(list(range(250)) + [400])            # gap starts at frame 249, inside the first 250
    assert not latency_excluded(list(range(251)) + [400])        # gap starts at frame 250, after the window


def test_published_totals(gt):
    """REAL-Colon paper: 2,757,723 frames, 351,264 boxes, 132 lesions; TRUE-Colon Table 1 boxed frames."""
    v = gt.videos.groupby("split")
    assert v["n_frames"].sum().to_dict() == {"train": 1_794_907, "val": 371_169, "test": 591_647}
    assert v["n_boxed_frames"].sum().to_dict() == {"train": 230_936, "val": 29_067, "test": 82_106}
    assert v["n_lesions"].sum().to_dict() == {"train": 101, "val": 10, "test": 21}
    assert len(gt.boxes) == 351_264
    assert len(gt.lesions) == 132 and gt.lesions["boxed"].all() and gt.lesions["in_lesion_info"].all()
    assert (~gt.lesions["is_polyp"]).sum() == 19          # histologically not polyps


def test_polyp_free_videos(gt):
    free = gt.videos[gt.videos["n_lesions"] == 0].groupby("split")["video"].apply(sorted).to_dict()
    assert free == {"train": ["001-002", "001-008", "002-001", "004-002", "004-007", "004-010"],
                    "val": ["001-011", "001-012", "002-012", "003-011"],
                    "test": ["001-013", "001-015", "002-013", "002-015"]}


def test_video_info_lesion_count_mismatch(gt):
    """Only known mismatch: 003-012 has 3 annotated lesions, video_info.csv says 2."""
    v = gt.videos
    assert v.loc[v["n_lesions"] != v["n_lesions_video_info"], "video"].tolist() == ["003-012"]


def test_boxes_inside_frames(gt):
    b = gt.boxes.merge(gt.videos[["video", "width", "height"]], on="video")
    assert (b["xmin"] >= 0).all() and (b["ymin"] >= 0).all()
    assert (b["xmax"] < b["width"]).all() and (b["ymax"] < b["height"]).all()
    assert (b["xmin"] < b["xmax"]).all() and (b["ymin"] < b["ymax"]).all()
    assert set(b["unique_id"]) == set(gt.lesions["unique_id"])
    assert (b["frame"] < b.merge(gt.videos, on="video")["n_frames"]).all()


def test_frame_size_matches_cached_frames(gt):
    """The XML <size> agrees with the actual frames (recorded by the frame cache for val + test)."""
    config = load_realcolon_config()
    for row in gt.videos[gt.videos["split"] != "train"].itertuples():
        cached = cache_info(row.video, config)
        if cached is not None:
            assert cached["original_wh"] == [row.width, row.height], row.video


def test_matches_raw_reader(gt):
    """The table holds the raw boxes of 004-012 (none needed clipping there)."""
    raw = load_boxes("004-012")
    rows = gt.boxes[gt.boxes["video"] == "004-012"]
    assert len(rows) == sum(len(v) for v in raw.values())
    r = rows[rows["frame"] == 10000].iloc[0]
    assert (r["unique_id"], r["xmin"], r["ymin"], r["xmax"], r["ymax"]) == ("004-012_1", 359, 193, 857, 773)


def test_lesion_spans(gt):
    les = gt.lesions.set_index("unique_id")
    assert les.loc["004-012_1", ["first_frame", "last_frame", "n_boxed_frames"]].tolist() == [9307, 13494, 3592]
    assert (les["first_frame"] <= les["last_frame"]).all()
    assert (les["n_boxed_frames"] <= les["last_frame"] - les["first_frame"] + 1).all()
