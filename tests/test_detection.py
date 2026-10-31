"""Detection metrics (metrics/detection.py) on a hand-made video."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from polypbench.metrics import detection as d


@pytest.fixture
def gt():
    # 100 frames at 10 fps, inside the patient 5..94; one lesion boxed on 20..29 and 35..39.
    frames = list(range(20, 30)) + list(range(35, 40))
    boxes = pd.DataFrame({"video": "v", "frame": frames, "unique_id": "v_1",
                          "xmin": 10, "ymin": 10, "xmax": 19, "ymax": 19})
    lesions = pd.DataFrame([{"unique_id": "v_1", "video": "v", "first_frame": 20, "last_frame": 39,
                             "n_boxed_frames": 15, "latency_excluded": False, "size_mm": 4.0,
                             "site": "cecum", "histology_class": "AD", "is_polyp": True}])
    videos = pd.DataFrame([{"video": "v", "fps": 10.0, "n_frames": 100, "inside_first": 5, "inside_last": 94}])
    return SimpleNamespace(boxes=boxes, lesions=lesions, videos=videos)


@pytest.fixture
def pred():
    rows = [(22, 15, 15, 30, 30, 0.9),     # overlaps the lesion box: a hit
            (25, 25, 25, 30, 30, 0.95),    # positive frame, but misses the box
            (50, 0, 0, 5, 5, 0.7),         # after removal
            (70, 0, 0, 5, 5, 0.6),         # clean
            (2, 0, 0, 5, 5, 0.8),          # outside the patient
            (80, 0, 0, 5, 5, 0.99)]        # clean, but too small
    p = pd.DataFrame(rows, columns=["frame", "xmin", "ymin", "xmax", "ymax", "peak"]).assign(video="v")
    return p.assign(mean_prob=p["peak"] - 0.1, mean_logit=p["peak"] * 10, area=[0.01] * 5 + [0.0001])


def test_phases(gt):
    f = d.frame_table(gt, after_removal_s=2).set_index("frame")["phase"]
    assert (f[0:4] == "outside").all() and (f[95:99] == "outside").all()
    assert (f[20:29] == "positive").all() and (f[35:39] == "positive").all()
    assert (f[30:34] == "inspection").all()
    assert (f[40:59] == "after_removal").all() and f[60] == "clean" and f[19] == "clean"


def test_frame_rates(gt, pred):
    frames, lesion = d.score_boxes(pred, gt.boxes, "peak", min_area=0.001)
    s = d.scored_frames(d.frame_table(gt, after_removal_s=2), frames)
    lf = d.scored_lesion_frames(gt.boxes, lesion)
    r = d.frame_rates(s, [0.65], lf).iloc[0]
    assert (r["n_pos"], r["n_neg"]) == (15, 85)
    assert r["tpr"] == 1 / 15 and r["box_sensitivity"] == 1 / 15
    assert r["fpr"] == 2 / 85                                   # frames 50 and 2; frame 80 dropped by min_area
    assert d.threshold_for_fpr(s, 1 / 85) == 0.8
    assert d.threshold_for_fpr(s, 0.5) == 0.6
    assert d.threshold_for_fpr(s, 0) > 0.8
    inside = d.frame_rates(s[s["in_patient"]], [0.65]).iloc[0]
    assert inside["fpr"] == 1 / 75                              # only frame 50 among in-patient negatives
    phase = d.alarm_rates_by_phase(s, 0.65).set_index("phase")
    assert phase.loc["outside", "alarm_rate"] == 1 / 10 and phase.loc["after_removal", "alarm_rate"] == 1 / 20
    assert phase.loc["clean", "alarm_rate"] == 0


def test_mean_score(gt, pred):
    frames, _ = d.score_boxes(pred, gt.boxes, "mean_prob")
    assert np.isclose(frames.set_index("frame").loc[22, "hit"], 0.8)
    frames, _ = d.score_boxes(pred, gt.boxes, "mean_logit")
    assert np.isclose(frames.set_index("frame").loc[22, "hit"], 9.0)


def test_negative_scores(gt, pred):
    """Logit scores can be <= 0; frames without a box must still never count."""
    frames, lesion = d.score_boxes(pred.assign(peak=pred["peak"] - 5), gt.boxes, "peak")
    s = d.scored_frames(d.frame_table(gt, after_removal_s=2), frames)
    r = d.frame_rates(s, [-4.5], d.scored_lesion_frames(gt.boxes, lesion)).iloc[0]
    assert r["tpr"] == 1 / 15 and r["fpr"] == 4 / 85   # frames 2, 50, 70, 80; nothing else
    assert d.threshold_for_fpr(s, 1 / 85) == pytest.approx(0.99 - 5)


def test_lesion_metrics(gt, pred):
    _, lesion = d.score_boxes(pred, gt.boxes, "peak")
    lf = d.scored_lesion_frames(gt.boxes, lesion)
    m = d.lesion_metrics(lf, gt.lesions, gt.videos, tau=0.5).iloc[0]
    assert m["detected"] and not m["persistent"] and m["n_hit_frames"] == 1
    assert m["latency_frames"] == 2 and m["latency_s"] == pytest.approx(0.2) and m["within_1s"]
    missed = d.lesion_metrics(lf, gt.lesions, gt.videos, tau=0.95).iloc[0]
    assert not missed["detected"] and np.isnan(missed["latency_s"]) and not missed["within_5s"]
