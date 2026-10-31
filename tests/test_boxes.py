"""Map -> box extraction (postproc/boxes.py) on hand-made maps."""

import numpy as np

from polypbench.postproc.boxes import extract_boxes, scale_box


def test_scale_box():
    assert scale_box(0, 0, 0, 0, (2, 2), (4, 4)) == (0, 0, 1, 1)
    assert scale_box(1, 1, 1, 1, (2, 2), (4, 4)) == (2, 2, 3, 3)
    assert scale_box(0, 0, 351, 351, (352, 352), (1240, 1080)) == (0, 0, 1239, 1079)   # whole map -> whole frame
    x0, y0, x1, y1 = scale_box(100, 50, 199, 149, (352, 352), (1240, 1080))
    assert (x0, x1) == (352, 704) and (y0, y1) == (153, 460)                            # floor / ceil of the cell edges


def logit(p):
    return np.log(p / (1 - p))


def test_extract_boxes():
    lg = np.full((20, 20), -5.0, np.float32)
    lg[2:6, 3:9] = 0.0              # sigmoid 0.5 >= 0.5: in
    lg[3, 4] = 30.0                 # peak of that component: a probability would be 1.0 here
    lg[10:15, 10:12] = -0.01        # just below 0.5: out
    lg[18, 18] = 40.0               # 1-pixel speck, dropped by min_area_px=2
    boxes = extract_boxes(lg.astype(np.float16), (40, 40), threshold=0.5, min_area_px=2)
    assert len(boxes) == 1
    xmin, ymin, xmax, ymax, area, peak, mean_logit, mean_prob = boxes[0]
    assert (xmin, ymin, xmax, ymax) == (6, 4, 17, 11)
    assert area == 24 / 400
    assert peak == 30.0
    assert np.isclose(mean_logit, 30 / 24)
    assert np.isclose(mean_prob, (23 * 0.5 + 1.0) / 24)
    assert extract_boxes(np.full((8, 8), -1.0), (8, 8), 0.5) == []


def test_threshold_is_a_probability():
    lg = np.array([[logit(0.69), logit(0.71)]], np.float32)
    assert len(extract_boxes(lg, (2, 1), threshold=0.7)) == 1
    assert len(extract_boxes(lg, (2, 1), threshold=0.5)) == 1
    assert extract_boxes(lg, (2, 1), threshold=0.72) == []
