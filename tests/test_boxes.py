"""Map -> box extraction (postproc/boxes.py) on hand-made maps."""

import numpy as np

from polypbench.postproc.boxes import extract_boxes, scale_box


def test_scale_box():
    assert scale_box(0, 0, 0, 0, (2, 2), (4, 4)) == (0, 0, 1, 1)
    assert scale_box(1, 1, 1, 1, (2, 2), (4, 4)) == (2, 2, 3, 3)
    assert scale_box(0, 0, 351, 351, (352, 352), (1240, 1080)) == (0, 0, 1239, 1079)   # whole map -> whole frame
    x0, y0, x1, y1 = scale_box(100, 50, 199, 149, (352, 352), (1240, 1080))
    assert (x0, x1) == (352, 704) and (y0, y1) == (153, 460)                            # floor / ceil of the cell edges


def test_extract_boxes():
    prob = np.zeros((20, 20), np.uint8)
    prob[2:6, 3:9] = 128            # 128 / 255 >= 0.5: in
    prob[3, 4] = 230                # peak of that component
    prob[10:15, 10:12] = 127        # 127 / 255 < 0.5: out
    prob[18, 18] = 255              # 1-pixel speck, dropped by min_area_px=2
    boxes = extract_boxes(prob, (40, 40), threshold=0.5, min_area_px=2)
    assert len(boxes) == 1
    xmin, ymin, xmax, ymax, area, peak, mean = boxes[0]
    assert (xmin, ymin, xmax, ymax) == (6, 4, 17, 11)
    assert area == 24 / 400
    assert peak == 230 / 255
    assert np.isclose(mean, (23 * 128 + 230) / 24 / 255)
    assert extract_boxes(np.zeros((8, 8), np.uint8), (8, 8), 0.5) == []
