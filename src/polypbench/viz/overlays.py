"""Drawing helpers. All images are RGB uint8; convert to BGR only when writing with OpenCV."""

import cv2
import numpy as np

PRED_COLOR = (0, 255, 0)       # green: model prediction
GT_COLOR = (255, 40, 40)       # red: ground truth
TEXT_COLOR = (255, 255, 255)


def overlay(rgb, mask, color, fill_alpha=0.35, thickness=2):
    """Tint `mask` pixels with `color` and outline them."""
    out = rgb.copy()
    mask = mask.astype(np.uint8)
    if fill_alpha > 0 and mask.any():
        tinted = out.copy()
        tinted[mask > 0] = color
        out = cv2.addWeighted(out, 1.0 - fill_alpha, tinted, fill_alpha, 0.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def label(rgb, lines, scale=0.55, origin=(8, 22), line_height=22):
    """Write lines of text top-left, with a dark outline so it reads on any background."""
    out = rgb.copy()
    x, y = origin
    for i, text in enumerate(lines):
        pos = (x, y + i * line_height)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, 1, cv2.LINE_AA)
    return out


def draw_box(rgb, box, thickness=2):
    """Outline box (x0, y0, x1, y1) in white over a dark edge, so it reads on any tissue colour."""
    out = rgb.copy()
    p0, p1 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(out, p0, p1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.rectangle(out, p0, p1, TEXT_COLOR, thickness, cv2.LINE_AA)
    return out


def header_cell(text, width, height=36):
    cell = np.full((height, width, 3), 30, dtype=np.uint8)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    x = max(4, (width - size[0]) // 2)
    cv2.putText(cell, text, (x, (height + size[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
    return cell


class VideoWriter:
    """Writes RGB frames to an .mp4."""

    def __init__(self, path, fps, width, height):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")

    def write(self, rgb):
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()
