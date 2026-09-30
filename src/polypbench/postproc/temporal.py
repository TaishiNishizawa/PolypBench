"""Temporal smoothing of per-frame probability maps.

Ported from LambdaDiceEval/infer_dice_eval.py (temporal_smooth_prediction,
apply_area_fallback). Each frame's probability map is blended with the
previous frame's, warped by optical flow; the blend weight adapts to how much
the frame changed, and large changes are treated as scene cuts. If the
predicted area collapses relative to the previous frame, the previous map is
mostly kept (area fallback).

The old script's hysteresis thresholding is not ported: its result was only
fed back into itself and never reached the final mask.
"""

import cv2
import numpy as np


def _flow(src_gray, dst_gray):
    return cv2.calcOpticalFlowFarneback(
        src_gray, dst_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def _warp(img, flow):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1],
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class TemporalSmoother:
    """Stateful: call reset() at the start of every sequence, then once per frame in order."""

    def __init__(self, alpha_high, alpha_med, alpha_low, diff_low, diff_med, diff_reset,
                 area_fallback_ratio, area_threshold):
        self.alpha_high, self.alpha_med, self.alpha_low = alpha_high, alpha_med, alpha_low
        self.diff_low, self.diff_med, self.diff_reset = diff_low, diff_med, diff_reset
        self.area_fallback_ratio = area_fallback_ratio
        self.area_threshold = area_threshold
        self.reset()

    def reset(self):
        self.prev_prob = None
        self.prev_gray = None

    def _alpha(self, diff):
        if diff < self.diff_low:
            return self.alpha_high
        if diff < self.diff_med:
            return self.alpha_med
        return self.alpha_low

    def __call__(self, prob, rgb):
        """prob: HxW float32 at model resolution; rgb: the full-size frame it came from."""
        h, w = prob.shape
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA)

        out = prob
        if self.prev_prob is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - self.prev_gray.astype(np.float32)))) / 255.0
            if diff <= self.diff_reset:                     # otherwise: scene cut, keep raw prob
                alpha = self._alpha(diff)
                warped_prev = _warp(self.prev_prob.astype(np.float32), _flow(self.prev_gray, gray))
                out = alpha * warped_prev + (1.0 - alpha) * prob

            # Area fallback: if the mask nearly vanished, lean on the previous map.
            curr_area = float((out >= self.area_threshold).sum())
            prev_area = float((self.prev_prob >= self.area_threshold).sum())
            if prev_area > 0 and curr_area < self.area_fallback_ratio * prev_area:
                out = 0.8 * self.prev_prob + 0.2 * out

        self.prev_prob = out
        self.prev_gray = gray
        return out
