"""Inter-frame stability: how much a model's mask changes between consecutive frames.

Ported from LambdaDiceEval/log_interframe_iou.py. For each transition k-1 -> k:

    iou_raw        IoU(mask_k-1, mask_k) as-is; includes genuine camera/polyp motion.
    iou_warped     IoU(mask_k-1 warped into frame k by optical flow, mask_k), over pixels
                   whose flow source lies inside the frame. Motion is compensated, so what
                   remains is mostly the model disagreeing with itself. Primary metric.
    frame_diff     mean |gray_k - gray_k-1| / 255 at 352x352 (same definition as the
                   temporal smoother's scene-change measure).
    flow_mag       mean optical-flow magnitude in pixels (full res): how far the scene moved.
    warp_residual  mean |gray_k - warp(gray_k-1)| / 255 after flow compensation: frame change
                   NOT explained by motion (noise, lighting, occlusion, codec, flow failure).
    area_prev, area_curr   mask areas in pixels, to recover appear/vanish events.

IoU is NaN when both masks are empty (nothing to be consistent about); 0 if only one is.
"""

import cv2
import numpy as np

TRANSITION_COLUMNS = ["area_prev", "area_curr", "iou_raw", "iou_warped",
                      "frame_diff", "flow_mag", "warp_residual"]

DIFF_SIZE = 352     # resolution for frame_diff, matching the temporal smoother
VALID_MIN = 0.999   # a pixel is valid for the warped comparison only if its flow source is in-frame


def _flow(src_gray, dst_gray):
    return cv2.calcOpticalFlowFarneback(
        src_gray, dst_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


class FrameMotion:
    """Motion between two consecutive frames; computed once and shared by every model.

    Backward warping needs the flow from CURRENT to PREVIOUS: out(x) = prev(x + flow_cur_to_prev(x)).
    (Sampling prev with the prev->cur flow shifts things the wrong way.)
    """

    def __init__(self, prev_rgb, curr_rgb):
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        h, w = curr_gray.shape

        flow = _flow(curr_gray, prev_gray)   # cur -> prev
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.map_x = gx + flow[..., 0]
        self.map_y = gy + flow[..., 1]
        self.valid = self._warp(np.ones((h, w), np.float32)) >= VALID_MIN

        small = (DIFF_SIZE, DIFF_SIZE)
        prev_small = cv2.resize(prev_gray, small, interpolation=cv2.INTER_AREA).astype(np.float32)
        curr_small = cv2.resize(curr_gray, small, interpolation=cv2.INTER_AREA).astype(np.float32)
        self.frame_diff = float(np.mean(np.abs(curr_small - prev_small)) / 255.0)

        if self.valid.any():
            self.flow_mag = float(np.linalg.norm(flow, axis=2)[self.valid].mean())
            residual = np.abs(curr_gray.astype(np.float32) - self._warp(prev_gray.astype(np.float32)))
            self.warp_residual = float(residual[self.valid].mean() / 255.0)
        else:
            self.flow_mag = self.warp_residual = float("nan")

    def _warp(self, img):
        return cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def warp_mask(self, mask):
        return self._warp(mask.astype(np.float32)) >= 0.5


def mask_iou(a, b):
    """IoU of two boolean masks; NaN if both are empty, 0 if only one is."""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def transition_metrics(motion, prev_mask, curr_mask):
    """All per-transition quantities for prev -> curr (boolean masks at full resolution)."""
    valid = motion.valid
    warped = motion.warp_mask(prev_mask)
    return {
        "area_prev": int(prev_mask.sum()),
        "area_curr": int(curr_mask.sum()),
        "iou_raw": mask_iou(prev_mask, curr_mask),
        "iou_warped": mask_iou(warped & valid, curr_mask & valid),
        "frame_diff": motion.frame_diff,
        "flow_mag": motion.flow_mag,
        "warp_residual": motion.warp_residual,
    }


def summarize_transitions(df):
    """Summary of one group of transitions (a DataFrame with TRANSITION_COLUMNS)."""
    both_empty = (df.area_prev == 0) & (df.area_curr == 0)
    toggles = (df.area_prev == 0) != (df.area_curr == 0)
    active = df[~both_empty]
    out = {
        "n_transitions": len(df),
        "n_both_empty": int(both_empty.sum()),
        "n_appear_vanish": int(toggles.sum()),          # mask on in exactly one of the two frames
    }
    for col in ("iou_warped", "iou_raw"):
        v = active[col].dropna()
        out[f"{col}_mean"] = v.mean() if len(v) else np.nan
        out[f"{col}_median"] = v.median() if len(v) else np.nan
        out[f"{col}_p5"] = v.quantile(0.05) if len(v) else np.nan   # worst 5% of transitions
        out[f"{col}_min"] = v.min() if len(v) else np.nan
    both = active.dropna(subset=["iou_raw"])
    if len(both) > 2 and both.frame_diff.std() > 0 and both.iou_raw.std() > 0:
        # high = raw IoU drops are mostly explained by the scene changing
        out["corr_iou_drop_frame_diff"] = np.corrcoef(1.0 - both.iou_raw, both.frame_diff)[0, 1]
    else:
        out["corr_iou_drop_frame_diff"] = np.nan
    return out
