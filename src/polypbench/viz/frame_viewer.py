"""Data and rendering behind scripts/view_predictions.py: one REAL-Colon video, its npz
predictions from several models, and its ground-truth boxes.

Prediction maps are at model resolution (e.g. 352x352, the whole frame stretched), so they
are stretched back to the frame's shape for display; boxes are in original pixels.
"""

import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from polypbench.data import load_dataset
from polypbench.data.realcolon import inside_patient, load_boxes
from polypbench.io import predictions
from polypbench.viz.overlays import PRED_COLOR, draw_box, overlay

PAGE = Path(__file__).with_name("frame_viewer.html")
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots, fixed order
TIMELINE_THRESHOLD = 128                                      # uint8 for p >= 0.5
CACHED_CHUNKS = 6                                             # ~250 MB each when decompressed
KEEP_ALL_GB = 16      # keep every chunk in memory (instant jumps) if they fit; a cold chunk takes ~10 s


class FrameViewer:
    def __init__(self, dataset, video, model_ids, postproc="none", display_width=760):
        if predictions.storage_format(dataset) != "npz":
            raise ValueError(f"{dataset}: the viewer reads npz predictions (REAL-Colon only)")
        self.dataset, self.video, self.postproc = dataset, video, postproc
        self.display_width = display_width
        seq = load_dataset(dataset)[video]
        self.frame_paths, self.n_frames, self.fps = seq.frame_paths, seq.n_frames, seq.fps
        self.inside = inside_patient(video)
        self.boxes = load_boxes(video)
        self.models = []
        for model_id in model_ids:
            if not predictions.sequence_complete(dataset, model_id, postproc, video):
                raise FileNotFoundError(f"No complete predictions for {model_id} on {dataset}/{video}")
            info = predictions.sequence_info(dataset, model_id, postproc, video)
            if info["n_frames"] != self.n_frames:
                raise ValueError(f"{model_id}: {info['n_frames']} predicted frames, video has {self.n_frames}")
            self.models.append(model_id)
            map_bytes = info["map_hw"][0] * info["map_hw"][1]
        self.memory_gb = self.n_frames * map_bytes * len(self.models) / 1e9
        self.keep_all = self.memory_gb <= KEEP_ALL_GB
        self._chunks, self._lock = OrderedDict(), threading.Lock()
        self.timeline = {m: self._area_per_frame(m) for m in self.models}

    def _chunk(self, model_id, k):
        key = (model_id, k)
        with self._lock:
            if key in self._chunks:
                self._chunks.move_to_end(key)
                return self._chunks[key]
        chunk = predictions.read_chunk(self.dataset, model_id, self.postproc, self.video, k)
        with self._lock:
            self._chunks[key] = chunk
            while not self.keep_all and len(self._chunks) > CACHED_CHUNKS:
                self._chunks.popitem(last=False)
        return chunk

    def _area_per_frame(self, model_id):
        """Fraction of each frame predicted >= 0.5, in per mille (for the timeline)."""
        n_chunks = predictions.sequence_info(self.dataset, model_id, self.postproc, self.video)["n_chunks"]
        area = np.zeros(self.n_frames, dtype=np.int16)
        for k in range(n_chunks):
            frames, probs = self._chunk(model_id, k) if self.keep_all else \
                predictions.read_chunk(self.dataset, model_id, self.postproc, self.video, k)
            area[frames] = np.rint((probs >= TIMELINE_THRESHOLD).mean(axis=(1, 2)) * 1000)
        return area

    def prob(self, model_id, index):
        """float32 probability map at model resolution for one frame."""
        frames, probs = self._chunk(model_id, index // predictions.CHUNK_SIZE)
        pos = index - int(frames[0])
        if not (0 <= pos < len(frames) and frames[pos] == index):   # writer keeps CHUNK_SIZE frames per chunk, in order
            raise KeyError(f"frame {index} not where expected in {model_id}'s chunks")
        return probs[pos].astype(np.float32) / 255.0

    def meta(self):
        boxed = np.zeros(self.n_frames, dtype=bool)
        boxed[list(self.boxes)] = True
        edges = np.flatnonzero(np.diff(np.r_[0, boxed.astype(np.int8), 0]))   # boxed runs as [start, end)
        return {
            "dataset": self.dataset, "video": self.video, "n_frames": self.n_frames, "fps": self.fps,
            "inside": list(self.inside), "boxed_runs": edges.reshape(-1, 2).tolist(),
            "models": [{"id": m, "color": SERIES_COLORS[i % len(SERIES_COLORS)]} for i, m in enumerate(self.models)],
            "timeline": {m: area.tolist() for m, area in self.timeline.items()},
        }

    def stats(self, index, threshold):
        out = {"boxes": [list(b) for b in self.boxes.get(index, [])], "models": {}}
        for m in self.models:
            p = self.prob(m, index)
            out["models"][m] = {"peak": round(float(p.max()), 3), "area": round(float((p >= threshold).mean()), 4)}
        return out

    def render(self, model_id, index, threshold, show_gt=True):
        """JPEG bytes: the frame at display width, >= threshold region tinted and outlined, GT boxes in white."""
        bgr = cv2.imread(str(self.frame_paths[index]))
        if bgr is None:
            raise FileNotFoundError(self.frame_paths[index])
        h, w = bgr.shape[:2]
        scale = min(1.0, self.display_width / w)
        size = (round(w * scale), round(h * scale))
        rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(self.prob(model_id, index), size, interpolation=cv2.INTER_LINEAR) >= threshold
        out = overlay(rgb, mask, PRED_COLOR, fill_alpha=0.25, thickness=2)
        if show_gt:
            for _, *box in self.boxes.get(index, []):
                out = draw_box(out, [v * scale for v in box])
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return jpg.tobytes()
