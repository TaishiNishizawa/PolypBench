"""The on-disk prediction format shared by every stage.

Every frame's prediction is saved, never min-max normalized, under

    <artifacts_root>/predictions/<dataset>/<model_id>/<postproc>/<sequence>/

plus a run.json in <dataset>/<model_id>/<postproc>/, written only once every
sequence is done, so its presence marks a complete run. The encoding depends
on the dataset (storage_format):

  png  (in-house, SUN-SEG): one <frame:06d>.png per frame at the frame's
       original resolution, the foreground probability as uint8 (0-255).
       Read with read_prediction(frame_path(...)).
  npz  (REAL-Colon, ~1M frames): chunk_<k:04d>.npz files of up to CHUNK_SIZE
       frames ("frames": int32 frame indices, "logits": float16 N x H x W) at
       the model's own output resolution, not resized to the frame. Logits are
       what the model's sigmoid is applied to; probabilities would tie at 1.0
       for confident regions (uint8 above p = 0.998, float32 above logit ~17),
       which hides how confident a false region is. done.json in the sequence
       folder marks the sequence complete, so an interrupted run resumes per
       sequence. Read with read_chunk(...) (logits) or read_sequence(...)
       (probabilities). Chunks written before 2026-10-30 hold uint8
       probabilities ("probs"); both readers still read them, as logits that
       saturate at about +-16.

Writers for both: open_sequence(...).write(index, values, frame_hw), then
close(), where values are probabilities for png and logits for npz.
"""

import json
import os

import cv2
import numpy as np

from polypbench.paths import ARTIFACTS_ROOT

PREDICTIONS_ROOT = ARTIFACTS_ROOT / "predictions"
CHUNK_SIZE = 2000
PROB_EPS = 1e-7                                          # probabilities -> logits: clip to [eps, 1 - eps]
FLOAT16_MAX = float(np.finfo(np.float16).max)


def storage_format(dataset):
    return "npz" if dataset.startswith("realcolon_") else "png"


def run_dir(dataset, model_id, postproc):
    return PREDICTIONS_ROOT / dataset / model_id / postproc


def sequence_dir(dataset, model_id, postproc, sequence):
    return run_dir(dataset, model_id, postproc) / sequence


def frame_path(dataset, model_id, postproc, sequence, index):
    return sequence_dir(dataset, model_id, postproc, sequence) / f"{index:06d}.png"


def to_uint8(prob):
    """float probabilities in [0, 1] -> uint8 0-255."""
    return np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))   # clip: avoid exp overflow


def prob_to_logit(prob):
    """Probabilities -> logits, for maps only available as probabilities (postprocs, old uint8 chunks).
    Saturates at about +-16 (float32 can't hold 1 - p any closer to 0)."""
    p = np.clip(np.asarray(prob, dtype=np.float32), PROB_EPS, 1 - PROB_EPS)
    return np.log(p) - np.log1p(-p)


def write_prediction(path, prob):
    """prob: float HxW in [0, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), to_uint8(prob)):
        raise RuntimeError(f"Failed to write {path}")


def read_prediction(path):
    """Return the probability map as float32 HxW in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def write_run_info(dataset, model_id, postproc, info):
    path = run_dir(dataset, model_id, postproc) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def is_complete(dataset, model_id, postproc):
    return (run_dir(dataset, model_id, postproc) / "run.json").exists()


def sequence_complete(dataset, model_id, postproc, sequence):
    """True once a sequence is fully written. Only npz runs track sequences; png runs are tracked as a whole."""
    if storage_format(dataset) != "npz":
        return False
    return (sequence_dir(dataset, model_id, postproc, sequence) / "done.json").exists()


def run_finished(dataset, model_id, postproc, all_sequences, written):
    """Whether a run may now be marked complete (run.json): every sequence of the dataset is written.

    npz tracks sequences, so sequences written by earlier (e.g. --sequences)
    runs count; png doesn't, so this run must have written them all itself.
    """
    if storage_format(dataset) == "npz":
        return all(sequence_complete(dataset, model_id, postproc, s) for s in all_sequences)
    return set(written) >= set(all_sequences)


def open_sequence(dataset, model_id, postproc, sequence):
    """A writer for one sequence's predictions, in the dataset's storage format."""
    if storage_format(dataset) == "npz":
        return NpzSequenceWriter(sequence_dir(dataset, model_id, postproc, sequence))
    return PngSequenceWriter(dataset, model_id, postproc, sequence)


class PngSequenceWriter:
    """One PNG per frame, resized to the frame's resolution."""

    def __init__(self, dataset, model_id, postproc, sequence):
        self.key = (dataset, model_id, postproc, sequence)

    def write(self, index, prob, frame_hw=None):
        """prob: float HxW in [0, 1]; resized to frame_hw (h, w) if given and different."""
        if frame_hw is not None and prob.shape != tuple(frame_hw):
            h, w = frame_hw
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        write_prediction(frame_path(*self.key, index), prob)

    def close(self):
        pass


class NpzSequenceWriter:
    """Chunks of CHUNK_SIZE frames of float16 logits at the model's output resolution; done.json on close()."""

    def __init__(self, seq_dir):
        self.dir = seq_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in [*self.dir.glob("chunk_*.npz"), self.dir / "done.json"]:   # from an interrupted run
            stale.unlink(missing_ok=True)
        self.indices, self.maps = [], []
        self.map_hw, self.n_frames, self.n_chunks = None, 0, 0

    def write(self, index, logits, frame_hw=None):
        """logits: float HxW, before the sigmoid; saved as float16 (frame_hw is ignored: maps stay at
        model resolution)."""
        if self.map_hw is None:
            self.map_hw = logits.shape
        elif logits.shape != self.map_hw:
            raise ValueError(f"{self.dir}: frame {index} map is {logits.shape}, earlier frames {self.map_hw}")
        if not np.isfinite(logits).all():
            raise ValueError(f"{self.dir}: frame {index} has non-finite logits")
        self.indices.append(index)
        self.maps.append(np.clip(logits, -FLOAT16_MAX, FLOAT16_MAX).astype(np.float16))
        if len(self.maps) == CHUNK_SIZE:
            self._flush()

    def _flush(self):
        path = self.dir / f"chunk_{self.n_chunks:04d}.npz"
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:   # a file object, so numpy doesn't append ".npz" to the name
            np.savez_compressed(f, frames=np.array(self.indices, dtype=np.int32), logits=np.stack(self.maps))
        os.replace(tmp, path)        # a chunk on disk is always whole
        self.n_frames += len(self.maps)
        self.n_chunks += 1
        self.indices, self.maps = [], []

    def close(self):
        if self.maps:
            self._flush()
        with open(self.dir / "done.json", "w") as f:
            json.dump({"n_frames": self.n_frames, "n_chunks": self.n_chunks,
                       "map_hw": list(self.map_hw) if self.map_hw else None, "values": "logit_f16"}, f, indent=2)


def sequence_info(dataset, model_id, postproc, sequence):
    """done.json of a complete npz sequence: n_frames, n_chunks, map_hw, values ("logit_f16", or
    "prob_u8" for sequences written before logits were stored)."""
    with open(sequence_dir(dataset, model_id, postproc, sequence) / "done.json") as f:
        return {"values": "prob_u8", **json.load(f)}


def read_chunk(dataset, model_id, postproc, sequence, k):
    """(frames int32 N, logits N x H x W) of chunk k of an npz sequence (random access).
    float16 as stored; old uint8-probability chunks come back as float16 logits saturated at
    about +-16."""
    with np.load(sequence_dir(dataset, model_id, postproc, sequence) / f"chunk_{k:04d}.npz") as chunk:
        if "logits" in chunk:
            return chunk["frames"], chunk["logits"]
        return chunk["frames"], prob_to_logit(chunk["probs"] / np.float32(255.0)).astype(np.float16)


def read_sequence(dataset, model_id, postproc, sequence):
    """Yield (index, float32 HxW probability map in [0, 1]) for every frame of an npz sequence, in order."""
    if storage_format(dataset) != "npz":
        raise ValueError(f"{dataset} stores PNGs: read them with read_prediction(frame_path(...))")
    if not sequence_complete(dataset, model_id, postproc, sequence):
        raise FileNotFoundError(f"No complete predictions for {dataset}/{model_id}/{postproc}/{sequence}")
    for k in range(sequence_info(dataset, model_id, postproc, sequence)["n_chunks"]):
        frames, logits = read_chunk(dataset, model_id, postproc, sequence, k)
        for index, lg in zip(frames, logits):
            yield int(index), sigmoid(lg.astype(np.float32))
