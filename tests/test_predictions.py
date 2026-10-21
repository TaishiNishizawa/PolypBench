"""Checks for the prediction writers and readers (polypbench/io/predictions.py), in a temporary folder."""

import json

import numpy as np
import pytest

from polypbench.io import predictions

NPZ = ("realcolon_test", "m@x", "none", "001-013")
PNG = ("inhouse", "m@x", "none", "first_polyp_seen")


@pytest.fixture(autouse=True)
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(predictions, "PREDICTIONS_ROOT", tmp_path)


def maps(n, hw=(6, 8), seed=0):
    return np.random.default_rng(seed).random((n, *hw), dtype=np.float32)


def write_npz(probs, close=True):
    writer = predictions.open_sequence(*NPZ)
    for i, p in enumerate(probs):
        writer.write(i, p, (1080, 1352))     # frame size is ignored: maps stay at model resolution
    if close:
        writer.close()


def test_storage_format():
    assert predictions.storage_format("realcolon_val") == "npz"
    assert predictions.storage_format("realcolon_test") == "npz"
    assert predictions.storage_format("inhouse") == "png"
    assert predictions.storage_format("sunseg_hard_unseen") == "png"


def test_npz_round_trip_across_chunks():
    n = 2 * predictions.CHUNK_SIZE + 500
    probs = maps(n)
    write_npz(probs)
    seq_dir = predictions.sequence_dir(*NPZ)
    assert sorted(p.name for p in seq_dir.glob("chunk_*")) == ["chunk_0000.npz", "chunk_0001.npz", "chunk_0002.npz"]
    assert json.loads((seq_dir / "done.json").read_text()) == {"n_frames": n, "n_chunks": 3, "map_hw": [6, 8]}
    assert predictions.sequence_complete(*NPZ)

    read = list(predictions.read_sequence(*NPZ))
    assert [i for i, _ in read] == list(range(n))
    expected = predictions.to_uint8(probs).astype(np.float32) / 255.0
    np.testing.assert_array_equal(np.stack([p for _, p in read]), expected)


def test_unclosed_sequence_is_incomplete():
    write_npz(maps(10), close=False)
    assert not predictions.sequence_complete(*NPZ)
    with pytest.raises(FileNotFoundError):
        list(predictions.read_sequence(*NPZ))


def test_rewrite_clears_stale_chunks():
    write_npz(maps(predictions.CHUNK_SIZE + 1))         # 2 chunks
    write_npz(maps(10, seed=1))                         # rerun: 1 chunk
    assert len(list(predictions.sequence_dir(*NPZ).glob("chunk_*"))) == 1
    assert len(list(predictions.read_sequence(*NPZ))) == 10


def test_map_size_must_not_change():
    writer = predictions.open_sequence(*NPZ)
    writer.write(0, maps(1, (6, 8))[0])
    with pytest.raises(ValueError):
        writer.write(1, maps(1, (8, 6))[0])


def test_png_writer_resizes_to_frame():
    writer = predictions.open_sequence(*PNG)
    writer.write(3, maps(1, (352, 352))[0], (360, 480))
    writer.close()
    prob = predictions.read_prediction(predictions.frame_path(*PNG, 3))
    assert prob.shape == (360, 480)
    assert not predictions.sequence_complete(*PNG)      # png runs are tracked as a whole (run.json)
    with pytest.raises(ValueError):
        list(predictions.read_sequence(*PNG))
