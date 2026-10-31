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


def logit_maps(n, hw=(6, 8), seed=0):
    return np.random.default_rng(seed).normal(0, 10, (n, *hw)).astype(np.float32)


def write_npz(logits, close=True):
    writer = predictions.open_sequence(*NPZ)
    for i, lg in enumerate(logits):
        writer.write(i, lg, (1080, 1352))    # frame size is ignored: maps stay at model resolution
    if close:
        writer.close()


def test_storage_format():
    assert predictions.storage_format("realcolon_val") == "npz"
    assert predictions.storage_format("realcolon_test") == "npz"
    assert predictions.storage_format("inhouse") == "png"
    assert predictions.storage_format("sunseg_hard_unseen") == "png"


def test_npz_round_trip_across_chunks():
    n = 2 * predictions.CHUNK_SIZE + 500
    logits = logit_maps(n)
    write_npz(logits)
    seq_dir = predictions.sequence_dir(*NPZ)
    assert sorted(p.name for p in seq_dir.glob("chunk_*")) == ["chunk_0000.npz", "chunk_0001.npz", "chunk_0002.npz"]
    assert json.loads((seq_dir / "done.json").read_text()) == {
        "n_frames": n, "n_chunks": 3, "map_hw": [6, 8], "values": "logit_f16"}
    assert predictions.sequence_complete(*NPZ)

    stored = np.concatenate([predictions.read_chunk(*NPZ, k)[1] for k in range(3)])
    assert stored.dtype == np.float16
    np.testing.assert_array_equal(stored, logits.astype(np.float16))
    read = list(predictions.read_sequence(*NPZ))
    assert [i for i, _ in read] == list(range(n))
    np.testing.assert_allclose(np.stack([p for _, p in read]), predictions.sigmoid(logits), atol=1e-3)


def test_logits_keep_confident_regions_apart():
    """Why logits: probabilities this confident all become 255 in uint8 (and 1.0 in float32)."""
    logits = np.array([[8.0, 12.0, 20.0, 40.0]], np.float32)
    assert len(set(predictions.to_uint8(predictions.sigmoid(logits)).ravel())) == 1
    write_npz([logits])
    assert len(set(predictions.read_chunk(*NPZ, 0)[1].ravel())) == 4


def test_old_uint8_chunks_still_read():
    """Chunks written before 2026-10-30 hold uint8 probabilities under "probs" and no "values" in done.json."""
    seq_dir = predictions.sequence_dir(*NPZ)
    seq_dir.mkdir(parents=True)
    probs = np.array([[[0, 128, 255]]], np.uint8)
    np.savez_compressed(seq_dir / "chunk_0000.npz", frames=np.array([0], np.int32), probs=probs)
    (seq_dir / "done.json").write_text(json.dumps({"n_frames": 1, "n_chunks": 1, "map_hw": [1, 3]}))
    assert predictions.sequence_info(*NPZ)["values"] == "prob_u8"
    logits = predictions.read_chunk(*NPZ, 0)[1].astype(np.float32)
    assert logits[0, 0, 1] > 0 and logits[0, 0, 0] < 0
    assert (np.abs(logits[0, 0, [0, 2]]) > 15.5).all() and (np.abs(logits) < 16.5).all()   # saturated at ~16
    (_, prob), = predictions.read_sequence(*NPZ)
    np.testing.assert_allclose(prob, probs[0] / 255.0, atol=1e-3)


def test_non_finite_logits_rejected():
    writer = predictions.open_sequence(*NPZ)
    with pytest.raises(ValueError):
        writer.write(0, np.array([[np.inf]], np.float32))


def test_unclosed_sequence_is_incomplete():
    write_npz(logit_maps(10), close=False)
    assert not predictions.sequence_complete(*NPZ)
    with pytest.raises(FileNotFoundError):
        list(predictions.read_sequence(*NPZ))


def test_rewrite_clears_stale_chunks():
    write_npz(logit_maps(predictions.CHUNK_SIZE + 1))   # 2 chunks
    write_npz(logit_maps(10, seed=1))                   # rerun: 1 chunk
    assert len(list(predictions.sequence_dir(*NPZ).glob("chunk_*"))) == 1
    assert len(list(predictions.read_sequence(*NPZ))) == 10


def test_map_size_must_not_change():
    writer = predictions.open_sequence(*NPZ)
    writer.write(0, logit_maps(1, (6, 8))[0])
    with pytest.raises(ValueError):
        writer.write(1, logit_maps(1, (8, 6))[0])


def test_png_writer_resizes_to_frame():
    writer = predictions.open_sequence(*PNG)
    writer.write(3, maps(1, (352, 352))[0], (360, 480))
    writer.close()
    prob = predictions.read_prediction(predictions.frame_path(*PNG, 3))
    assert prob.shape == (360, 480)
    assert not predictions.sequence_complete(*PNG)      # png runs are tracked as a whole (run.json)
    with pytest.raises(ValueError):
        list(predictions.read_sequence(*PNG))


def test_run_finished_npz_counts_earlier_sequences():
    """A --sequences run marks the run complete only once every sequence has its done.json."""
    names = ["001-013", "001-014"]
    assert not predictions.run_finished("realcolon_test", "m@x", "none", names, ["001-013"])
    for name in names:
        writer = predictions.open_sequence("realcolon_test", "m@x", "none", name)
        writer.write(0, maps(1)[0])
        if name == "001-013":
            writer.close()
    assert not predictions.run_finished("realcolon_test", "m@x", "none", names, ["001-014"])   # 001-014 unclosed
    predictions.open_sequence("realcolon_test", "m@x", "none", "001-014").close()
    assert predictions.run_finished("realcolon_test", "m@x", "none", names, ["001-014"])


def test_run_finished_png_needs_every_sequence_this_run():
    names = ["a", "b"]
    assert not predictions.run_finished("inhouse", "m@x", "none", names, ["a"])
    assert predictions.run_finished("inhouse", "m@x", "none", names, ["a", "b"])
