"""Run SALI over a dataset and save a probability map for every frame.

Runs in the `sali` env (envs/sali.yml), not the main one:

    conda activate sali
    python scripts/predict_sali.py --dataset inhouse                                  # sali@sunseg
    python scripts/predict_sali.py --dataset inhouse --model_id sali@sunseg-causal

Inference follows SALI's own test_video.py (see polypbench/models/sali.py),
with the partner frame (next or previous) set by `partner:` in the model
config. The one deliberate difference: SALI's script min-max rescales every
frame to [0, 1], which forces confident pixels even on frames with no polyp.
Here the raw sigmoid is saved (postproc "none").

Predictions use the same layout as predict.py, so evaluate.py and
visualize.py work unchanged.
"""

import argparse
import subprocess
import time
from datetime import datetime

import torch

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.models import load_model_config
from polypbench.models.sali import build_sali, build_transform, predict_sequence
from polypbench.paths import ARTIFACTS_ROOT, REPO_ROOT

POSTPROC = "none"


def load_sali(config):
    model = build_sali(config)
    state = torch.load(ARTIFACTS_ROOT / config["weights"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_id", default="sali@sunseg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id
    dataset = load_dataset(args.dataset)
    if predictions.is_complete(dataset.name, model_id, POSTPROC) and not args.overwrite:
        print(f"[skip] {model_id} / {POSTPROC}: already complete (use --overwrite)")
        return

    config = load_model_config(model_id)
    if config["arch"] != "sali":
        raise SystemExit(f"{model_id} is not a SALI model (arch {config['arch']})")
    model = load_sali(config)
    transform = build_transform(config)

    started = datetime.now().isoformat(timespec="seconds")
    for seq in dataset:
        t0 = time.time()
        for idx, prob in predict_sequence(model, seq, transform, config["mem_freq"], config["partner"]):
            predictions.write_prediction(predictions.frame_path(dataset.name, model_id, POSTPROC, seq.name, idx), prob)
        dt = time.time() - t0
        print(f"[{model_id}] {seq.name}: {seq.n_frames} frames in {dt:.1f}s ({seq.n_frames / dt:.1f} fps)")

    sali_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT / config["code"], text=True).strip()
    predictions.write_run_info(dataset.name, model_id, POSTPROC, {
        "dataset": dataset.name,
        "model_id": model_id,
        "postproc": POSTPROC,
        "model_config": config,
        "sequences": {seq.name: seq.n_frames for seq in dataset},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "sali_commit": sali_commit,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
