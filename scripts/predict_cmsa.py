"""Run CMSA-Net over a dataset and save a probability map for every frame.

Runs in the `sali` env (envs/sali.yml), not the main one:

    conda activate sali
    python scripts/predict_cmsa.py --dataset sunseg_hard_unseen --model_id cmsa-net@sunseg-ft-s0
    python scripts/predict_cmsa.py --dataset realcolon_val --model_id cmsa-net@sunseg-ft-s0 --sequences 004-012

Inference follows CMSA-Net's own my_test.py (see polypbench/models/cmsa.py):
6-frame windows of two dynamically updated references and the 4 most recent
frames, run frame by frame. my_test.py saves the sigmoid output as is (no
min-max), and so does this (postproc "none").

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
from polypbench.models.cmsa import build_cmsa, build_transform, predict_sequence
from polypbench.paths import ARTIFACTS_ROOT, REPO_ROOT

POSTPROC = "none"


def load_cmsa(config):
    model = build_cmsa(config)
    state = torch.load(ARTIFACTS_ROOT / config["weights"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequences", nargs="+",
                        help="only these sequences (default: all); run.json is written once every sequence is done")
    args = parser.parse_args()

    model_id = args.model_id
    dataset = load_dataset(args.dataset)
    if predictions.is_complete(dataset.name, model_id, POSTPROC) and not args.overwrite:
        print(f"[skip] {model_id} / {POSTPROC}: already complete (use --overwrite)")
        return

    config = load_model_config(model_id)
    if config["arch"] != "cmsa-net":
        raise SystemExit(f"{model_id} is not a CMSA-Net model (arch {config['arch']})")
    model = load_cmsa(config)
    transform = build_transform(config)

    full_res = predictions.storage_format(dataset.name) == "png"   # npz keeps CMSA-Net's 352x352 output
    started = datetime.now().isoformat(timespec="seconds")
    selected = [dataset[name] for name in args.sequences] if args.sequences else list(dataset)
    for seq in selected:
        if not args.overwrite and predictions.sequence_complete(dataset.name, model_id, POSTPROC, seq.name):
            print(f"[skip] {model_id} {seq.name}: already complete")
            continue
        t0 = time.time()
        writer = predictions.open_sequence(dataset.name, model_id, POSTPROC, seq.name)
        for idx, prob in predict_sequence(model, seq, transform, config, full_res):
            writer.write(idx, prob)
        writer.close()
        dt = time.time() - t0
        print(f"[{model_id}] {seq.name}: {seq.n_frames} frames in {dt:.1f}s ({seq.n_frames / dt:.1f} fps)")

    if not predictions.run_finished(dataset.name, model_id, POSTPROC, [s.name for s in dataset],
                                    [s.name for s in selected]):
        print(f"[partial] {model_id}: not every sequence of {dataset.name} is done; no run.json yet")
        return
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT / config["code"], text=True).strip()
    predictions.write_run_info(dataset.name, model_id, POSTPROC, {
        "dataset": dataset.name,
        "model_id": model_id,
        "postproc": POSTPROC,
        "model_config": config,
        "storage": predictions.storage_format(dataset.name),
        "sequences": {seq.name: seq.n_frames for seq in dataset},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cmsa_commit": code_commit,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
