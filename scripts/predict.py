"""Run models over a dataset and save a probability map for every frame.

Each model runs once per sequence; every requested post-processing variant is
applied to that same model output and saved separately (see
polypbench/io/predictions.py for the layout). Runs whose run.json already
exists are skipped unless --overwrite is given.

Usage:
    python scripts/predict.py --dataset inhouse --models sacnet-b2@colondb
    python scripts/predict.py --dataset inhouse --models all --postprocs none temporal_smooth
"""

import argparse
import subprocess
import time
from datetime import datetime

import cv2

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.models import MODELS_CONFIG_DIR, load_model, load_model_config
from polypbench.paths import REPO_ROOT
from polypbench.postproc import load_postproc, load_postproc_config


def git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_model(dataset, model_id, postproc_names, overwrite):
    todo = [p for p in postproc_names if overwrite or not predictions.is_complete(dataset.name, model_id, p)]
    for p in set(postproc_names) - set(todo):
        print(f"[skip] {model_id} / {p}: already complete")
    if not todo:
        return

    model = load_model(model_id)
    postprocs = {name: load_postproc(name) for name in todo}
    started = datetime.now().isoformat(timespec="seconds")

    for seq in dataset:
        t0 = time.time()
        for post in postprocs.values():
            if post is not None:
                post.reset()
        for idx, rgb in seq.frames():
            prob = model.predict(rgb)
            h, w = rgb.shape[:2]
            for name, post in postprocs.items():
                out = prob if post is None else post(prob, rgb)
                out_full = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
                predictions.write_prediction(
                    predictions.frame_path(dataset.name, model_id, name, seq.name, idx), out_full)
        dt = time.time() - t0
        print(f"[{model_id}] {seq.name}: {seq.n_frames} frames in {dt:.1f}s ({seq.n_frames / dt:.1f} fps)")

    for name in todo:
        predictions.write_run_info(dataset.name, model_id, name, {
            "dataset": dataset.name,
            "model_id": model_id,
            "postproc": name,
            "model_config": load_model_config(model_id),
            "postproc_config": load_postproc_config(name),
            "sequences": {seq.name: seq.n_frames for seq in dataset},
            "providers": model.providers,
            "git_commit": git_commit(),
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model ids, or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none", "temporal_smooth"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_ids = args.models
    if model_ids == ["all"]:
        model_ids = sorted(p.stem for p in MODELS_CONFIG_DIR.glob("*.yaml"))

    dataset = load_dataset(args.dataset)
    for model_id in model_ids:
        if load_model_config(model_id)["backend"] == "external":
            print(f"[skip] {model_id}: external model, run its own script (see its config)")
            continue
        run_model(dataset, model_id, args.postprocs, args.overwrite)


if __name__ == "__main__":
    main()
