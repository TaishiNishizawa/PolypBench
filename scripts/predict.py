"""Run models over a dataset and save a probability map for every frame.

Each model runs once per sequence; every requested post-processing variant is
applied to that same model output and saved separately (see
polypbench/io/predictions.py for the layout: PNGs at frame resolution, or npz
chunks at model resolution for REAL-Colon). Runs whose run.json already
exists are skipped unless --overwrite is given; for REAL-Colon, finished
sequences of an interrupted run are skipped too.

Usage:
    python scripts/predict.py --dataset inhouse --models sacnet-b2@colondb
    python scripts/predict.py --dataset inhouse --models all --postprocs none temporal_smooth
    python scripts/predict.py --dataset realcolon_val --models emcad-b2@sunseg-prime --postprocs none --sequences 004-012
"""

import argparse
import subprocess
import time
from datetime import datetime

import cv2

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.models import load_model, load_model_config, resolve_models
from polypbench.paths import ARTIFACTS_ROOT, REPO_ROOT
from polypbench.postproc import load_postproc, load_postproc_config


def git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_model(dataset, model_id, postproc_names, overwrite, threads, sequences=None):
    """sequences: names of the sequences to run (default: all of `dataset`)."""
    todo = [p for p in postproc_names if overwrite or not predictions.is_complete(dataset.name, model_id, p)]
    for p in set(postproc_names) - set(todo):
        print(f"[skip] {model_id} / {p}: already complete")
    if not todo:
        return

    model = load_model(model_id, threads=threads)
    postprocs = {name: load_postproc(name) for name in todo}
    started = datetime.now().isoformat(timespec="seconds")
    selected = [dataset[name] for name in sequences] if sequences else list(dataset)

    for seq in selected:
        names = [n for n in todo if overwrite or not predictions.sequence_complete(dataset.name, model_id, n, seq.name)]
        if not names:
            print(f"[skip] {model_id} {seq.name}: already complete")
            continue
        t0 = time.time()
        for name in names:
            if postprocs[name] is not None:
                postprocs[name].reset()
        writers = {name: predictions.open_sequence(dataset.name, model_id, name, seq.name) for name in names}
        for idx, rgb in seq.frames():
            prob = model.predict(rgb)
            for name in names:
                post = postprocs[name]
                out = prob if post is None else post(prob, rgb)
                writers[name].write(idx, out, rgb.shape[:2])   # png: resized to the frame; npz: kept as is
        for writer in writers.values():
            writer.close()
        dt = time.time() - t0
        print(f"[{model_id}] {seq.name}: {seq.n_frames} frames in {dt:.1f}s ({seq.n_frames / dt:.1f} fps)")

    all_names = [seq.name for seq in dataset]
    for name in todo:
        if not predictions.run_finished(dataset.name, model_id, name, all_names, [s.name for s in selected]):
            print(f"[partial] {model_id} / {name}: not every sequence of {dataset.name} is done; no run.json yet")
            continue
        predictions.write_run_info(dataset.name, model_id, name, {
            "dataset": dataset.name,
            "model_id": model_id,
            "postproc": name,
            "model_config": load_model_config(model_id),
            "postproc_config": load_postproc_config(name),
            "storage": predictions.storage_format(dataset.name),
            "sequences": {seq.name: seq.n_frames for seq in dataset},
            "providers": model.providers,
            "git_commit": git_commit(),
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model ids, group names (configs/groups.yaml), or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none", "temporal_smooth"])
    parser.add_argument("--sequences", nargs="+",
                        help="only these sequences (default: all); run.json is written once every sequence is done")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU threads for OpenCV and onnxruntime (default 4; their default is every core, "
                             "which overloads a shared server when several runs go at once)")
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)

    model_ids = resolve_models(args.models)

    dataset = load_dataset(args.dataset)
    for model_id in model_ids:
        config = load_model_config(model_id)
        if config["backend"] == "external":
            print(f"[skip] {model_id}: external model, run its own script (see its config)")
            continue
        if not (ARTIFACTS_ROOT / config["weights"]).exists():
            print(f"[skip] {model_id}: no weights yet at {config['weights']} (not trained/exported?)")
            continue
        run_model(dataset, model_id, args.postprocs, args.overwrite, args.threads, args.sequences)


if __name__ == "__main__":
    main()
