"""Turn saved REAL-Colon probability maps into boxes (one per connected component).

Reads the npz predictions written by predict.py / predict_sali.py and writes one
box table per video (see polypbench/postproc/boxes.py for the columns), scaled to
each video's original frame size from the ground-truth videos table
(scripts/build_realcolon_gt.py). Skips videos already done unless --overwrite;
run.json is written once every video of the dataset is done. One video per
worker process; I/O bound (each video's chunks are read once).

Usage:
    python scripts/extract_boxes.py --dataset realcolon_val --models emcad-b2@sunseg-ft-s0 --sequences 004-012
    python scripts/extract_boxes.py --dataset realcolon_test --models emcad-b2@sunseg-ft-s0 sali@sunseg-causal-ft-s0
"""

import argparse
import subprocess
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import cv2

from polypbench.data.realcolon import SPLITS, load_realcolon_config
from polypbench.data.realcolon_gt import load_ground_truth
from polypbench.io import predictions
from polypbench.models import resolve_models
from polypbench.paths import REPO_ROOT
from polypbench.postproc.boxes import (boxes_path, extract_boxes, load_boxes_config, write_boxes_run_info,
                                       write_sequence_boxes)


def git_commit():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    return commit + ("-dirty" if dirty else "")


def extract_video(job, dataset, boxes_name, config):
    """job: (model_id, postproc, video, (width, height)). Returns (job, n_frames, n_boxes, seconds)."""
    cv2.setNumThreads(1)
    model_id, postproc, video, frame_wh = job
    t0 = time.time()
    rows, n_frames = [], 0
    info = predictions.sequence_info(dataset, model_id, postproc, video)
    for k in range(info["n_chunks"]):
        frames, probs = predictions.read_chunk(dataset, model_id, postproc, video, k)
        for frame, prob in zip(frames, probs):
            rows += [(int(frame), *box) for box in extract_boxes(
                prob, frame_wh, config["threshold"], config["connectivity"], config["min_area_px"])]
        n_frames += len(frames)
    if n_frames != info["n_frames"]:
        raise ValueError(f"{model_id}/{postproc}/{video}: read {n_frames} frames, done.json says {info['n_frames']}")
    write_sequence_boxes(boxes_path(dataset, model_id, postproc, boxes_name, video), rows)
    return job, n_frames, len(rows), time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(SPLITS))
    parser.add_argument("--models", nargs="+", required=True, help="model ids, group names (configs/groups.yaml), or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none"])
    parser.add_argument("--boxes", default="cc-0.5", help="configs/boxes/<name>.yaml (default cc-0.5)")
    parser.add_argument("--sequences", nargs="+", help="only these videos (default: all of the split)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_boxes_config(args.boxes)
    all_videos = load_realcolon_config()["splits"][SPLITS[args.dataset]]
    videos = args.sequences or all_videos
    if unknown := sorted(set(videos) - set(all_videos)):
        raise SystemExit(f"Not in {args.dataset}: {unknown}")
    sizes = {r.video: (r.width, r.height) for r in load_ground_truth().videos.itertuples()}

    jobs, runs = [], []
    for model_id in resolve_models(args.models):
        for postproc in args.postprocs:
            runs.append((model_id, postproc))
            for video in videos:
                if not predictions.sequence_complete(args.dataset, model_id, postproc, video):
                    print(f"[skip] {model_id}/{postproc} {video}: no complete predictions")
                elif boxes_path(args.dataset, model_id, postproc, args.boxes, video).exists() and not args.overwrite:
                    print(f"[skip] {model_id}/{postproc} {video}: boxes already extracted")
                else:
                    jobs.append((model_id, postproc, video, sizes[video]))

    started = datetime.now().isoformat(timespec="seconds")
    work = partial(extract_video, dataset=args.dataset, boxes_name=args.boxes, config=config)
    with Pool(min(args.workers, max(1, len(jobs)))) as pool:
        for (model_id, postproc, video, _), n_frames, n_boxes, dt in pool.imap_unordered(work, jobs):
            print(f"[{model_id}/{postproc}] {video}: {n_frames:,} frames, {n_boxes:,} boxes "
                  f"({n_boxes / n_frames:.2f}/frame) in {dt:.0f}s", flush=True)

    for model_id, postproc in runs:
        if not all(boxes_path(args.dataset, model_id, postproc, args.boxes, v).exists() for v in all_videos):
            print(f"[partial] {model_id}/{postproc}: not every video of {args.dataset} has boxes; no run.json yet")
            continue
        write_boxes_run_info(args.dataset, model_id, postproc, args.boxes, {
            "dataset": args.dataset, "model_id": model_id, "postproc": postproc, "boxes_config": config,
            "videos": all_videos, "git_commit": git_commit(), "started": started,
            "finished": datetime.now().isoformat(timespec="seconds")})


if __name__ == "__main__":
    main()
