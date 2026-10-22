"""Build the REAL-Colon frame cache: a smaller copy of every frame, read by the models.

Reads the original frames (NAS, via <data_root>/REAL-Colon) once and writes

    <data_root>/<cache root>/L<long_side>/<video>/<frame:06d>.jpg

resized with INTER_AREA so the long side is `long_side` (aspect ratio kept, no
crop or pad, never upscaled), JPEG at `jpeg_quality` with 4:4:4 chroma.
Every frame is kept; frame ids and frame rate are unchanged. Settings come
from `cache:` in configs/datasets/realcolon.yaml.

<video>/done.json is written last (frame count, original and cached size,
settings), so its presence marks a complete video: reruns skip finished
videos and redo unfinished ones from the start.

Videos run in parallel (--workers), largest first. Each worker reads ahead
with --read-threads threads (NFS latency, ~65 ms per file, is the limit for
one reader) and then does ~60 frames/s, CPU-bound. All workers together are
limited by the NAS (~300 KB per original frame; ~170 files/s measured while
the train-split download was running, up to ~350 with the NAS idle): val +
test (963k frames) takes roughly 1-2 h.

Usage (in tmux):
    python scripts/build_realcolon_cache.py --dry-run          # plan and size only
    python scripts/build_realcolon_cache.py                    # val + test
    python scripts/build_realcolon_cache.py --splits test --workers 4
    python scripts/build_realcolon_cache.py --videos 001-013
"""

import argparse
import json
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import islice
from multiprocessing import Pool

import cv2
import numpy as np

from polypbench.data.realcolon import (cache_dir, cache_info, frame_path, load_realcolon_config,
                                       load_video_info)

KB_PER_FRAME = 121    # measured on 001-013 at L640, q95, 4:4:4
READ_AHEAD = 32       # frames being read ahead per worker


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_ahead(paths, threads):
    """Yield (path, file bytes) in order, keeping up to READ_AHEAD reads in flight.

    Each NFS read waits ~65 ms on the network; one at a time that caps a worker
    at ~15 frames/s, so reads overlap in threads (measured: 8 threads ~7x faster).
    """
    paths = iter(paths)
    with ThreadPoolExecutor(threads) as pool:
        pending = deque((p, pool.submit(read_bytes, p)) for p in islice(paths, READ_AHEAD))
        while pending:
            path, future = pending.popleft()
            nxt = next(paths, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(read_bytes, nxt)))
            yield path, future.result()


def cache_video(job):
    """Cache one video. Returns (video, status, seconds, message); never raises, so other videos go on."""
    video, n_frames, config, read_threads = job
    cv2.setNumThreads(1)                     # parallelism comes from the worker processes
    t0 = time.time()
    try:
        long_side, quality = config["cache"]["long_side"], config["cache"]["jpeg_quality"]
        params = [cv2.IMWRITE_JPEG_QUALITY, quality,
                  cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
        out = cache_dir(video, config)
        out.mkdir(parents=True, exist_ok=True)
        original_wh = cached_wh = None
        sources = (frame_path(video, i, config) for i in range(n_frames))
        for i, (src, data) in enumerate(read_ahead(sources, read_threads)):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"failed to decode {src}")
            h, w = img.shape[:2]
            if original_wh is None:
                original_wh = (w, h)
                scale = min(1.0, long_side / max(w, h))
                cached_wh = (round(w * scale), round(h * scale))
            elif (w, h) != original_wh:
                raise RuntimeError(f"{src} is {w}x{h}, earlier frames are {original_wh[0]}x{original_wh[1]}")
            if cached_wh != original_wh:
                img = cv2.resize(img, cached_wh, interpolation=cv2.INTER_AREA)
            dst = out / f"{i:06d}.jpg"
            if not cv2.imwrite(str(dst), img, params):
                raise RuntimeError(f"failed to write {dst}")
            if (i + 1) % 10000 == 0:
                rate = (i + 1) / (time.time() - t0)
                log(f"{video}: {i + 1}/{n_frames} ({rate:.0f} frames/s, ~{(n_frames - i - 1) / rate / 60:.0f} min left)")
        with open(out / "done.json", "w") as f:
            json.dump({
                "video": video, "n_frames": n_frames,
                "original_wh": list(original_wh), "cached_wh": list(cached_wh),
                "long_side": long_side, "jpeg_quality": quality, "chroma": "4:4:4", "interpolation": "INTER_AREA",
                "source": str(frame_path(video, 0, config).parent),
                "opencv": cv2.__version__, "finished": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2)
        return video, "done", time.time() - t0, f"{original_wh[0]}x{original_wh[1]} -> {cached_wh[0]}x{cached_wh[1]}"
    except Exception as e:                   # report and let the other videos finish
        return video, "failed", time.time() - t0, str(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--videos", nargs="+", help="only these videos (overrides --splits)")
    parser.add_argument("--workers", type=int, default=6,
                        help="videos cached at once (default 6, about what the NAS link can feed)")
    parser.add_argument("--read-threads", type=int, default=8,
                        help="NAS reads in flight per worker (default 8; NFS latency, not bandwidth, is the limit)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and estimated size only")
    args = parser.parse_args()

    config = load_realcolon_config()
    info = load_video_info(config)
    videos = args.videos or [v for s in args.splits for v in config["splits"][s]]
    unknown = sorted(set(videos) - set(info))
    if unknown:
        raise SystemExit(f"Unknown videos: {unknown}")

    todo, skipped = [], []
    for video in videos:
        n_frames = int(info[video]["num_frames"])
        cached = cache_info(video, config)
        if cached is not None and cached["n_frames"] == n_frames:
            skipped.append(video)
        else:
            todo.append((video, n_frames, config, args.read_threads))
    todo.sort(key=lambda job: -job[1])      # largest first, so the long videos don't finish last

    n_todo = sum(job[1] for job in todo)
    need_gb = n_todo * KB_PER_FRAME / 1e6
    root = cache_dir(videos[0], config).parent
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(root).free / 1e9
    log(f"cache: {root} -> {root.resolve()}")
    log(f"{len(skipped)} videos already cached, {len(todo)} to do: {n_todo:,} frames, "
        f"~{need_gb:.0f} GB (free {free_gb:,.0f} GB), ~{n_todo / 350 / 60:.0f}-{n_todo / 170 / 60:.0f} min "
        f"depending on NAS load")
    for video, n_frames, *_ in todo:
        log(f"  {video}: {n_frames:,} frames")
    if args.dry_run or not todo:
        return
    if need_gb > 0.9 * free_gb:
        raise SystemExit("Not enough free space for the cache.")

    t0, failed = time.time(), []
    with Pool(min(args.workers, len(todo))) as pool:
        for video, status, seconds, message in pool.imap_unordered(cache_video, todo):
            log(f"{video}: {status} in {seconds / 60:.1f} min ({message})")
            if status == "failed":
                failed.append(video)
    log(f"finished in {(time.time() - t0) / 60:.0f} min; {len(todo) - len(failed)} cached, {len(failed)} failed")
    if failed:
        raise SystemExit(f"Failed: {failed} (rerun to retry; finished videos are skipped)")


if __name__ == "__main__":
    main()
