"""Stability graphs and side-by-side videos from the CSVs written by stability.py.

Ported from LambdaDiceEval/plot_interframe_iou.py. Per model x postproc x clip:
  - a static graph: consecutive-mask IoU (motion-compensated and raw) on top,
    frame change below;
  - a side-by-side video: the prediction overlaid on the clip on the left, the
    same graph on the right with a cursor and a red dot on the
    motion-compensated IoU line tracking the current frame.
Per postproc x clip, also one comparison graph with every model's
motion-compensated IoU.

Frame t is drawn against the transition (t-1 -> t). Frame 0 has no
transition, and frames whose masks are empty in both t-1 and t have no IoU;
in both cases only the cursor is drawn.

Outputs, under <artifacts>/reports/<dataset>/stability/:
    <postproc>/<model_id>/<clip>.png               one per model (not grouped)
    <postproc>/<model_id>/<clip>_side_by_side.mp4
    compare/<name>/<postproc>/<clip>.png           <name> = model group (configs/groups.yaml) or --name

Usage:
    python scripts/visualize_stability.py --dataset inhouse --models sunseg
    python scripts/visualize_stability.py --dataset inhouse --models sali@sunseg --no-videos
"""

import argparse

import cv2
import numpy as np
import pandas as pd

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.metrics.overlap import binarize
from polypbench.models import load_groups, resolve_models
from polypbench.paths import ARTIFACTS_ROOT
from polypbench.viz.overlays import PRED_COLOR, VideoWriter, label, overlay
from polypbench.viz.stability import (PANEL_H, comparison_graph, cursor_geometry, iou_to_pixel_y,
                                      stability_graph)

DOT_RED = (229, 35, 27)   # RGB
INK = (11, 11, 11)


def save_png(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def render_side_by_side(dataset_name, seq, model_id, postproc, threshold, df, graph, ax_iou, ax_diff, out_path):
    xs, (top0, top1), (bot0, bot1) = cursor_geometry(ax_iou, ax_diff, seq.n_frames, PANEL_H)
    iou_by_frame = df.set_index("frame")["iou_warped"]
    writer = None
    try:
        for t, rgb in seq.frames():
            prob = predictions.read_prediction(predictions.frame_path(dataset_name, model_id, postproc, seq.name, t))
            left = label(overlay(rgb, binarize(prob, threshold), PRED_COLOR), [f"{model_id} | {postproc}"])
            vid_w = int(round(rgb.shape[1] * PANEL_H / rgb.shape[0]))
            left = cv2.resize(left, (vid_w, PANEL_H), interpolation=cv2.INTER_LINEAR)
            if writer is None:
                writer = VideoWriter(out_path, seq.fps, vid_w + graph.shape[1], PANEL_H)

            canvas = graph.copy()
            cx = int(round(xs[t]))
            cv2.line(canvas, (cx, top0), (cx, top1), INK, 1, cv2.LINE_AA)
            cv2.line(canvas, (cx, bot0), (cx, bot1), INK, 1, cv2.LINE_AA)
            iou = iou_by_frame.get(t, np.nan)
            if np.isfinite(iou):
                cy = iou_to_pixel_y(ax_iou, iou, PANEL_H)
                cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 6, DOT_RED, -1, cv2.LINE_AA)
                text = f"frame {t}   IoU {iou:.3f}"
            elif t == 0:
                text = "frame 0   (no previous frame)"
            else:
                text = f"frame {t}   IoU n/a (no mask in either frame)"
            # right-aligned on the legend row, below the title
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, text, (canvas.shape[1] - tw - 25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1,
                        cv2.LINE_AA)
            writer.write(np.hstack([left, canvas]))
    finally:
        if writer is not None:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="model ids, group names (configs/groups.yaml), or 'all'")
    parser.add_argument("--postprocs", nargs="+", default=["none", "temporal_smooth"])
    parser.add_argument("--sequences", nargs="+", help="only these clips (default: all)")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--name", help="comparison-graph folder (default: the group name, when --models is one group)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for OpenCV (default: every core)")
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    name = args.name or (args.models[0] if len(args.models) == 1 and
                         (args.models[0] == "all" or args.models[0] in load_groups()["groups"]) else None)
    if name is None:
        raise SystemExit("Pass --name for a custom set of models, or use a group name")

    model_ids = resolve_models(args.models)
    dataset = load_dataset(args.dataset)
    stab_dir = ARTIFACTS_ROOT / "stability" / dataset.name
    out_root = ARTIFACTS_ROOT / "reports" / dataset.name / "stability"

    for postproc in args.postprocs:
        tables = {}
        for model_id in model_ids:
            path = stab_dir / model_id / f"{postproc}.csv"
            if path.exists():
                tables[model_id] = pd.read_csv(path)
            else:
                print(f"[skip] {model_id} / {postproc}: no stability CSV (run stability.py first)")
        if not tables:
            continue
        for seq in dataset:
            if args.sequences and seq.name not in args.sequences:
                continue
            per_model = {m: df[df.sequence == seq.name] for m, df in tables.items()}
            save_png(out_root / "compare" / name / postproc / f"{seq.name}.png",
                     comparison_graph(per_model, seq.n_frames,
                                      f"{seq.name} - inter-frame mask consistency ({postproc})"))
            for model_id, df in per_model.items():
                threshold = float(df.threshold.iloc[0])
                graph, ax_iou, ax_diff = stability_graph(
                    df, seq.n_frames, f"{model_id} | {seq.name} - inter-frame mask consistency ({postproc})")
                save_png(out_root / postproc / model_id / f"{seq.name}.png", graph)
                if not args.no_videos:
                    render_side_by_side(dataset.name, seq, model_id, postproc, threshold, df, graph,
                                        ax_iou, ax_diff, out_root / postproc / model_id / f"{seq.name}_side_by_side.mp4")
            print(f"[{postproc}] {seq.name}: {len(per_model)} models")
    print(f"Written to {out_root}")


if __name__ == "__main__":
    main()
