"""Stability graphs, ported from LambdaDiceEval/plot_interframe_iou.py. Images are RGB uint8."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

# palette shared with LambdaDiceEval's plots
BLUE = "#2a78d6"     # motion-compensated IoU
ORANGE = "#eb6834"   # raw IoU
AQUA = "#1baf7a"     # frame change
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#a35bd8", "#d4a017", "#e0457b", "#5b6b7a", "#3fb6c9"]

PANEL_H = 540
GRAPH_W = 960
DPI = 100


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _to_rgb(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return img


def stability_graph(df, n_frames, title):
    """One run on one clip: top = consecutive-mask IoU (motion-compensated and raw),
    bottom = how much the frame itself changed. Returns (RGB image, ax_iou, ax_diff);
    the axes stay usable for mapping data coordinates to pixels."""
    fig, (ax_iou, ax_diff) = plt.subplots(
        2, 1, sharex=True, figsize=(GRAPH_W / DPI, PANEL_H / DPI), dpi=DPI,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )
    fig.subplots_adjust(left=0.10, right=0.975, top=0.875, bottom=0.12)
    x = df["frame"].to_numpy()

    ax_iou.plot(x, df["iou_raw"], color=ORANGE, lw=1.0, alpha=0.85, label="raw IoU (k-1 vs k)")
    ax_iou.plot(x, df["iou_warped"], color=BLUE, lw=1.4, label="motion-compensated IoU")
    ax_iou.set_ylim(0, 1.03)
    ax_iou.set_ylabel("Mask IoU", color=INK, fontsize=10)
    ax_iou.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8.5, frameon=False, ncol=2)

    ax_diff.plot(x, df["frame_diff"], color=AQUA, lw=1.2)
    ax_diff.set_ylim(bottom=0)
    ax_diff.set_ylabel("Frame change\n(mean |Δgray|)", color=INK, fontsize=10)
    ax_diff.set_xlabel("Frame index", color=INK, fontsize=10)
    ax_diff.set_xlim(0, n_frames - 1)
    for ax in (ax_iou, ax_diff):
        _style(ax)

    fig.text(0.10, 0.965, title, color=INK, fontsize=11, fontweight="bold", va="center")
    img = _to_rgb(fig)
    plt.close(fig)
    return img, ax_iou, ax_diff


def comparison_graph(runs, n_frames, title):
    """Several runs on one clip: motion-compensated IoU per run, plus frame change.
    runs: {label: DataFrame}. Returns an RGB image."""
    fig, (ax_iou, ax_diff) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6.5), dpi=DPI,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.12},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.10)
    for i, (label, df) in enumerate(runs.items()):
        ax_iou.plot(df["frame"], df["iou_warped"], color=SERIES[i % len(SERIES)], lw=1.2,
                    label=f"{label}  (mean {df['iou_warped'].mean():.3f})")
    ax_iou.set_ylim(0, 1.03)
    ax_iou.set_ylabel("Motion-compensated IoU", color=INK, fontsize=10)
    ax_iou.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8.5, frameon=False, ncol=3)

    any_df = next(iter(runs.values()))
    ax_diff.plot(any_df["frame"], any_df["frame_diff"], color=MUTED, lw=1.0)
    ax_diff.set_ylim(bottom=0)
    ax_diff.set_ylabel("Frame change", color=INK, fontsize=10)
    ax_diff.set_xlabel("Frame index", color=INK, fontsize=10)
    ax_diff.set_xlim(0, n_frames - 1)
    for ax in (ax_iou, ax_diff):
        _style(ax)

    fig.text(0.08, 0.965, title, color=INK, fontsize=12, fontweight="bold", va="center")
    img = _to_rgb(fig)
    plt.close(fig)
    return img


def cursor_geometry(ax_iou, ax_diff, n_frames, height):
    """Pixel x of every frame index, and the pixel y extents of the two panels (image coords)."""
    xs = ax_iou.transData.transform(np.column_stack([np.arange(n_frames), np.zeros(n_frames)]))[:, 0]

    def y_extent(ax):
        bb = ax.get_window_extent()
        return int(round(height - bb.y1)), int(round(height - bb.y0))

    return xs, y_extent(ax_iou), y_extent(ax_diff)


def iou_to_pixel_y(ax_iou, iou, height):
    return int(round(height - ax_iou.transData.transform((0, iou))[1]))
