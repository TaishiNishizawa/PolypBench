"""Build the official SUN-SEG layout from the raw SUN frames and SUN-SEG annotations.

Same mapping as PNS+'s utils/reorganize.py (github.com/GewelsJI/VPS): the mask
<split>/GT/<clip>/<name>.png belongs to the raw frame SUN-Positive/<case>/<name>.jpg,
where <case> is <clip> up to the first "_" (case1_3 -> case1). Split membership
comes entirely from the annotation download.

Unlike the official script, nothing is moved or deleted: every file is
hard-linked into the new layout, so it takes no extra disk space and the raw
downloads stay intact. Safe to rerun: existing links are skipped.

    <data_root>/SUN-SEG/
    ├── SUN-Positive/                         raw frames            (input, untouched)
    ├── SUN-SEG-Annotation/SUN-SEG-Annotation raw annotations       (input, untouched)
    └── SUN-SEG/<split>/{Frame, GT, Edge, ...}  official layout     (output)

Usage:
    python scripts/build_sunseg.py
"""

import os
from pathlib import Path

from polypbench.paths import DATA_ROOT

RAW_ROOT = DATA_ROOT / "SUN-SEG"
FRAMES_ROOT = RAW_ROOT / "SUN-Positive"
ANNOTATION_ROOT = RAW_ROOT / "SUN-SEG-Annotation" / "SUN-SEG-Annotation"
OUT_ROOT = RAW_ROOT / "SUN-SEG"

# split -> (clips, frames), from the SUN-SEG paper / PNS+ docs
EXPECTED = {
    "TrainDataset": (112, 19544),
    "TestEasyDataset/Seen": (33, 4719),
    "TestEasyDataset/Unseen": (86, 12351),
    "TestHardDataset/Seen": (17, 3882),
    "TestHardDataset/Unseen": (37, 8640),
}


def link(src, dst):
    """Hard-link src to dst. Returns True if a new link was made, False if it already existed."""
    if dst.exists():
        if not os.path.samefile(src, dst):
            raise RuntimeError(f"{dst} exists but is not a link to {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
    return True


def link_annotations(split):
    """Link every annotation file (GT, Edge, Scribble, ...) of a split, keeping relative paths."""
    src_root = ANNOTATION_ROOT / split
    made = 0
    for src in src_root.rglob("*"):
        rel = src.relative_to(src_root)
        if src.is_file() and rel.parts[0] != "Frame":   # the download ships an empty Frame/ dir
            made += link(src, OUT_ROOT / split / rel)
    return made


def link_frames(split):
    """Link the raw frame for every GT mask into <split>/Frame/<clip>/."""
    made = 0
    for mask in sorted((ANNOTATION_ROOT / split / "GT").glob("*/*.png")):
        clip = mask.parent.name
        case = clip.split("_")[0]
        src = FRAMES_ROOT / case / f"{mask.stem}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"No raw frame for {mask}: expected {src}")
        made += link(src, OUT_ROOT / split / "Frame" / clip / f"{mask.stem}.jpg")
    return made


def verify(split):
    """Frame/ and GT/ have the same clips and frame names, in the expected numbers."""
    frame_dir, gt_dir = OUT_ROOT / split / "Frame", OUT_ROOT / split / "GT"
    frames = {p.relative_to(frame_dir).with_suffix("") for p in frame_dir.glob("*/*.jpg")}
    masks = {p.relative_to(gt_dir).with_suffix("") for p in gt_dir.glob("*/*.png")}
    if frames != masks:
        raise RuntimeError(f"{split}: {len(frames - masks)} frames without a mask, "
                           f"{len(masks - frames)} masks without a frame")
    clips = len({p.parent for p in frames})
    if (clips, len(frames)) != EXPECTED[split]:
        raise RuntimeError(f"{split}: got {clips} clips / {len(frames)} frames, expected {EXPECTED[split]}")
    return clips, len(frames)


def main():
    for root in (FRAMES_ROOT, ANNOTATION_ROOT):
        if not root.is_dir():
            raise FileNotFoundError(root)
    print(f"Writing {OUT_ROOT}")
    for split in EXPECTED:
        n_ann = link_annotations(split)
        n_frames = link_frames(split)
        clips, frames = verify(split)
        print(f"{split:24s} {clips:4d} clips {frames:6d} frames  "
              f"(new links: {n_frames} frames, {n_ann} annotation files)")
    print("Done. Raw folders are untouched.")


if __name__ == "__main__":
    main()
