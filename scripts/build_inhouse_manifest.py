"""
This file reads the inhouse dataset directory struct and generates a manifest file for the dataset. 
The manifest file should be a .yaml file that contains the following information for each video in the dataset:
- video type (first_polyp_seen, first_polyp_removed, second_polyp_seen, second_polyp_removed)
- video name
- source_path: the starting frame for that clip in the 8-minute source video
- annotations: 0-based video frame to file name mapping for the annotated frames in that clip


Target shape: 
name: inhouse
root: in-house-polyp-video        # relative to data_root
images_dir: annotations/images
masks_dir: annotations/masks
sequences:
  first_polyp_seen:
    video: first_polyp_seen.mp4
    source_start: 0               # 0-based frame in the 8-min source video
    annotations:                  # 0-based video frame -> file name (same in images/ and masks/)
      0: frame1.png
      15: frame2.png
      # ...
  # first_polyp_removed, second_polyp_seen, second_polyp_removed
"""
import cv2 
import numpy as np
import yaml 

from polypbench.paths import CONFIG_DIR, DATA_ROOT

DATASET_ROOT = "in-house-polyp-video"
IMAGES_DIR = "annotations/images"
MASKS_DIR = "annotations/masks"
OUT_PATH = CONFIG_DIR / "datasets" / "inhouse.yaml"

# Mean abs diff in [0, 1]. Exact frames score ~0.001-0.002 (compression noise).
MAX_DIFF = 0.01

CLIPS = {
    "first_polyp_seen": {
        "video": "first_polyp_seen.mp4",
        "source_start": 0,
        "annotations":  list(range(1, 32))
    },
    "first_polyp_removed": {
        "video": "first_polyp_removed.mp4",
        "source_start": 3295,
        "annotations":  list(range(32, 63))
    },
    "second_polyp_seen": {
        "video": "second_polyp_seen.mp4",
        "source_start": 3750,
        "annotations":  [*range(63, 70), *range(71, 94)]
    },
    "second_polyp_removed": {
        "video": "second_polyp_removed.mp4",
        "source_start": 4834,
        "annotations":  [*range(94, 101), 70, *range(101, 124)]
    }    
}

def read_all_frames(video_path):
    "Returns a list of all frames in the video, in order."
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from video {video_path}")
    return frames, fps

def mean_abs_diff(frame1, frame2):
    "Mean absolute difference between two frames, normalized to [0, 1]."
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    return float(np.mean(diff) / 255.0) 

def locate(image, frames):
    diffs = np.array([mean_abs_diff(image, f) for f in frames])
    order = np.argsort(diffs)               # frame indices, best first
    return int(order[0]), float(diffs[order[0]]), float(diffs[order[1]])

def build_sequence(name, clip):
    "Match every annotation of a clip; returns this clip's dict for the yaml."
    root = DATA_ROOT / DATASET_ROOT
    frames, fps = read_all_frames(root / clip["video"])
    matches = {}
    for n in clip["annotations"]:
        filename = f"frame{n}.png"
        img_path = root / IMAGES_DIR / filename
        img = cv2.imread(str(img_path))
       
        if img is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        
        idx, diff, second = locate(img, frames)
        print(f"{name:22s} {filename:13s} -> frame {idx:3d}  diff {diff:.4f}  (2nd best {second:.4f})")
        if diff > MAX_DIFF:
            raise RuntimeError(f"Frame {filename} in clip {name} does not match any frame in the video (best diff {diff:.4f})")
        if idx in matches: 
            raise RuntimeError(f"Frame {filename} in clip {name} matches the same video frame {idx} as {matches[idx]} (diff {diff:.4f})")
        
        matches[idx] = filename

    return {
        "video": clip["video"],
        "n_frames": len(frames),
        "fps": round(fps, 3),
        "source_start": clip["source_start"],
        "annotations": dict(sorted(matches.items())),
    }


def check(sequences):
    "Sanity check across the whole dataset."
    files = [f for seq in sequences.values() for f in seq["annotations"].values()]
    if len(files) != 123:
        raise ValueError(f"Expected 123 annotations, got {len(files)}")
    if len(set(files)) != len(files):
        raise ValueError("An annotation file is assigned to more than one clip")
    masks_dir = DATA_ROOT / DATASET_ROOT / MASKS_DIR
    missing = [f for f in files if not (masks_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"No mask for: {missing}")


def main():
    sequences = {name: build_sequence(name, clip) for name, clip in CLIPS.items()}
    check(sequences)
    manifest = {
        "name": "inhouse",
        "root": DATASET_ROOT,
        "images_dir": IMAGES_DIR,
        "masks_dir": MASKS_DIR,
        "sequences": sequences,
    }
    with open(OUT_PATH, "w") as f:
        f.write("# Generated by build_inhouse_manifest.py\n")
        yaml.safe_dump(manifest, f, sort_keys=False)
    

if __name__ == "__main__":
    main()