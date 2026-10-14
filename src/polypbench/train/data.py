"""Frame-level SUN-SEG data for training image segmentation models.

Each item is one (frame, mask) pair; clip membership only decides train vs
validation (see configs/datasets/sunseg.yaml). Augmentation and mask handling
follow utils/dataloader_polyp.py of EMCAD / MK-UNet / SACNet.
"""

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from polypbench.data.sunseg import load_sunseg_config
from polypbench.paths import DATA_ROOT


def sunseg_pairs(split, clips=None, exclude=()):
    """Sorted (frame_path, mask_path) pairs of a SUN-SEG split, optionally restricted to/excluding clips."""
    cfg = load_sunseg_config()
    root = DATA_ROOT / cfg["root"] / split
    frame_root, mask_root = root / cfg["frames_dir"], root / cfg["masks_dir"]
    pairs = []
    for clip_dir in sorted(frame_root.iterdir()):
        if (clips is not None and clip_dir.name not in clips) or clip_dir.name in exclude:
            continue
        for frame in sorted(clip_dir.glob("*.jpg")):
            mask = mask_root / clip_dir.name / f"{frame.stem}.png"
            if not mask.exists():
                raise FileNotFoundError(f"No mask for {frame}")
            pairs.append((frame, mask))
    if not pairs:
        raise RuntimeError(f"No frames found for {split} (clips={clips}, exclude={exclude})")
    return pairs


def train_val_pairs():
    cfg = load_sunseg_config()
    val_clips = set(cfg["val_clips"])
    train = sunseg_pairs(cfg["train_split"], exclude=val_clips)
    val = sunseg_pairs(cfg["train_split"], clips=val_clips)
    return train, val


def build_transform(recipe, train):
    size = recipe["img_size"]
    ops = []
    if train:
        aug = recipe["augmentation"]
        ops += [
            A.Rotate(limit=aug["rotate_limit"], p=aug["rotate_p"]),
            A.VerticalFlip(p=aug["vflip_p"]),
            A.HorizontalFlip(p=aug["hflip_p"]),
        ]
    ops += [A.Resize(height=size, width=size), A.Normalize(mean=recipe["mean"], std=recipe["std"]), ToTensorV2()]
    return A.Compose(ops)


class SegmentationFrames(Dataset):
    """(image, mask) tensors: image 3xSxS normalized float, mask 1xSxS float in {0, 1}."""

    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        frame_path, mask_path = self.pairs[i]
        image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        out = self.transform(image=image, mask=mask)
        mask = torch.as_tensor(out["mask"])
        # binarize as the original loader does: 0/255 masks (with interpolation noise) -> > 20
        foreground = (mask > 20) if mask.max() > 127 else (mask >= 1)
        return out["image"], foreground.float().unsqueeze(0)
