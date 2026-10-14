"""Causal training clips for SALI (scripts/train_sali.py). Runs in the `sali` env.

Mirrors VideoDataset in SALI's dataloaders/video_list.py, and reuses its
augmentation functions, except that the partner frame is the PREVIOUS frame
rather than the next one. The sample for frame i of a clip is

    images [memory, frame i, partner]   masks [memory mask, frame i mask]
    i = 0:   [0, 0, 0]                        [all zeros, 0]
    i >= 1:  [m, i, i-1]                      [m, i]      m drawn from [max(i - max_skip, 0), i - 1]

Frame 0's memory mask is all zeros: at test time frame 0 reads its own
features with no mask (use_mem=False), and its own ground truth would hand the
model the answer. As in SALI, the memory frames are drawn once when the
dataset is built (scripts/train_sali.py rebuilds it whenever max_skip changes).

Frames are in playback order (polypbench.data.sunseg.frame_order). SALI sorts
them alphabetically, which misorders 8 of the 112 TrainDataset clips.
"""

import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def rgb_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def binary_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("L")


class CausalClips(Dataset):
    """Needs SALI's code on sys.path (polypbench.models.sali.import_sali)."""

    def __init__(self, sequences, img_size, max_skip, seed):
        from dataloaders.video_list import colorEnhance, cv_random_flip, randomPeper, randomRotation
        self.flip, self.rotate, self.color, self.pepper = cv_random_flip, randomRotation, colorEnhance, randomPeper

        rng = random.Random(seed)
        self.samples = []   # (memory frame, frame, partner frame, memory mask or None, frame mask, frame index)
        for seq in sequences:
            frames = seq.frame_paths
            masks = [seq.annotations[i] for i in range(seq.n_frames)]
            self.samples.append((frames[0], frames[0], frames[0], None, masks[0], 0))
            for i in range(1, seq.n_frames):
                m = rng.randint(max(i - max_skip, 0), i - 1)
                self.samples.append((frames[m], frames[i], frames[i - 1], masks[m], masks[i], i))

        # SALI's VideoDataset transforms
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mem, cur, partner, mem_mask, mask, frame_idx = self.samples[index]
        imgs = [rgb_loader(p) for p in (mem, cur, partner)]
        gts = [binary_loader(mem_mask if mem_mask is not None else mask), binary_loader(mask)]

        # same augmentation, in the same order, as SALI's VideoDataset.__getitem__
        imgs, gts = self.flip(imgs, gts)
        imgs, gts = self.rotate(imgs, gts)
        imgs = self.color(imgs)
        imgs = [self.img_transform(img) for img in imgs]
        gts = [self.gt_transform(self.pepper(gt)) for gt in gts]

        if mem_mask is None:
            gts[0] = torch.zeros_like(gts[1])
        return imgs, gts, frame_idx
