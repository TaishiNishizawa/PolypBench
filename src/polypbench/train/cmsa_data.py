"""Training clips for CMSA-Net (scripts/train_cmsa.py). Runs in the `sali` env.

Mirrors CMSAVideoDataset in CMSA-Net's lib/dataloader/dataloader.py (ttype "train")
and reuses its augmentation classes, in the same order (get_cmsa_video_dataset's
trsf_main). For each clip, one sample per start frame b = 1 .. n - 5:

    frames [0, 1, b, b+1, b+2, b+3]      masks for all six (the loss uses only the last)

Frame 0 and 1 are the references, b+3 is the current frame. The only difference
is the frame order: playback order (polypbench.data.sunseg.frame_order), as for
every other model, where CMSA-Net's training sort puts the original video id
first. The two differ only in case35_4 and case63_1, which splice segments of
two recordings, so the segment order (not the order within a segment) differs.
CMSA-Net's dataloader module imports its argparse config, so the dataset is
rebuilt here; only lib/dataloader/preprocess.py is imported.
"""

import torch
from PIL import Image
from torch.utils.data import Dataset

from polypbench.paths import REPO_ROOT
from polypbench.train.models import import_repo_module


class CMSAClips(Dataset):
    def __init__(self, sequences, model_config, recipe):
        pre = import_repo_module(REPO_ROOT / model_config["code"], "lib.dataloader.preprocess")
        size, n_adj = recipe["img_size"], recipe["adjacent_frames"]
        self.transform = pre.Compose_imglabel([
            pre.Resize_video(size, size),
            pre.Random_crop_Resize_Video(20),
            pre.Random_horizontal_flip_video(0.5),
            pre.Random_rotation(0.5, angle=25),
            pre.toTensor_video(),
            pre.Normalize_video(model_config["mean"], model_config["std"]),
        ])

        self.samples = []   # list of 6 (frame path, mask path)
        for seq in sequences:
            pairs = [(seq.frame_paths[i], seq.annotations[i]) for i in range(seq.n_frames)]
            for begin in range(1, len(pairs) - (n_adj - 1) - 1):
                self.samples.append([pairs[0], pairs[1]] + pairs[begin:begin + n_adj])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        imgs, labels = [], []
        for frame, mask in self.samples[index]:
            with open(frame, "rb") as f:
                imgs.append(Image.open(f).convert("RGB"))
            with open(mask, "rb") as f:
                labels.append(Image.open(f).convert("L"))
        imgs, labels = self.transform(imgs, labels)
        return torch.stack(imgs), torch.stack(labels)   # [6,3,S,S] float, [6,1,S,S] long in {0, 1}
