"""SALI (third_party/SALI) inference, shared by scripts/predict_sali.py and the
validation in scripts/train_sali.py. Runs in the `sali` env.

Follows SALI's own test_video.py, except for the partner frame, which is set
by `partner:` in the model config:
  - "next": frame i is segmented from [frame i, frame i+1], the last frame from
    [last, previous]. This is what SALI's released code does (not causal).
  - "prev": frame i is segmented from [frame i, frame i-1], frame 0 from
    [frame 0, frame 0]. Causal, as the SALI paper describes (Sec. 2.1).
The long-term memory bank is cleared at the start of each sequence, and written
on the first frame and every `mem_freq`-th frame after it.
"""

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from polypbench.paths import REPO_ROOT

PARTNERS = ("next", "prev")


def import_sali(config):
    """Put SALI's code on sys.path and check its deformable-conv extension is built."""
    sali_root = REPO_ROOT / config["code"]
    if str(sali_root) not in sys.path:
        sys.path.insert(0, str(sali_root))
    import lib.DCN_Module as dcn
    if dcn.modulated_deform_conv is None:   # SALI only prints a warning if the CUDA op is missing
        raise RuntimeError("SALI's deformable-conv extension isn't built; see envs/sali.yml")
    return sali_root


def build_sali(config):
    """An untrained VideoModel (no weights loaded)."""
    import_sali(config)
    from lib import VideoModel
    opt = SimpleNamespace(trainsize=config["input_size"], testsize=config["input_size"],
                          pretrained_weights=None, mem_freq=config["mem_freq"])
    return VideoModel(opt)


def build_transform(config):
    """Test-time preprocessing: resize (PIL bilinear), ImageNet normalization."""
    return T.Compose([
        T.Resize((config["input_size"], config["input_size"])),
        T.ToTensor(),
        T.Normalize(config["mean"], config["std"]),
    ])


def with_partner(frames, partner):
    """Yield (index, frame, partner_frame) for every frame of a sequence, in order."""
    if partner not in PARTNERS:
        raise ValueError(f"partner must be one of {PARTNERS}, got '{partner}'")
    prev = None
    if partner == "prev":
        for idx, cur in frames:
            yield idx, cur, prev if prev is not None else cur
            prev = cur
        return
    it = iter(frames)
    try:
        idx, cur = next(it)
    except StopIteration:
        return
    for nxt_idx, nxt in it:
        yield idx, cur, nxt
        prev, (idx, cur) = cur, (nxt_idx, nxt)
    yield idx, cur, prev if prev is not None else cur


@torch.no_grad()
def predict_sequence(model, seq, transform, mem_freq, partner):
    """Yield (index, probability map at frame size) for every frame of `seq`."""
    model.mem_bank.clear_memory()
    for idx, rgb, other in with_partner(seq.frames(), partner):
        images = [transform(Image.fromarray(img)).unsqueeze(0).cuda() for img in (rgb, other)]
        if idx == 0:
            out = model(images, [idx], mode="val", use_mem=False, add_mem=True)
        elif idx % mem_freq == 0:
            out = model(images, [idx], mode="val", use_mem=True, add_mem=True)
        else:
            out = model(images, [idx], mode="val", use_mem=True, add_mem=False)
        logits = F.interpolate(out[-1], size=rgb.shape[:2], mode="bilinear", align_corners=False)
        yield idx, logits.sigmoid().squeeze().cpu().numpy()
