"""Run SALI over a dataset and save a probability map for every frame.

Runs in the `sali` env (envs/sali.yml), not the main one:

    conda activate sali
    python scripts/predict_sali.py --dataset inhouse

Inference follows SALI's own test_video.py exactly:
  - frame i is segmented from the pair [frame i, frame i+1]; the last frame
    uses [last, previous];
  - the long-term memory bank is cleared at the start of each sequence, and
    written on the first frame and every `mem_freq`-th frame after it;
  - preprocessing: resize to 352x352 (PIL bilinear), ImageNet normalization;
  - the final output is upsampled to the frame size, then passed through a sigmoid.

The one deliberate difference: SALI's script min-max rescales every frame to
[0, 1], which forces confident pixels even on frames with no polyp. Here the
raw sigmoid is saved (postproc "none").

Predictions use the same layout as predict.py, so evaluate.py and
visualize.py work unchanged.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from polypbench.data import load_dataset
from polypbench.io import predictions
from polypbench.models import load_model_config
from polypbench.paths import ARTIFACTS_ROOT, REPO_ROOT

MODEL_ID = "sali@sunseg"
POSTPROC = "none"


def load_sali(config):
    sali_root = REPO_ROOT / config["code"]
    sys.path.insert(0, str(sali_root))
    import lib.DCN_Module as dcn
    if dcn.modulated_deform_conv is None:   # SALI only prints a warning if the CUDA op is missing
        raise RuntimeError(f"SALI's deformable-conv extension isn't built; see envs/sali.yml")
    from lib import VideoModel

    opt = SimpleNamespace(trainsize=config["input_size"], testsize=config["input_size"],
                          pretrained_weights=None, mem_freq=config["mem_freq"])
    model = VideoModel(opt)
    state = torch.load(ARTIFACTS_ROOT / config["weights"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def with_partner(frames):
    """Yield (index, frame, partner): partner is the next frame, or the previous one for the last frame."""
    prev = None
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
def run_sequence(model, seq, transform, mem_freq, dataset_name):
    model.mem_bank.clear_memory()
    for idx, rgb, partner in with_partner(seq.frames()):
        images = [transform(Image.fromarray(img)).unsqueeze(0).cuda() for img in (rgb, partner)]
        if idx == 0:
            out = model(images, [idx], mode="val", use_mem=False, add_mem=True)
        elif idx % mem_freq == 0:
            out = model(images, [idx], mode="val", use_mem=True, add_mem=True)
        else:
            out = model(images, [idx], mode="val", use_mem=True, add_mem=False)
        logits = F.interpolate(out[-1], size=rgb.shape[:2], mode="bilinear", align_corners=False)
        prob = logits.sigmoid().squeeze().cpu().numpy()
        predictions.write_prediction(predictions.frame_path(dataset_name, MODEL_ID, POSTPROC, seq.name, idx), prob)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if predictions.is_complete(dataset.name, MODEL_ID, POSTPROC) and not args.overwrite:
        print(f"[skip] {MODEL_ID} / {POSTPROC}: already complete (use --overwrite)")
        return

    config = load_model_config(MODEL_ID)
    model = load_sali(config)
    transform = T.Compose([
        T.Resize((config["input_size"], config["input_size"])),
        T.ToTensor(),
        T.Normalize(config["mean"], config["std"]),
    ])

    started = datetime.now().isoformat(timespec="seconds")
    for seq in dataset:
        t0 = time.time()
        run_sequence(model, seq, transform, config["mem_freq"], dataset.name)
        dt = time.time() - t0
        print(f"[{MODEL_ID}] {seq.name}: {seq.n_frames} frames in {dt:.1f}s ({seq.n_frames / dt:.1f} fps)")

    sali_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT / config["code"], text=True).strip()
    predictions.write_run_info(dataset.name, MODEL_ID, POSTPROC, {
        "dataset": dataset.name,
        "model_id": MODEL_ID,
        "postproc": POSTPROC,
        "model_config": config,
        "sequences": {seq.name: seq.n_frames for seq in dataset},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "sali_commit": sali_commit,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
