"""Train SALI with each frame paired with the previous one (causal).

SALI's released code segments frame i using frame i+1, although the paper
describes an online model that conditions on frame i-1 (see
configs/models/sali@sunseg-causal.yaml). This retrains SALI the way the paper
describes, with SALI's own recipe (configs/train/sali.yaml), not the lab's
shared one in scripts/train.py.

The model, loss, gradient clamp and augmentation are SALI's, imported from the
unchanged third_party/SALI. The encoder starts from ImageNet PVTv2-b5, as the
paper states and as our lab's models do (SALI's README links a trained image
segmentation model instead; see the model config). Only the training clips differ
(polypbench/train/sali_data.py), plus checkpoint selection: the recipe's
val_split decides where the best epoch is chosen, either the same held-out
validation videos as scripts/train.py ("heldout", configs/train/sali.yaml) or
TestHardDataset/Seen with all of TrainDataset used for training, as SALI's
train_video.py does (configs/train/sali_ft.yaml). Each validation video is run
frame by frame exactly as at test time (polypbench/models/sali.py).

Runs in the `sali` env (envs/sali.yml). Writes to
<artifacts>/checkpoints/<model_id>/ the same files as scripts/train.py:
    best.pth          state_dict of the best validation epoch (predict_sali.py loads this)
    last_state.pth    model + optimizer state, for --resume
    train_log.csv     per epoch: train loss, val Dice, max_skip, lr, time (+ test Dice with --test_every)
    run.json          recipe, model config, git commits, device, versions

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_sali.py --model_id sali@sunseg-causal
    CUDA_VISIBLE_DEVICES=0 python scripts/train_sali.py --model_id sali@sunseg-causal --resume
    python scripts/train_sali.py --model_id sali@sunseg-causal --smoke    # 20 steps, 1 val clip, to a scratch folder
    CUDA_VISIBLE_DEVICES=0 python scripts/train_sali.py --model_id sali@sunseg-causal-ft-s0 --test_every 1

The recipe defaults to the model config's train.recipe (else sali). A model_id
ending in -s<N> trains the same config with seed N. --test_every N also logs
Dice on every Nth clip of each SUN-SEG test split per epoch (whole clips, since
SALI runs them frame by frame); monitoring only, never used to pick the epoch.
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from polypbench.data.sunseg import load_sunseg_config, load_sunseg_split, load_sunseg_train_val
from polypbench.models import load_model_config
from polypbench.models.sali import build_sali, build_transform, import_sali, predict_sequence
from polypbench.paths import ARTIFACTS_ROOT, CONFIG_DIR, REPO_ROOT
from polypbench.train.models import load_backbone
from polypbench.train.sali_data import CausalClips

LOG_FIELDS = ["epoch", "train_loss", "val_dice", "max_skip", "lr", "epoch_minutes", "best"]


def git_commit(path):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pretrained(model, spec):
    """ImageNet weights into SALI's PVTv2 encoder, checked like our lab's models (every tensor must load)."""
    path = ARTIFACTS_ROOT / spec["pretrained_backbone"]
    if sha256(path) != spec["pretrained_sha256"]:
        raise RuntimeError(f"{path}: sha256 does not match pretrained_sha256 in the model config")
    load_backbone(model.backbone.feat_net.pvtv2_en, path)


def max_skip_at(recipe, epoch):
    schedule = recipe["max_skip_schedule"]
    return schedule[min((epoch - 1) // recipe["max_skip_stage_epochs"], len(schedule) - 1)]


def train_one_epoch(model, loader, optimizer, structure_loss, clip_gradient, recipe, max_steps=None, log_every=200):
    """One epoch of SALI's train(): loss on all four outputs, element-wise gradient clamp."""
    model.train()
    total, n = 0.0, 0
    for step, (images, gts, frame_idx) in enumerate(loader, start=1):
        optimizer.zero_grad()
        images = [x.cuda(non_blocking=True) for x in images]
        gt_mem, gt = gts[0].cuda(non_blocking=True), gts[1].cuda(non_blocking=True)
        preds = model(images, frame_idx, mode="train", gt_mem=gt_mem, use_mem=True, add_mem=False)
        loss = sum(structure_loss(p, gt) for p in preds)
        loss.backward()
        clip_gradient(optimizer, recipe["grad_clip"])
        optimizer.step()
        total += loss.item() * len(gt)
        n += len(gt)
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss {total / n:.4f}", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    return total / n


def validate(model, sequences, model_config, recipe):
    """Mean per-frame Dice (raw sigmoid >= threshold, at frame size) over the validation videos,
    each run as at test time. A frame with empty GT and empty prediction scores 1."""
    model.eval()
    transform = build_transform(model_config)
    dices = []
    for seq in sequences:
        for idx, prob in predict_sequence(model, seq, transform, recipe["mem_freq"], model_config["partner"]):
            pred = prob >= recipe["val_threshold"]
            gt = seq.load_mask(idx).astype(bool)
            denom = pred.sum() + gt.sum()
            dices.append(2 * (pred & gt).sum() / denom if denom > 0 else 1.0)
    model.mem_bank.clear_memory()
    return float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--recipe", help="configs/train/<recipe>.yaml (default: the model config's train.recipe, else sali)")
    parser.add_argument("--resume", action="store_true", help="continue from last_state.pth")
    parser.add_argument("--overwrite", action="store_true", help="start over even if checkpoints exist")
    parser.add_argument("--smoke", action="store_true", help="quick end-to-end check: 1 epoch of 20 steps, 1 val clip")
    parser.add_argument("--test_every", type=int, default=0,
                        help="also log Dice on every Nth clip of each test split per epoch (monitoring only; 0 = off)")
    args = parser.parse_args()

    model_config = load_model_config(args.model_id)
    spec = model_config["train"]
    with open(CONFIG_DIR / "train" / f"{args.recipe or spec.get('recipe', 'sali')}.yaml") as f:
        recipe = yaml.safe_load(f)
    if "seed" in model_config:   # a -s<N> seed run
        recipe["seed"] = model_config["seed"]
    if model_config["partner"] != "prev":
        raise SystemExit(f"{args.model_id}: train_sali.py builds causal clips, but the config says partner: "
                         f"{model_config['partner']}")

    out_dir = ARTIFACTS_ROOT / "checkpoints" / args.model_id
    if args.smoke:
        out_dir = ARTIFACTS_ROOT / "checkpoints" / "_smoke" / args.model_id
        recipe["epochs"] = 1
    state_path, best_path, log_path = out_dir / "last_state.pth", out_dir / "best.pth", out_dir / "train_log.csv"
    if best_path.exists() and not (args.resume or args.overwrite or args.smoke):
        raise SystemExit(f"{out_dir} already has a trained model; use --resume or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(recipe["seed"]); np.random.seed(recipe["seed"]); torch.manual_seed(recipe["seed"])

    import_sali(model_config)
    from train_video import structure_loss   # SALI's loss, unchanged
    from utils.utils import clip_gradient    # SALI's element-wise gradient clamp

    train_set, val_set = load_sunseg_train_val(recipe["val_split"])
    train_seqs, val_seqs = list(train_set), list(val_set)
    if args.smoke:
        val_seqs = val_seqs[:1]
    test_seqs = {}                    # monitoring only, never used to pick the best epoch
    if args.test_every:
        for split in load_sunseg_config()["test_splits"]:
            name = "test_" + split.replace("Test", "").replace("Dataset", "").replace("/", "_").lower()
            test_seqs[name] = list(load_sunseg_split(split, name))[::args.test_every]
            if args.smoke:
                test_seqs[name] = test_seqs[name][:1]
    log_fields = LOG_FIELDS + list(test_seqs)
    print(f"{args.model_id}: {sum(s.n_frames for s in train_seqs)} train frames in {len(train_seqs)} clips, "
          f"{sum(s.n_frames for s in val_seqs)} val frames in {len(val_seqs)} clips, {recipe['epochs']} epochs")

    model = build_sali(model_config)
    load_pretrained(model, spec)
    model = model.cuda()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt_cfg = recipe["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])

    start_epoch, best_dice = 1, -1.0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_dice = state["epoch"] + 1, state["best_dice"]
        print(f"Resumed after epoch {state['epoch']} (best val Dice {best_dice:.4f})")
    elif log_path.exists():
        log_path.unlink()

    run_info = {
        "model_id": args.model_id,
        "recipe": recipe,
        "model_config": model_config,
        "train_frames": sum(s.n_frames for s in train_seqs),
        "val_frames": sum(s.n_frames for s in val_seqs),
        "test_every": args.test_every,
        "polypbench_commit": git_commit(REPO_ROOT),
        "model_code_commit": git_commit(REPO_ROOT / model_config["code"]),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    loader, loader_skip = None, None
    for epoch in range(start_epoch, recipe["epochs"] + 1):
        max_skip = max_skip_at(recipe, epoch)
        if max_skip != loader_skip:   # redraw the memory frames, as SALI's reload every 5 epochs does
            stage_seed = recipe["seed"] * 1000 + (epoch - 1) // recipe["max_skip_stage_epochs"]
            loader = DataLoader(CausalClips(train_seqs, recipe["img_size"], max_skip, stage_seed),
                                batch_size=recipe["batch_size"], shuffle=True, num_workers=recipe["num_workers"],
                                pin_memory=True, drop_last=False)
            loader_skip = max_skip

        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{recipe['epochs']}  lr {lr:.2e}  max_skip {max_skip}  {len(loader)} steps", flush=True)
        train_loss = train_one_epoch(model, loader, optimizer, structure_loss, clip_gradient, recipe,
                                     max_steps=20 if args.smoke else None)
        val_dice = validate(model, val_seqs, model_config, recipe)
        test_dice = {name: validate(model, seqs, model_config, recipe) for name, seqs in test_seqs.items()}

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_dice": best_dice}, state_path)

        minutes = (time.time() - t0) / 60
        new_log = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            if new_log:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "train_loss": f"{train_loss:.5f}", "val_dice": f"{val_dice:.5f}",
                             "max_skip": max_skip, "lr": f"{lr:.3e}", "epoch_minutes": f"{minutes:.1f}",
                             "best": int(is_best), **{name: f"{d:.5f}" for name, d in test_dice.items()}})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}  val Dice {val_dice:.4f}"
              f"{'  (best)' if is_best else ''}  [{minutes:.1f} min]"
              + "".join(f"  {name} {d:.4f}" for name, d in test_dice.items()), flush=True)

    run_info.update(finished=datetime.now().isoformat(timespec="seconds"), best_val_dice=best_dice)
    with open(out_dir / "run.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Done. Best val Dice {best_dice:.4f}. "
          f"Next: python scripts/predict_sali.py --model_id {args.model_id} --dataset inhouse")


if __name__ == "__main__":
    main()
