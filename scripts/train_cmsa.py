"""Train CMSA-Net (PVTv2-B2 variant) on SUN-SEG with its own recipe (configs/train/cmsa_ft.yaml).

The model and augmentation are CMSA-Net's, imported from third_party/CMSA-Net
(model code unchanged except the `backbone` option that re-enables its
commented-out PVTv2-B2 encoder). The loss is a copy of VPS_Seg_Loss from its
scripts/my_train.py (which can't be imported: it parses CMSA-Net's argparse
config and sets CUDA_VISIBLE_DEVICES on import), applied, as there, to the
current (last) frame of each clip on all three outputs. The encoder starts from
ImageNet PVTv2-B2, as the paper states. Hyperparameters follow the paper where
it gives them and the code otherwise (see the recipe).

Checkpoint selection: the recipe's val_split (TestHardDataset/Seen for
cmsa_ft, as CMSA-Net's my_train.py does). Each validation clip is run frame by
frame exactly as at test time, with the dynamic reference update
(polypbench/models/cmsa.py); my_train.py instead scores the current frame of
each window with references fixed to frames 0 and 1.

Runs in the `sali` env (envs/sali.yml). Writes to <artifacts>/checkpoints/<model_id>/
the same files as scripts/train_sali.py:
    best.pth          state_dict of the best validation epoch (predict_cmsa.py loads this)
    last_state.pth    model + optimizer + scheduler state, for --resume
    train_log.csv     per epoch: train loss, val Dice, lr, time (+ test Dice with --test_every)
    run.json          recipe, model config, git commits, device, versions

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_cmsa.py --model_id cmsa-net@sunseg-ft-s0 --test_every 4
    CUDA_VISIBLE_DEVICES=0 python scripts/train_cmsa.py --model_id cmsa-net@sunseg-ft-s0 --resume
    python scripts/train_cmsa.py --model_id cmsa-net@sunseg-ft-s0 --smoke    # 20 steps, 1 val clip, to a scratch folder

The recipe defaults to the model config's train.recipe. A model_id ending in
-s<N> trains the same config with seed N. --test_every N also logs Dice on
every Nth clip of each SUN-SEG test split per epoch (whole clips, run as at
test time); monitoring only, never used to pick the epoch.
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from polypbench.data.sunseg import load_sunseg_config, load_sunseg_split, load_sunseg_train_val
from polypbench.models import load_model_config
from polypbench.models.cmsa import build_cmsa, build_transform, predict_sequence
from polypbench.paths import ARTIFACTS_ROOT, CONFIG_DIR, REPO_ROOT
from polypbench.train.cmsa_data import CMSAClips
from polypbench.train.models import load_backbone

LOG_FIELDS = ["epoch", "train_loss", "val_dice", "lr", "epoch_minutes", "best"]


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


class VPSSegLoss(nn.Module):
    """VPS_Seg_Loss from CMSA-Net's scripts/my_train.py, unchanged: structure loss (weighted BCE +
    weighted IoU) + Dice loss. Two quirks kept: reduce="none" is the legacy reduce=True, so the
    "weighted" BCE is a plain mean; dice_loss applies a sigmoid to the target too."""

    def dice_loss(self, input, target):
        input = torch.sigmoid(input)
        target = torch.sigmoid(target)
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        return 1 - loss.sum() / N

    def structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
        wbce = (weit * wbce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(-2, -1))
        union = ((pred + mask) * weit).sum(dim=(-2, -1))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

    def forward(self, pred, target):
        return (self.structure_loss(pred.squeeze(), target.squeeze().float())
                + self.dice_loss(pred.squeeze(), target.squeeze().float()))


def load_pretrained(model, spec):
    """ImageNet weights into the PVTv2-B2 encoder; every tensor must load."""
    path = ARTIFACTS_ROOT / spec["pretrained_backbone"]
    if sha256(path) != spec["pretrained_sha256"]:
        raise RuntimeError(f"{path}: sha256 does not match pretrained_sha256 in the model config")
    load_backbone(model.feature_extractor, path)


def train_one_epoch(model, loader, optimizer, loss_func, max_steps=None, log_every=200):
    """One epoch of my_train.py's train(): loss on the current frame of all three outputs."""
    model.train()
    total, n = 0.0, 0
    for step, (images, gts) in enumerate(loader, start=1):
        optimizer.zero_grad()
        images, gts = images.cuda(non_blocking=True), gts.cuda(non_blocking=True)
        B, T = gts.shape[:2]
        gt_cur = gts[:, T - 1]
        loss = 0.0
        for pred in model(images, mask_on=True, sdpm_on=True, ddfe_on=True, mode="train"):
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            pred_cur = pred.view(B, T, 1, *pred.shape[-2:])[:, T - 1]
            if pred_cur.shape[-2:] != gt_cur.shape[-2:]:
                pred_cur = F.interpolate(pred_cur, size=gt_cur.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + loss_func(pred_cur.contiguous(), gt_cur.contiguous())
        loss.backward()
        optimizer.step()
        total += loss.item() * B
        n += B
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss {total / n:.4f}", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    return total / n


def validate(model, sequences, model_config, recipe):
    """Mean per-frame Dice (raw sigmoid >= threshold, at frame size) over the validation clips,
    each run as at test time. A frame with empty GT and empty prediction scores 1."""
    model.eval()
    transform = build_transform(model_config)
    dices = []
    for seq in sequences:
        for idx, prob in predict_sequence(model, seq, transform, model_config):
            pred = prob >= recipe["val_threshold"]
            gt = seq.load_mask(idx).astype(bool)
            denom = pred.sum() + gt.sum()
            dices.append(2 * (pred & gt).sum() / denom if denom > 0 else 1.0)
    return float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--recipe", help="configs/train/<recipe>.yaml (default: the model config's train.recipe)")
    parser.add_argument("--resume", action="store_true", help="continue from last_state.pth")
    parser.add_argument("--overwrite", action="store_true", help="start over even if checkpoints exist")
    parser.add_argument("--smoke", action="store_true", help="quick end-to-end check: 1 epoch of 20 steps, 1 val clip")
    parser.add_argument("--test_every", type=int, default=0,
                        help="also log Dice on every Nth clip of each test split per epoch (monitoring only; 0 = off)")
    args = parser.parse_args()

    model_config = load_model_config(args.model_id)
    if model_config["arch"] != "cmsa-net":
        raise SystemExit(f"{args.model_id} is not a CMSA-Net model (arch {model_config['arch']})")
    spec = model_config["train"]
    with open(CONFIG_DIR / "train" / f"{args.recipe or spec['recipe']}.yaml") as f:
        recipe = yaml.safe_load(f)
    if "seed" in model_config:   # a -s<N> seed run
        recipe["seed"] = model_config["seed"]
    if recipe["adjacent_frames"] != model_config["adjacent_frames"]:
        raise SystemExit("adjacent_frames differs between the recipe and the model config")

    out_dir = ARTIFACTS_ROOT / "checkpoints" / args.model_id
    if args.smoke:
        out_dir = ARTIFACTS_ROOT / "checkpoints" / "_smoke" / args.model_id
        recipe["epochs"] = 1
    state_path, best_path, log_path = out_dir / "last_state.pth", out_dir / "best.pth", out_dir / "train_log.csv"
    if best_path.exists() and not (args.resume or args.overwrite or args.smoke):
        raise SystemExit(f"{out_dir} already has a trained model; use --resume or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(recipe["seed"]); np.random.seed(recipe["seed"]); torch.manual_seed(recipe["seed"])

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

    train_data = CMSAClips(train_seqs, model_config, recipe)
    loader = DataLoader(train_data, batch_size=recipe["batch_size"], shuffle=True, num_workers=recipe["num_workers"],
                        pin_memory=True, drop_last=False, persistent_workers=True)
    print(f"{args.model_id}: {sum(s.n_frames for s in train_seqs)} train frames in {len(train_seqs)} clips "
          f"({len(train_data)} training windows), {sum(s.n_frames for s in val_seqs)} val frames in "
          f"{len(val_seqs)} clips, {recipe['epochs']} epochs")

    model = build_cmsa(model_config)
    load_pretrained(model, spec)
    model = model.cuda()
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    opt_cfg, sched_cfg = recipe["optimizer"], recipe["scheduler"]
    base_params = [p for name, p in model.named_parameters() if "feature_extractor" in name]
    head_params = [p for name, p in model.named_parameters() if "feature_extractor" not in name]
    optimizer = torch.optim.AdamW([{"params": base_params, "lr": opt_cfg["backbone_lr_scale"] * opt_cfg["lr"]},
                                   {"params": head_params, "lr": opt_cfg["lr"]}],
                                  weight_decay=opt_cfg["weight_decay"], amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched_cfg["t_max"],
                                                           eta_min=sched_cfg["eta_min"])
    loss_func = VPSSegLoss()

    start_epoch, best_dice = 1, -1.0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_dice = state["epoch"] + 1, state["best_dice"]
        print(f"Resumed after epoch {state['epoch']} (best val Dice {best_dice:.4f})")
    elif log_path.exists():
        log_path.unlink()

    run_info = {
        "model_id": args.model_id,
        "recipe": recipe,
        "model_config": model_config,
        "train_frames": sum(s.n_frames for s in train_seqs),
        "train_windows": len(train_data),
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

    for epoch in range(start_epoch, recipe["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[1]["lr"]   # head lr (the encoder's is in param_groups[0])
        print(f"Epoch {epoch}/{recipe['epochs']}  lr {lr:.2e} (encoder {optimizer.param_groups[0]['lr']:.2e})  "
              f"{len(loader)} steps", flush=True)
        train_loss = train_one_epoch(model, loader, optimizer, loss_func, max_steps=20 if args.smoke else None)
        val_dice = validate(model, val_seqs, model_config, recipe)
        test_dice = {name: validate(model, seqs, model_config, recipe) for name, seqs in test_seqs.items()}
        scheduler.step()

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_dice": best_dice}, state_path)

        minutes = (time.time() - t0) / 60
        new_log = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            if new_log:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "train_loss": f"{train_loss:.5f}", "val_dice": f"{val_dice:.5f}",
                             "lr": f"{lr:.3e}", "epoch_minutes": f"{minutes:.1f}", "best": int(is_best),
                             **{name: f"{d:.5f}" for name, d in test_dice.items()}})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}  val Dice {val_dice:.4f}"
              f"{'  (best)' if is_best else ''}  [{minutes:.1f} min]"
              + "".join(f"  {name} {d:.4f}" for name, d in test_dice.items()), flush=True)

    run_info.update(finished=datetime.now().isoformat(timespec="seconds"), best_val_dice=best_dice)
    with open(out_dir / "run.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Done. Best val Dice {best_dice:.4f}. "
          f"Next: python scripts/predict_cmsa.py --model_id {args.model_id} --dataset sunseg_hard_unseen")


if __name__ == "__main__":
    main()
