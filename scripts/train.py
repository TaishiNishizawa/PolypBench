"""Train one of our lab's models with a shared recipe.

The recipe (configs/train/<recipe>.yaml) is identical for every model; the
model's own choices (architecture, supervised outputs, pretrained backbone)
come from the `train:` section of configs/models/<model_id>.yaml, which may
also name its recipe (`recipe:`; --recipe overrides, default sunseg). The
recipe's val_split decides where the best epoch is chosen: held-out validation
videos ("heldout"), or a SUN-SEG test split with all of TrainDataset used for
training (see configs/datasets/sunseg.yaml). A model_id ending in -s<N> trains
the same config with seed N (polypbench.models.load_model_config).

Runs in the `train` env (envs/train.yml). Writes to
<artifacts>/checkpoints/<model_id>/:
    best.pth          state_dict of the best validation epoch (export this)
    last_state.pth    model + optimizer + scheduler state, for --resume
    train_log.csv     per epoch: train loss, val Dice, lr, time (+ test Dice with --test_every)
    run.json          recipe, model config, git commits, device, versions

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model_id emcad-b2@sunseg
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model_id emcad-b2@sunseg --resume
    python scripts/train.py --model_id mkunet@sunseg --smoke     # 20 steps, 1 epoch, to a scratch folder
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model_id emcad-b2@sunseg-ft-s0 --test_every 1

--test_every N also logs Dice on every Nth frame of each SUN-SEG test split per
epoch. That is for monitoring only (e.g. how much picking the epoch on a test
split would flatter the result); the best epoch is still chosen on val_split.
"""

import argparse
import csv
import json
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from polypbench.data.sunseg import load_sunseg_config
from polypbench.models import load_model_config
from polypbench.paths import ARTIFACTS_ROOT, CONFIG_DIR, REPO_ROOT
from polypbench.train.data import SegmentationFrames, build_transform, sunseg_pairs, train_val_pairs
from polypbench.train.loop import train_one_epoch, validate
from polypbench.train.models import build_model

LOG_FIELDS = ["epoch", "train_loss", "val_dice", "lr", "epoch_minutes", "best"]


def git_commit(path):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--recipe", help="configs/train/<recipe>.yaml (default: the model config's train.recipe, else sunseg)")
    parser.add_argument("--resume", action="store_true", help="continue from last_state.pth")
    parser.add_argument("--overwrite", action="store_true", help="start over even if checkpoints exist")
    parser.add_argument("--smoke", action="store_true", help="quick end-to-end check: 1 epoch of 20 steps")
    parser.add_argument("--test_every", type=int, default=0,
                        help="also log Dice on every Nth frame of each test split per epoch (monitoring only; 0 = off)")
    args = parser.parse_args()

    model_config = load_model_config(args.model_id)
    spec = model_config["train"]
    with open(CONFIG_DIR / "train" / f"{args.recipe or spec.get('recipe', 'sunseg')}.yaml") as f:
        recipe = yaml.safe_load(f)
    if "seed" in model_config:   # a -s<N> seed run
        recipe["seed"] = model_config["seed"]

    out_dir = ARTIFACTS_ROOT / "checkpoints" / args.model_id
    if args.smoke:
        out_dir = ARTIFACTS_ROOT / "checkpoints" / "_smoke" / args.model_id
        recipe["epochs"] = 1
    state_path, best_path, log_path = out_dir / "last_state.pth", out_dir / "best.pth", out_dir / "train_log.csv"
    if best_path.exists() and not (args.resume or args.overwrite or args.smoke):
        raise SystemExit(f"{out_dir} already has a trained model; use --resume or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(recipe["seed"]); np.random.seed(recipe["seed"]); torch.manual_seed(recipe["seed"])
    device = torch.device("cuda")

    train_pairs, val_pairs = train_val_pairs(recipe["val_split"])
    if args.smoke:
        train_pairs = random.Random(0).sample(train_pairs, 20 * recipe["batch_size"])
        val_pairs = val_pairs[::50]
    train_loader = DataLoader(SegmentationFrames(train_pairs, build_transform(recipe, train=True)),
                              batch_size=recipe["batch_size"], shuffle=True, num_workers=recipe["num_workers"],
                              pin_memory=True, drop_last=False, persistent_workers=True)
    val_loader = DataLoader(SegmentationFrames(val_pairs, build_transform(recipe, train=False)),
                            batch_size=16, shuffle=False, num_workers=recipe["num_workers"], pin_memory=True)
    test_loaders = {}                 # monitoring only, never used to pick the best epoch
    if args.test_every:
        for split in load_sunseg_config()["test_splits"]:
            name = "test_" + split.replace("Test", "").replace("Dataset", "").replace("/", "_").lower()
            pairs = sunseg_pairs(split)[::args.test_every * (50 if args.smoke else 1)]
            test_loaders[name] = DataLoader(SegmentationFrames(pairs, build_transform(recipe, train=False)),
                                            batch_size=16, shuffle=False, num_workers=recipe["num_workers"],
                                            pin_memory=True)
    log_fields = LOG_FIELDS + list(test_loaders)
    print(f"{args.model_id}: {len(train_pairs)} train frames, {len(val_pairs)} val frames, "
          f"{len(train_loader)} steps/epoch, {recipe['epochs']} epochs")

    model = build_model(model_config).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt_cfg = recipe["optimizer"]
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}   # PraNet: Adam, no weight decay
    if opt_cfg["name"] not in optimizers:
        raise SystemExit(f"optimizer.name must be one of {list(optimizers)}, got '{opt_cfg['name']}'")
    optimizer = optimizers[opt_cfg["name"]](model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"],
                                                           eta_min=recipe["scheduler"]["eta_min"])

    start_epoch, best_dice = 1, -1.0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
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
        "train_frames": len(train_pairs),
        "val_frames": len(val_pairs),
        "test_every": args.test_every,
        "polypbench_commit": git_commit(REPO_ROOT),
        "model_code_commit": git_commit(REPO_ROOT / spec["code"]),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    for epoch in range(start_epoch, recipe["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{recipe['epochs']}  lr {lr:.2e}", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, recipe, spec, device)
        val_dice = validate(model, val_loader, spec["final_output"], recipe["val_threshold"], device)
        test_dice = {name: validate(model, loader, spec["final_output"], recipe["val_threshold"], device)
                     for name, loader in test_loaders.items()}
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
    print(f"Done. Best val Dice {best_dice:.4f}. Next: python scripts/export_onnx.py --model_id {args.model_id}")


if __name__ == "__main__":
    main()
