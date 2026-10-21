"""Training and validation steps, ported from train_polyp.py of EMCAD / MK-UNet / SACNet."""

import torch
import torch.nn.functional as F

from polypbench.train.models import as_list, final_prediction


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU, with extra weight near mask boundaries (PraNet)."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def supervised_loss(outputs, mask, supervision, final_output):
    outputs = as_list(outputs)
    if supervision == "heads_and_sum":      # EMCAD / SACNet: every head, plus their sum
        return sum(structure_loss(o, mask) for o in outputs) + structure_loss(sum(outputs), mask)
    if supervision == "heads":              # CASCADE: every head (their sum is the prediction, but not supervised)
        return sum(structure_loss(o, mask) for o in outputs)
    if supervision == "final_only":         # MK-UNet
        return structure_loss(outputs[final_output], mask)
    raise ValueError(f"Unknown supervision '{supervision}'")


def train_one_epoch(model, loader, optimizer, recipe, spec, device, log_every=200):
    """One epoch; every batch is trained at each multiscale rate. Returns mean loss at rate 1."""
    model.train()
    total, n = 0.0, 0
    for step, (images, masks) in enumerate(loader, start=1):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        for rate in recipe["multiscale"]:
            x, y = images, masks
            if rate != 1:
                size = int(round(recipe["img_size"] * rate / 32) * 32)
                x = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=True)
                y = F.interpolate(masks, size=(size, size), mode="nearest")
            optimizer.zero_grad()
            loss = supervised_loss(model(x), y, spec["supervision"], spec["final_output"])
            loss.backward()
            for group in optimizer.param_groups:          # clip_gradient(): clamp each element
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.data.clamp_(-recipe["grad_clip"], recipe["grad_clip"])
            optimizer.step()
            if rate == 1:
                total += loss.item() * len(images)
                n += len(images)
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss {total / n:.4f}", flush=True)
    return total / n


@torch.no_grad()
def validate(model, loader, final_output, threshold, device):
    """Mean per-frame Dice of the final output (raw sigmoid >= threshold) at training resolution."""
    model.eval()
    dices = []
    for images, masks in loader:
        pred = torch.sigmoid(final_prediction(model(images.to(device)), final_output)) >= threshold
        gt = masks.to(device) > 0.5
        inter = (pred & gt).flatten(1).sum(1).float()
        denom = (pred.flatten(1).sum(1) + gt.flatten(1).sum(1)).float()
        dices.append(torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom)))
    return torch.cat(dices).mean().item()
