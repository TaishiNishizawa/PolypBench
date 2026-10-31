"""CMSA-Net (third_party/CMSA-Net) inference, shared by scripts/predict_cmsa.py and the
validation in scripts/train_cmsa.py. Runs in the `sali` env (timm 0.5.4, as CMSA-Net pins).

Follows CMSA-Net's scripts/my_test.py (VPSTest_WithTestingTimeReferUpdate_2Ref), run
frame by frame instead of over a pre-built window list, with identical results:
  - every forward pass sees a 6-frame window [ref_sem, ref_conf, 4 most recent frames];
    the references start as frames 0 and 1;
  - the first window (frames 0-3) gives the predictions of frames 0-3; after that each
    window gives the prediction of its last (current) frame, as my_test.py keeps the
    first prediction of every frame;
  - after every window the Dynamic Multi-source Reference (DMR) rule may replace either
    reference with the previous frame (the window's second-to-last frame).
my_test.py imports CMSA-Net's argparse config on import, so the DMR helpers are copied
here rather than imported. Preprocessing calls CMSA-Net's own transforms, including its
Normalize_video (see the model config's note on what it actually normalizes).

Sequences shorter than 4 frames (none in SUN-SEG, in-house or REAL-Colon) get no
window in my_test.py; here the window is padded with copies of frame 0.
"""

import torch
import torch.nn.functional as F
from PIL import Image

from polypbench.paths import REPO_ROOT
from polypbench.train.models import import_repo_module


def import_cmsa(config):
    """Put CMSA-Net's code first on sys.path; returns its model module."""
    return import_repo_module(REPO_ROOT / config["code"], "lib.model_arch.cmsa_net")


def build_cmsa(config):
    """An untrained CMSANet with the config's backbone (encoder not initialized)."""
    return import_cmsa(config).CMSANet(backbone=config["backbone"])


def build_transform(config):
    """Test-time preprocessing: CMSA-Net's Resize_video + toTensor_video + Normalize_video
    (my_test.py's trsf_eval). uint8 HxWx3 RGB -> float 3xSxS tensor."""
    pre = import_repo_module(REPO_ROOT / config["code"], "lib.dataloader.preprocess")
    size = config["input_size"]
    ops = [pre.Resize_video(size, size), pre.toTensor_video(), pre.Normalize_video(config["mean"], config["std"])]

    def transform(rgb):
        imgs = [Image.fromarray(rgb)]
        for op in ops:
            imgs = op(imgs)
        return imgs[0]
    return transform


# ---- DMR helpers, copied from CMSA-Net's scripts/my_test.py ----

def _cos(a, b, eps=1e-6):
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)


def fg_bg_prototypes(x2_bchw, p_b1hw, eps=1e-6):
    """Probability-weighted foreground / background mean features, each [B, C]."""
    f = x2_bchw.flatten(2).transpose(1, 2)   # [B,HW,C]
    p = p_b1hw.flatten(2).transpose(1, 2)    # [B,HW,1]
    w_fg, w_bg = p, 1.0 - p
    mu_fg = (f * w_fg).sum(dim=1) / w_fg.sum(dim=1).clamp_min(eps)
    mu_bg = (f * w_bg).sum(dim=1) / w_bg.sum(dim=1).clamp_min(eps)
    return mu_fg, mu_bg


def pred_confidence(p_b1hw, eps=1e-6):
    """1 - mean binary entropy of the probability map, [B]."""
    p = p_b1hw.clamp(min=eps, max=1 - eps)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return 1.0 - ent.mean(dim=(1, 2, 3))


def update_references(st, images, x2_btchw, prob_full, dmr):
    """my_test.py's reference update for one window (B = 1). images: [1,T,3,S,S] with the
    current references in slots 0 and 1; updates st (ref_sem, ref_conf, cool_sem, cool_conf)."""
    _, T = images.shape[:2]
    C, H2, W2 = x2_btchw.shape[1:]
    H, W = prob_full.shape[-2:]
    ref_sem_idx, ref_conf_idx, prev_idx, cur_idx = 0, 1, T - 2, T - 1
    x2 = x2_btchw.view(1, T, C, H2, W2)
    prob_x2 = F.interpolate(prob_full.view(T, 1, H, W), size=(H2, W2), mode="bilinear",
                            align_corners=False).view(1, T, 1, H2, W2)

    mu_fg_sem, mu_bg_sem = fg_bg_prototypes(x2[:, ref_sem_idx], prob_x2[:, ref_sem_idx])
    mu_fg_conf, _ = fg_bg_prototypes(x2[:, ref_conf_idx], prob_x2[:, ref_conf_idx])
    mu_fg_prev, mu_bg_prev = fg_bg_prototypes(x2[:, prev_idx], prob_x2[:, prev_idx])
    mu_fg_cur, _ = fg_bg_prototypes(x2[:, cur_idx], prob_x2[:, cur_idx])

    s_sep_sem = 1.0 - _cos(mu_fg_sem, mu_bg_sem)        # semantic separability
    s_sep_prev = 1.0 - _cos(mu_fg_prev, mu_bg_prev)
    s_cons_sem = _cos(mu_fg_sem, mu_fg_cur)             # temporal reusability (-> current frame)
    s_cons_conf = _cos(mu_fg_conf, mu_fg_cur)
    s_cons_prev = _cos(mu_fg_prev, mu_fg_cur)
    s_conf_conf = pred_confidence(prob_x2[:, ref_conf_idx])
    s_conf_prev = pred_confidence(prob_x2[:, prev_idx])

    alpha, beta = dmr["alpha"], dmr["beta"]
    score_sem_ref = beta * s_sep_sem + (1 - beta) * s_cons_sem
    score_sem_prev = beta * s_sep_prev + (1 - beta) * s_cons_prev
    score_conf_ref = alpha * s_conf_conf + (1 - alpha) * s_cons_conf
    score_conf_prev = alpha * s_conf_prev + (1 - alpha) * s_cons_prev

    if st["cool_sem"] > 0:
        st["cool_sem"] -= 1
    if st["cool_conf"] > 0:
        st["cool_conf"] -= 1
    old_ref_conf = st["ref_conf"]
    updated_sem = updated_conf = False

    if st["cool_conf"] == 0 and score_conf_prev.item() > score_conf_ref.item():   # (A) ref_conf, fast
        st["ref_conf"] = images[0, prev_idx].detach().clone()
        st["cool_conf"] = dmr["cooldown_conf"]
        updated_conf = True
    if st["cool_sem"] == 0 and score_sem_prev.item() > score_sem_ref.item():       # (B) ref_sem, slow
        st["ref_sem"] = images[0, prev_idx].detach().clone()
        st["cool_sem"] = dmr["cooldown_sem"]
        updated_sem = True
    if updated_sem or updated_conf:                                                # (C) de-duplication
        mu_fg_sem_new = mu_fg_prev if updated_sem else mu_fg_sem
        mu_fg_conf_new = mu_fg_prev if updated_conf else mu_fg_conf
        if _cos(mu_fg_sem_new, mu_fg_conf_new).item() > dmr["tau_dup"]:
            st["ref_conf"] = old_ref_conf   # keep sem, roll back conf
            st["cool_conf"] = 0


@torch.no_grad()
def predict_sequence(model, seq, transform, config, full_res=True, return_logits=False):
    """Yield (index, probability map) for every frame of `seq`, in order.

    full_res: upsample the logits to the frame size before the sigmoid (for PNG
    predictions and Dice); False keeps CMSA-Net's own output size (352x352).
    return_logits: yield the logit map instead (before the sigmoid), for npz storage.
    """
    n_adj = config["adjacent_frames"]
    recent = []    # (index, transformed frame on GPU, frame HxW) of the last n_adj frames
    pending = []   # frames not yet predicted (only before the first window)
    st = None

    def run_window(window, emit):
        nonlocal st
        if st is None:
            st = {"ref_sem": window[0][1], "ref_conf": window[1][1], "cool_sem": 0, "cool_conf": 0}
        images = torch.stack([st["ref_sem"], st["ref_conf"]] + [x for _, x, _ in window]).unsqueeze(0)
        logits, x2, prob = model(images, mode="eval", return_feats=True)
        outs = []
        for (idx, _, hw), pos in zip(window[-emit:], range(images.shape[1] - emit, images.shape[1])):
            lg = logits[pos:pos + 1]
            if full_res:
                lg = F.interpolate(lg, size=hw, mode="bilinear", align_corners=False)
            outs.append((idx, (lg if return_logits else lg.sigmoid()).squeeze().float().cpu().numpy()))
        update_references(st, images, x2, prob, config["dmr"])
        return outs

    for idx, rgb in seq.frames():
        item = (idx, transform(rgb).cuda(), rgb.shape[:2])
        recent = (recent + [item])[-n_adj:]
        if st is None:
            pending.append(item)
            if len(pending) == n_adj:     # first window: frames 0..3, refs = frames 0 and 1
                yield from run_window(pending, emit=n_adj)
                pending = []
            continue
        yield from run_window(recent, emit=1)

    if pending:                           # sequence shorter than one window: pad with frame 0
        window = [pending[0]] * (n_adj - len(pending)) + pending
        st = {"ref_sem": pending[0][1], "ref_conf": pending[min(1, len(pending) - 1)][1],
              "cool_sem": 0, "cool_conf": 0}
        yield from run_window(window, emit=len(pending))
