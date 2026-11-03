"""Box-level (detection) metrics on REAL-Colon, following TRUE-Colon's definitions.

Inputs are the ground-truth tables (data/realcolon_gt.py) and a predicted-box table
(postproc/boxes.py). Everything at any operating threshold tau follows from two
scores per frame, so a whole curve costs one pass:

  alarm  highest score of any predicted box on the frame (-inf if none). A polyp-free
         frame is a false alarm at tau if alarm >= tau.
  hit    highest score of a predicted box overlapping any ground-truth box (IoU > 0,
         i.e. sharing a pixel). A positive frame is a hit at tau if hit >= tau.

and one per (lesion, boxed frame): the highest score of a box overlapping that lesion's
box, for lesion-level metrics. A box's score (`score`) is one of the columns written
by postproc/boxes.py: its component's peak logit, mean logit or mean probability;
boxes smaller than `min_area` (fraction of the map) are dropped first. Frames with
no box score -inf, so no tau ever counts them.

Phases, one per frame, derived here so their parameters never touch the ground truth:
  positive       has a ground-truth box
  outside        not inside the patient (video_info in-patient range)
  inspection     within a lesion's first..last boxed frame, without a box (annotation gaps)
  after_removal  within `after_removal_s` seconds after a lesion's last box (the lesion's
                 last sighting; removal itself is not annotated)
  clean          every other in-patient frame
Earlier labels win. Negatives are every frame that is not positive.

Lesion metrics are ours (latency in seconds at each video's fps; "within N s" = first hit
within N s) plus TRUE-Colon's versions as tc_* columns, for comparison: seconds at a nominal
30 fps, "within N s" = at least 15 hit frames in the first N s, persistence also at 25%, and
their two lesions left out of the mean latency (an early annotation gap).
"""

import numpy as np
import pandas as pd

PHASES = ["positive", "outside", "inspection", "after_removal", "clean"]
SCORES = ("peak", "mean_logit", "mean_prob")
NO_BOX = -np.inf
WITHIN_S = (1, 3, 5)
PERSISTENT_FRAC = 0.5
TC_FPS = 30.0
TC_MIN_FRAMES_IN_WINDOW = 15
TC_PERSISTENT_FRAC = 0.25
TC_LATENCY_EXCLUSIONS = frozenset({"003-014_1", "004-014_1"})   # true_colon/_factory.py


def mask_tau(score, threshold):
    """The tau at which every box of a mask binarized at `threshold` (a probability) counts:
    the threshold itself for mean_prob, its logit for the logit scores."""
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got '{score}'")
    return float(threshold) if score == "mean_prob" else float(np.log(threshold / (1 - threshold)))


def frame_table(gt, after_removal_s, videos=None):
    """One row per frame of the given videos (default: all): video, frame, in_patient, positive, phase."""
    rows = gt.videos if videos is None else gt.videos[gt.videos["video"].isin(videos)]
    boxed = gt.boxes.groupby("video")["frame"].unique()
    lesions = gt.lesions.groupby("video")
    tables = []
    for v in rows.itertuples():
        n = v.n_frames
        frames = np.arange(n)
        in_patient = (frames >= v.inside_first) & (frames <= v.inside_last)
        phase = np.full(n, PHASES.index("clean"), dtype=np.int8)
        if v.video in lesions.groups:
            after = int(round(after_removal_s * v.fps))
            for les in lesions.get_group(v.video).itertuples():   # later assignments win: after, then inspection
                phase[les.last_frame + 1:les.last_frame + 1 + after] = PHASES.index("after_removal")
            for les in lesions.get_group(v.video).itertuples():
                phase[les.first_frame:les.last_frame + 1] = PHASES.index("inspection")
        phase[~in_patient] = PHASES.index("outside")
        positive = np.zeros(n, dtype=bool)
        if v.video in boxed.index:
            positive[boxed[v.video]] = True
        phase[positive] = PHASES.index("positive")
        tables.append(pd.DataFrame({"video": v.video, "frame": frames, "in_patient": in_patient,
                                    "positive": positive, "phase": np.array(PHASES)[phase]}))
    return pd.concat(tables, ignore_index=True)


def score_boxes(pred, gt_boxes, score="peak", min_area=0.0):
    """(frame scores: video, frame, alarm, hit; lesion scores: unique_id, video, frame, hit).
    Only frames / lesion frames with a box appear; join onto frame_table / the ground truth with 0 fill."""
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got '{score}'")
    p = pred.loc[pred["area"] >= min_area, ["video", "frame", "xmin", "ymin", "xmax", "ymax", score]]
    p = p.rename(columns={score: "score"})
    alarm = p.groupby(["video", "frame"], as_index=False)["score"].max().rename(columns={"score": "alarm"})
    pairs = p.merge(gt_boxes[["video", "frame", "unique_id", "xmin", "ymin", "xmax", "ymax"]],
                    on=["video", "frame"], suffixes=("", "_gt"))
    overlap = ((pairs["xmin"] <= pairs["xmax_gt"]) & (pairs["xmin_gt"] <= pairs["xmax"])
               & (pairs["ymin"] <= pairs["ymax_gt"]) & (pairs["ymin_gt"] <= pairs["ymax"]))
    pairs = pairs[overlap]
    lesion = pairs.groupby(["unique_id", "video", "frame"], as_index=False)["score"].max().rename(
        columns={"score": "hit"})
    hit = lesion.groupby(["video", "frame"], as_index=False)["hit"].max()
    frames = alarm.merge(hit, on=["video", "frame"], how="left").fillna({"hit": NO_BOX})
    return frames, lesion


def scored_frames(frames, frame_scores):
    """frame_table rows with alarm / hit columns (NO_BOX where no box)."""
    out = frames.merge(frame_scores, on=["video", "frame"], how="left")
    return out.fillna({"alarm": NO_BOX, "hit": NO_BOX})


def scored_lesion_frames(gt_boxes, lesion_scores):
    """One row per (lesion, boxed frame) with its hit score (NO_BOX where no overlapping box)."""
    lf = gt_boxes[["unique_id", "video", "frame"]].drop_duplicates()
    return lf.merge(lesion_scores, on=["unique_id", "video", "frame"], how="left").fillna({"hit": NO_BOX})


def _rate_at(scores, taus):
    """Fraction of scores >= each tau."""
    s = np.sort(np.asarray(scores, dtype=float))
    return (len(s) - np.searchsorted(s, taus, side="left")) / max(len(s), 1)


def frame_rates(scored, taus, lesion_frames=None):
    """Per tau: frame TPR (positive frames hit), frame FPR (negative frames with an alarm), and,
    if lesion_frames is given, box sensitivity (lesion boxes hit; differs from TPR on multi-box frames)."""
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    pos, neg = scored[scored["positive"]], scored[~scored["positive"]]
    out = pd.DataFrame({"tau": taus, "tpr": _rate_at(pos["hit"], taus), "fpr": _rate_at(neg["alarm"], taus),
                        "n_pos": len(pos), "n_neg": len(neg)})
    if lesion_frames is not None:
        out["box_sensitivity"] = _rate_at(lesion_frames["hit"], taus)
    return out


def curve_taus(scored, n=200):
    """Thresholds for a curve: n quantiles each of the finite negative alarms and positive hits,
    so the curve is dense where either rate changes."""
    taus = []
    for col, rows in (("alarm", ~scored["positive"]), ("hit", scored["positive"])):
        s = scored.loc[rows, col].to_numpy()
        s = s[np.isfinite(s)]
        if len(s):
            taus.append(np.quantile(s, np.linspace(0, 1, n)))
    return np.unique(np.concatenate(taus)) if taus else np.array([])


def alarm_rates_by_phase(scored, tau):
    """Per phase: n frames and the fraction with an alarm at tau (for 'positive', any box at all)."""
    g = scored.assign(alarm_at=scored["alarm"] >= tau).groupby("phase")
    return pd.DataFrame({"n": g.size(), "alarm_rate": g["alarm_at"].mean()}).reindex(
        [p for p in PHASES if p in g.groups]).reset_index()


def threshold_for_fpr(scored, target):
    """Smallest tau (among the scores that occur) whose frame FPR is <= target; just above the
    highest alarm (FPR 0) if none is."""
    alarms = scored.loc[~scored["positive"], "alarm"].to_numpy()
    candidates = np.unique(alarms[np.isfinite(alarms)])
    if len(candidates) == 0:            # no alarms at all: every box can be shown
        lowest = scored["alarm"].to_numpy()
        lowest = lowest[np.isfinite(lowest)]
        return float(lowest.min()) if len(lowest) else 0.0
    fpr = _rate_at(alarms, candidates)
    ok = np.flatnonzero(fpr <= target)
    if len(ok):
        return float(candidates[ok[0]])
    return float(np.nextafter(candidates[-1], np.inf))   # just above the highest alarm: FPR 0


def lesion_metrics(lesion_frames, lesions, videos, tau):
    """One row per lesion at tau: detected, hit fraction of its boxed frames, persistent,
    frames / seconds from its first boxed frame to its first hit, detected within 1 / 3 / 5 s;
    and TRUE-Colon's versions (tc_*, see the module docstring)."""
    lf = lesion_frames.assign(hit_at=lesion_frames["hit"] >= tau)
    g = lf.groupby("unique_id")
    first_hit = lf[lf["hit_at"]].groupby("unique_id")["frame"].min()
    out = lesions.set_index("unique_id")[["video", "first_frame", "n_boxed_frames", "latency_excluded",
                                          "size_mm", "site", "histology_class", "is_polyp"]].copy()
    out["n_hit_frames"] = g["hit_at"].sum().reindex(out.index).fillna(0).astype(int)
    out["hit_frac"] = out["n_hit_frames"] / out["n_boxed_frames"]
    out["detected"] = out["n_hit_frames"] > 0
    out["persistent"] = out["hit_frac"] >= PERSISTENT_FRAC
    out["latency_frames"] = first_hit.reindex(out.index) - out["first_frame"]
    fps = videos.set_index("video")["fps"]
    out["latency_s"] = out["latency_frames"] / out["video"].map(fps)
    for s in WITHIN_S:
        out[f"within_{s}s"] = out["latency_s"] <= s     # NaN (never detected) compares False
    out["tc_persistent"] = out["hit_frac"] >= TC_PERSISTENT_FRAC
    out["tc_latency_s"] = out["latency_frames"] / TC_FPS
    out["tc_latency_excluded"] = out.index.isin(TC_LATENCY_EXCLUSIONS)
    hits = lf.loc[lf["hit_at"], ["unique_id", "frame"]]
    since_first = hits["frame"].to_numpy() - out["first_frame"].reindex(hits["unique_id"]).to_numpy()
    for s in WITHIN_S:
        n = hits[since_first < int(s * TC_FPS)].groupby("unique_id").size()
        out[f"tc_within_{s}s"] = n.reindex(out.index).fillna(0) >= TC_MIN_FRAMES_IN_WINDOW
    return out.reset_index().assign(tau=tau)


def lesion_summary(m):
    """One row from lesion_metrics rows (one tau): n lesions; fractions detected / persistent /
    within N s; mean hit fraction; latency over detected lesions, ours (mean / median seconds,
    without latency_excluded) and TRUE-Colon's (mean frames, without their two lesions)."""
    det = m[m["detected"]]
    ours, tc = det[~det["latency_excluded"]], det[~det["tc_latency_excluded"]]
    row = {"n_lesions": len(m), "detected": m["detected"].mean(), "persistent": m["persistent"].mean(),
           "tc_persistent": m["tc_persistent"].mean(), "mean_hit_frac": m["hit_frac"].mean(),
           "mean_latency_s": ours["latency_s"].mean(), "median_latency_s": ours["latency_s"].median(),
           "tc_mean_latency_frames": tc["latency_frames"].mean()}
    for s in WITHIN_S:
        row[f"within_{s}s"] = m[f"within_{s}s"].mean()
        row[f"tc_within_{s}s"] = m[f"tc_within_{s}s"].mean()
    return row


def evaluate_boxes(pred, gt, frames, score, min_area, mask_threshold, target_fpr, val_taus=None,
                   curve_points=200):
    """Every metric for one box table, in two frame scopes ("all", "in_patient") at up to three
    thresholds: "mask" (every box: mask_tau), "val_fpr" (val_taus[scope] if given, else chosen
    here for target_fpr, i.e. in-sample on val) and, when val_taus is given, "test_fpr" (chosen
    here, for comparison only). Returns a dict of DataFrames: summary (frame rates + lesion
    summary per scope x threshold), lesion, phase, curve; and "chosen" ({scope: tau for
    target_fpr on these frames}, to save from val)."""
    frame_scores, lesion_scores = score_boxes(pred, gt.boxes, score, min_area)
    scored = scored_frames(frames, frame_scores)
    lf = scored_lesion_frames(gt.boxes, lesion_scores)
    summary, lesion, phase, curve, chosen = [], [], [], [], {}
    for scope, sub in (("all", scored), ("in_patient", scored[scored["in_patient"]])):
        chosen[scope] = threshold_for_fpr(sub, target_fpr)
        taus = {"mask": mask_tau(score, mask_threshold)}
        if val_taus is None:
            taus["val_fpr"] = chosen[scope]
        else:
            taus["val_fpr"], taus["test_fpr"] = val_taus[scope], chosen[scope]
        for kind, tau in taus.items():
            keys = {"scope": scope, "threshold": kind}
            m = lesion_metrics(lf, gt.lesions, gt.videos, tau)
            rates = frame_rates(sub, [tau], lf).iloc[0].to_dict()
            summary.append({**keys, **rates, **lesion_summary(m)})
            lesion.append(m.assign(**keys))
            phase.append(alarm_rates_by_phase(sub, tau).assign(**keys, tau=tau))
        curve.append(frame_rates(sub, curve_taus(sub, curve_points), lf).assign(scope=scope))
    return {"summary": pd.DataFrame(summary), "lesion": pd.concat(lesion, ignore_index=True),
            "phase": pd.concat(phase, ignore_index=True), "curve": pd.concat(curve, ignore_index=True),
            "chosen": chosen}
