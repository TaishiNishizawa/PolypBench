"""Box-level (detection) metrics on REAL-Colon, following TRUE-Colon's definitions.

Inputs are the ground-truth tables (data/realcolon_gt.py) and a predicted-box table
(postproc/boxes.py). Everything at any operating threshold tau follows from two
scores per frame, so a whole curve costs one pass:

  alarm  highest score of any predicted box on the frame (0 if none). A polyp-free
         frame is a false alarm at tau if alarm >= tau.
  hit    highest score of a predicted box overlapping any ground-truth box (IoU > 0,
         i.e. sharing a pixel). A positive frame is a hit at tau if hit >= tau.

and one per (lesion, boxed frame): the highest score of a box overlapping that lesion's
box, for lesion-level metrics. A box's score is its component's peak or mean
probability (`score`); boxes smaller than `min_area` (fraction of the map) are
dropped first. Frames with no score have none: 0, and tau is always > 0.

Phases, one per frame, derived here so their parameters never touch the ground truth:
  positive       has a ground-truth box
  outside        not inside the patient (video_info in-patient range)
  inspection     within a lesion's first..last boxed frame, without a box (annotation gaps)
  after_removal  within `after_removal_s` seconds after a lesion's last box (the lesion's
                 last sighting; removal itself is not annotated)
  clean          every other in-patient frame
Earlier labels win. Negatives are every frame that is not positive.
"""

import numpy as np
import pandas as pd

PHASES = ["positive", "outside", "inspection", "after_removal", "clean"]
SCORES = ("peak", "mean")
WITHIN_S = (1, 3, 5)
PERSISTENT_FRAC = 0.5


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
    frames = alarm.merge(hit, on=["video", "frame"], how="left").fillna({"hit": 0.0})
    return frames, lesion


def scored_frames(frames, frame_scores):
    """frame_table rows with alarm / hit columns (0 where no box)."""
    out = frames.merge(frame_scores, on=["video", "frame"], how="left")
    return out.fillna({"alarm": 0.0, "hit": 0.0})


def scored_lesion_frames(gt_boxes, lesion_scores):
    """One row per (lesion, boxed frame) with its hit score (0 where no overlapping box)."""
    lf = gt_boxes[["unique_id", "video", "frame"]].drop_duplicates()
    return lf.merge(lesion_scores, on=["unique_id", "video", "frame"], how="left").fillna({"hit": 0.0})


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


def alarm_rates_by_phase(scored, tau):
    """Per phase: n frames and the fraction with an alarm at tau (for 'positive', any box at all)."""
    g = scored.assign(alarm_at=scored["alarm"] >= tau).groupby("phase")
    return pd.DataFrame({"n": g.size(), "alarm_rate": g["alarm_at"].mean()}).reindex(
        [p for p in PHASES if p in g.groups]).reset_index()


def threshold_for_fpr(scored, target):
    """Smallest tau (among the scores that occur) whose frame FPR is <= target; inf if none is."""
    alarms = scored.loc[~scored["positive"], "alarm"].to_numpy()
    candidates = np.unique(alarms[alarms > 0])
    if len(candidates) == 0:
        return float(np.nextafter(0, 1))
    fpr = _rate_at(alarms, candidates)
    ok = np.flatnonzero(fpr <= target)
    if len(ok):
        return float(candidates[ok[0]])
    return float(np.nextafter(candidates[-1], np.inf))   # just above the highest alarm: FPR 0


def lesion_metrics(lesion_frames, lesions, videos, tau):
    """One row per lesion at tau: detected, hit fraction of its boxed frames, persistent,
    frames / seconds from its first boxed frame to its first hit, detected within 1 / 3 / 5 s."""
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
    return out.reset_index().assign(tau=tau)
