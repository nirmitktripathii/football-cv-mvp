"""Evaluate the identity linker against SkillCorner tracking, where the true player ids are known.

SkillCorner's *detected* runs play the role of tracklets: each is a stretch where the broadcast camera
saw the player. The true id of every run is known, so a link between two runs is right or wrong.
Positions are degraded the way the video pipeline degrades them: independent per-frame noise, a
per-tracklet offset (the pitch-mapping error measured in docs/evaluation.md section 10) and a share of
wrong team labels. Windows of a fixed length stand in for camera shots.

Limits worth stating: only SkillCorner's own detections are used as truth (its extrapolated positions
are not), kits and appearance are not in the data so the appearance veto is *not* tested here, and
gaps here come from the camera leaving a player rather than from tracker identity switches.

Pre-registered criterion (fixed before the first run): wrong-link precision >= 0.98 for the default
settings in the reference scenario (offset sigma 3 m, 5 % team noise, seed 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football_cv.evaluation import skillcorner as sc
from football_cv.tracking import identity

PITCH_HALF_L, PITCH_HALF_W = 52.5, 34.0     # SkillCorner is centred; the linker's frame is not, but only differences matter
SAMPLE_FPS = 5.0                            # the pipeline's sampling rate
PRECISION_TARGET = 0.98
EDGE_MARGIN_M = 3.0                         # a tracklet end this close to the view boundary counts as "at the edge"
REFERENCE = {"offset_sigma_m": 3.0, "team_noise": 0.05}


def window_tracklets(table: pd.DataFrame, f0: int, f1: int, min_len_s: float, rng: np.random.Generator,
                     offset_sigma_m: float, noise_sigma_m: float, team_noise: float,
                     views: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Degraded tracklet rows for frames [f0, f1) and ``{track_id: true player_id}``."""
    sub = table[(table.frame >= f0) & (table.frame < f1) & (table.frame % int(sc.FPS / SAMPLE_FPS) == 0)]
    runs = sc.detected_runs(table[(table.frame >= f0) & (table.frame < f1)], min_len_s)
    rows, truth = [], {}
    for tid, run in enumerate(runs.itertuples(), start=1):
        g = sub[(sub.player_id == run.player_id) & (sub.frame >= run.f0) & (sub.frame <= run.f1) & sub.detected]
        if len(g) < 3:
            continue
        truth[tid] = int(run.player_id)
        offset = rng.normal(0.0, offset_sigma_m, 2)
        team = run.team
        if rng.random() < team_noise:
            team = "away" if team == "home" else "home"
        edge = (sc.inside_margin_m(views, g["frame"].to_numpy(), g["x"].to_numpy(float), g["y"].to_numpy(float))
                < EDGE_MARGIN_M) if views is not None else np.zeros(len(g), bool)
        xy = g[["x", "y"]].to_numpy(float) + offset + rng.normal(0.0, noise_sigma_m, (len(g), 2))
        rows.append(pd.DataFrame({
            "track_id": tid, "shot_id": 0, "timestamp_s": g["t_s"].to_numpy(float),
            "x_s": xy[:, 0] + PITCH_HALF_L, "y_s": xy[:, 1] + PITCH_HALF_W,
            "jump": False, "edge": edge, "team_id": team, "team_status": "confident",
        }))
    if not rows:
        return pd.DataFrame(), {}
    return pd.concat(rows, ignore_index=True), truth


def true_adjacent_pairs(trk: pd.DataFrame, truth: dict, max_gap_s: float) -> set[tuple[int, int]]:
    """(earlier, later) tracklet pairs of one player that follow each other within ``max_gap_s``."""
    pairs = set()
    t = trk.assign(player=trk["track_id"].map(truth)).sort_values("t0")
    for _, g in t.groupby("player"):
        for a, b in zip(g.itertuples(), g.iloc[1:].itertuples()):
            if 0 < b.t0 - a.t1 <= max_gap_s:
                pairs.add((int(a.track_id), int(b.track_id)))
    return pairs


def score_window(decisions: pd.DataFrame, identities: pd.DataFrame, trk: pd.DataFrame, truth: dict,
                 max_gap_s: float) -> dict:
    """Counts for one window; summed over windows before dividing."""
    linked = decisions[decisions.status == "linked"]
    right = [truth[a] == truth[b] for a, b in zip(linked.pred_track_id, linked.succ_track_id)]
    gold = true_adjacent_pairs(trk, truth, max_gap_s)
    got = {(int(a), int(b)) for a, b in zip(linked.pred_track_id, linked.succ_track_id)}
    ident_players = identities.assign(player=identities["track_id"].map(truth)).dropna(subset=["player"])
    impure = (ident_players.groupby("identity_id")["player"].nunique() > 1).sum()
    return {
        "links": len(linked), "links_right": int(sum(right)),
        "gold_pairs": len(gold), "gold_linked": len(gold & got),
        "ambiguous": int((decisions.status == "ambiguous").sum()) if len(decisions) else 0,
        "candidates": len(decisions), "identities": int(identities["identity_id"].nunique()),
        "impure_identities": int(impure),
    }


def greedy_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Naive baseline: every predecessor takes its cheapest gated successor, ambiguity ignored."""
    if decisions.empty:
        return decisions
    best = decisions.groupby("pred_track_id")["cost"].idxmin()
    out = decisions.copy()
    out["status"] = "ambiguous"
    out.loc[best, "status"] = "linked"
    return out


def run_scenario(table: pd.DataFrame, cfg: dict, offset_sigma_m: float, team_noise: float, seed: int = 0,
                 n_windows: int = 60, window_s: float = 30.0, min_len_s: float = 1.0,
                 noise_sigma_m: float = 1.0, greedy: bool = False, period: int | None = None,
                 views: pd.DataFrame | None = None) -> dict:
    """Total counts and the derived rates for one scenario over ``n_windows`` random windows."""
    rng = np.random.default_rng(seed)
    if period is not None:
        table = table[table.period == period]
    s = identity.settings(cfg)
    span = int(window_s * sc.FPS)
    starts = rng.integers(int(table.frame.min()), int(table.frame.max()) - span, n_windows)
    total: dict[str, int] = {}
    for f0 in starts:
        traj, truth = window_tracklets(table, int(f0), int(f0) + span, min_len_s, rng, offset_sigma_m,
                                       noise_sigma_m, team_noise, views)
        if traj.empty:
            continue
        identities, decisions, trk = identity.link_tracklets(traj, cfg)
        if greedy:
            decisions = greedy_decisions(decisions)
            identities = identity.assign_identities(identities["track_id"], decisions)
        for k, v in score_window(decisions, identities, trk, truth, s["max_gap_s"]).items():
            total[k] = total.get(k, 0) + v
    out = dict(total)
    out["precision"] = out["links_right"] / out["links"] if out.get("links") else float("nan")
    out["recall"] = out["gold_linked"] / out["gold_pairs"] if out.get("gold_pairs") else float("nan")
    return out


def split_track_test(traj: pd.DataFrame, cfg: dict | None, gap_s: float, min_track_s: float = 8.0) -> dict:
    """Self-supervised check on real tracks: cut each long track around its middle, drop ``gap_s``, relink.

    Every other track stays whole and acts as a rival. A link is right only if it rejoins the two halves
    of the same track. This tests the linker on the pipeline's own noise (including its calibration
    error), but with a fixed, *known* mapping error structure only in the sense of this one clip.
    """
    good = traj[traj["x_s"].notna() & ~traj["jump"].astype(bool)]
    spans = good.groupby("track_id")["timestamp_s"].agg(["min", "max"])
    long_ids = spans[(spans["max"] - spans["min"]) >= min_track_s].index
    parts, truth = [], {}
    next_id = int(good["track_id"].max()) + 1
    for tid, g in good.groupby("track_id"):
        if tid not in long_ids:
            parts.append(g.assign(track_id=tid))
            truth[tid] = tid
            continue
        mid = 0.5 * (spans.loc[tid, "min"] + spans.loc[tid, "max"])
        a = g[g["timestamp_s"] < mid - gap_s / 2].assign(track_id=tid)
        b = g[g["timestamp_s"] > mid + gap_s / 2].assign(track_id=next_id)
        truth[tid], truth[next_id] = tid, tid
        next_id += 1
        parts += [a, b]
    identities, decisions, trk = identity.link_tracklets(pd.concat(parts, ignore_index=True), cfg)
    linked = decisions[decisions.status == "linked"]
    right = sum(truth[a] == truth[b] for a, b in zip(linked.pred_track_id, linked.succ_track_id))
    return {"split_tracks": len(long_ids), "links": len(linked), "links_right": int(right),
            "precision": right / len(linked) if len(linked) else float("nan"),
            "recall": right / len(long_ids) if len(long_ids) else float("nan"),
            "ambiguous": int((decisions.status == "ambiguous").sum()) if len(decisions) else 0}
