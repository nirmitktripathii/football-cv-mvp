"""Candidate attacking sequences: windows where the observed ball moves into a final third.

A candidate is found from observations only:

1. Take the calibrated, observed ball positions of one motion segment and split
   them where the ball is unobserved for more than ``MAX_BALL_GAP_S``.
2. Keep a window if the ball spends time in a final third (x < 35 or x > 70)
   and the window lasts at least ``MIN_DURATION_S``.
3. Name the attacking direction by the goal whose third the ball reaches, and
   the *probable* attacking team from ``home_attacks``. Ball proximity (which
   team's player is nearest the ball) is reported beside it as a check, not
   as possession.

The ranking score is the attacking team's peak zone xT in the window
(exploratory, see ``analytics.xt``). These are CANDIDATES for a human to
review: there is no event detection, no possession model and no ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_BALL_GAP_S = 2.0
MIN_DURATION_S = 3.0
LEFT_THIRD_X, RIGHT_THIRD_X = 35.0, 70.0
PAD_S = 2.0  # context shown before and after the ball observations


def _windows(ball: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    for _, g in ball.groupby("segment_id"):
        g = g.sort_values("timestamp_s")
        split = (g["timestamp_s"].diff() > MAX_BALL_GAP_S).cumsum()
        out.extend(w for _, w in g.groupby(split))
    return out


def _proximity_share(prox: pd.DataFrame, start: float, end: float, team: str) -> tuple[float, int]:
    w = prox[(prox["timestamp_s"] >= start) & (prox["timestamp_s"] <= end) & prox["nearest_team"].isin(["home", "away"])]
    return (round(float((w["nearest_team"] == team).mean()), 2) if len(w) else np.nan), len(w)


def find_candidates(
    ball_xt: pd.DataFrame, prox: pd.DataFrame, calib: pd.DataFrame, home_attacks: str, top_n: int = 5,
) -> pd.DataFrame:
    """Ranked candidate windows with their evidence and a confidence note."""
    cal = calib.set_index("sample_id")
    rows = []
    for w in _windows(ball_xt):
        dur = float(w["timestamp_s"].iloc[-1] - w["timestamp_s"].iloc[0])
        in_left, in_right = (w["x_s"] < LEFT_THIRD_X).mean(), (w["x_s"] > RIGHT_THIRD_X).mean()
        if dur < MIN_DURATION_S or max(in_left, in_right) == 0:
            continue
        direction = "right" if in_right >= in_left else "left"
        team = "home" if direction == home_attacks else "away"
        xt = w[f"xt_{team}_attacking"]
        share, n_prox = _proximity_share(prox, w["timestamp_s"].iloc[0], w["timestamp_s"].iloc[-1], team)
        kdt = cal.loc[cal.index.intersection(w["sample_id"]), "keyframe_dt_s"].abs()
        span = cal[(cal["timestamp_s"] >= w["timestamp_s"].iloc[0]) & (cal["timestamp_s"] <= w["timestamp_s"].iloc[-1])]
        rows.append({
            "segment_id": int(w["segment_id"].iloc[0]),
            "start_s": round(float(w["timestamp_s"].iloc[0]) - PAD_S, 1), "end_s": round(float(w["timestamp_s"].iloc[-1]) + PAD_S, 1),
            "ball_first_s": float(w["timestamp_s"].iloc[0]), "ball_last_s": float(w["timestamp_s"].iloc[-1]),
            "first_sample": int(w["sample_id"].iloc[0]), "last_sample": int(w["sample_id"].iloc[-1]),
            "attacking_direction": direction, "probable_attacking_team": team,
            "ball_obs": len(w), "ball_coverage": round(len(w) / max(len(span), 1), 2),
            "share_in_final_third": round(float(max(in_left, in_right)), 2),
            "ball_x_start": round(float(w["x_s"].iloc[0]), 1), "ball_x_end": round(float(w["x_s"].iloc[-1]), 1),
            "peak_xt": round(float(xt.max()), 4), "net_xt": round(float(xt.iloc[-1] - xt.iloc[0]), 4),
            "nearest_player_share_attacking_team": share, "proximity_obs": n_prox,
            "max_keyframe_dt_s": round(float(kdt.max()), 1) if len(kdt) else np.nan,
        })
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    cand = cand.sort_values("peak_xt", ascending=False).head(top_n).sort_values("start_s").reset_index(drop=True)
    cand.insert(0, "seq_id", [f"SEQ{i + 1}" for i in range(len(cand))])
    cand["confidence"] = cand.apply(_confidence, axis=1)
    return cand


def _confidence(r: pd.Series) -> str:
    """Plain-language caveats for one candidate; 'medium' is the best a CV-only candidate gets."""
    notes = []
    if r["ball_coverage"] < 0.5:
        notes.append(f"ball seen in only {r['ball_coverage']:.0%} of calibrated frames")
    if np.isfinite(r["max_keyframe_dt_s"]) and r["max_keyframe_dt_s"] > 6:
        notes.append(f"up to {r['max_keyframe_dt_s']:.0f} s from a manual keyframe (propagation drift)")
    share = r["nearest_player_share_attacking_team"]
    if not np.isfinite(share) or share < 0.4:
        notes.append("nearest-player proxy does not support the attacking-team attribution")
    level = "medium" if not notes else "low"
    return f"{level}: " + ("; ".join(notes) if notes else "ball well observed near keyframes; attribution by direction of play")
