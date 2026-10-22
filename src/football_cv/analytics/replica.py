"""Tracking-data export in the style of abdullahtarek/football_analysis, on this project's calibrated pitch metres.

Facts about the reference design this follows (written from its description, not copied from its code): a detector plus
tracker gives players and a ball, team colour clusters give a team per player, the ball goes to the nearest player foot
within a fixed radius, speed and distance come from positions in metres, and possession is the share of frames each team
holds the ball (the previous team is kept while nobody is close).

What differs here, on purpose:

* Positions are pitch metres from the per-frame homography (``calibration.projection``), not four hard-coded pixel
  corners. Only frames whose calibration status is in ``MEASURED`` give coordinates; nothing is invented elsewhere.
* Speed is displacement over a centred ~1 s window of observed points (a 0.2 s step at 5 fps is dominated by position
  noise), reported in km/h with a plausibility cap and a flag, and is NaN where the track is not observed around ``t``.
* Distance is the path length of lightly smoothed observed positions; time with no observation is not counted and is
  reported per track (``unmeasured_s``), so a player the camera lost is not credited with distance.
* The ball is the Viterbi path of ``tracking.ball_path`` with bounded gap filling. Its metre position is the ground-plane
  projection of the ball centre, which is biased while the ball is airborne.
* Possession keeps both a strict figure (frames with a holder within ``possession_radius_m``) and the reference-style
  carry-over figure, and reports the share of carried frames.

There is no ground truth for pitch-metre speed or possession on this footage; the output is a measurement, not a score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_cv.calibration import projection
from football_cv.calibration.motion import G_of, apply_h
from football_cv.tracking import ball_path as bp
from football_cv.tracking.tracker import BALL_DEFAULTS, static_ball_mask
from football_cv.trajectories.clean import MAX_GAP_S

MEASURED = ("keyframe", "propagated")
PLAYER_ROLES = ("player", "goalkeeper")
TEAMS = ("home", "away")
KMH = 3.6


@dataclass(frozen=True)
class ReplicaParams:
    speed_window_s: float = 1.0       # centred window used for speed
    window_tol_s: float = 0.25        # window ends must be observed within this of the target time
    max_speed_kmh: float = 36.0       # faster is flagged "implausible" and left out of the speed column
    smooth_samples: int = 5           # centred rolling mean (observed samples) before summing distance
    possession_radius_m: float = 3.0  # nearest player must be this close to hold the ball
    min_players_for_possession: int = 1


def _usable(traj: pd.DataFrame) -> pd.DataFrame:
    """Observed, calibrated, non-jump person rows with finite smoothed pitch positions."""
    ok = (
        traj["role"].isin(PLAYER_ROLES + ("referee",))
        & traj["calibration_status"].isin(MEASURED)
        & ~traj["jump"].fillna(False).astype(bool)
        & traj["x_s"].notna() & traj["y_s"].notna()
    )
    return traj[ok]


def _nearest_index(t: np.ndarray, target: float) -> int:
    i = int(np.searchsorted(t, target))
    cands = [j for j in (i - 1, i) if 0 <= j < len(t)]
    return min(cands, key=lambda j: abs(t[j] - target))


def track_kinematics(g: pd.DataFrame, p: ReplicaParams) -> pd.DataFrame:
    """Speed (km/h), flag and cumulative distance (m) for the observed rows of one track, in time order."""
    g = g.sort_values("timestamp_s")
    t = g["timestamp_s"].to_numpy(dtype=float)
    x, y = g["x_s"].to_numpy(dtype=float), g["y_s"].to_numpy(dtype=float)
    n = len(g)
    run = np.concatenate([[0], np.cumsum(np.diff(t) > MAX_GAP_S)]) if n else np.zeros(0, int)

    speed = np.full(n, np.nan)
    flag = np.array(["no_window"] * n, dtype=object)
    half = p.speed_window_s / 2
    for i in range(n):
        lo, hi = _nearest_index(t, t[i] - half), _nearest_index(t, t[i] + half)
        if abs(t[lo] - (t[i] - half)) > p.window_tol_s or abs(t[hi] - (t[i] + half)) > p.window_tol_s or hi <= lo:
            continue
        v = np.hypot(x[hi] - x[lo], y[hi] - y[lo]) / (t[hi] - t[lo]) * KMH
        if v > p.max_speed_kmh:
            flag[i] = "implausible"
        else:
            speed[i], flag[i] = v, "ok"

    xs = pd.Series(x).groupby(run).transform(lambda s: s.rolling(p.smooth_samples, center=True, min_periods=1).mean())
    ys = pd.Series(y).groupby(run).transform(lambda s: s.rolling(p.smooth_samples, center=True, min_periods=1).mean())
    step = np.zeros(n)
    same_run = run[1:] == run[:-1] if n > 1 else np.zeros(0, bool)
    step[1:] = np.where(same_run, np.hypot(np.diff(xs.to_numpy()), np.diff(ys.to_numpy())), 0.0)
    return g.assign(speed_kmh=speed, speed_flag=flag, run=run, cum_dist_m=np.cumsum(step))


def player_table(traj: pd.DataFrame, p: ReplicaParams = ReplicaParams()) -> pd.DataFrame:
    """One row per observed person per sample, with speed and cumulative distance; the export's main table."""
    use = _usable(traj)
    parts = [track_kinematics(g, p) for _, g in use.groupby("track_id", sort=True)]
    cols = ["sample_id", "frame_id", "timestamp_s", "track_id", "role", "team_id", "confidence",
            "x_m", "y_m", "x_s", "y_s", "speed_kmh", "speed_flag", "cum_dist_m", "calibration_status"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols].sort_values(["timestamp_s", "track_id"], ignore_index=True)


def ball_table(dets: pd.DataFrame, motion: pd.DataFrame, calib: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Ball path (detected + short interpolated gaps) in image pixels and, where calibrated, pitch metres."""
    prm = {**BALL_DEFAULTS, **(params or {})}
    d = dets[(dets["object_class"] == "ball") & dets["on_pitch"] & dets["sample_id"].isin(motion["sample_id"])].copy()
    d["cx"], d["cy"] = (d["bbox_x1"] + d["bbox_x2"]) / 2, (d["bbox_y1"] + d["bbox_y2"]) / 2
    d["segment_id"] = d["sample_id"].map(motion.set_index("sample_id")["segment_id"])
    d = d.reset_index(drop=True)
    G = {r.sample_id: G_of(r) for r in motion.itertuples()}
    xy = np.zeros((len(d), 2))
    for s, idx in d.groupby("sample_id").groups.items():
        xy[idx] = apply_h(G[s], d.loc[idx, ["cx", "cy"]].to_numpy(dtype=float))
    keep = ~static_ball_mask(d, xy, prm)
    cands = bp.candidates_from_detections(d[keep], xy[keep], motion.set_index("sample_id")["segment_id"].to_dict())
    track = bp.ball_track(cands, motion)
    cal = calib.set_index("sample_id")
    rows = []
    for r in track.itertuples():
        img = apply_h(np.linalg.inv(G[r.sample_id]), np.array([[r.x, r.y]]))[0]
        status = cal.at[r.sample_id, "calibration_status"] if r.sample_id in cal.index else "no_calibration"
        pitch = (np.nan, np.nan)
        if status in MEASURED:
            pitch = tuple(apply_h(projection.H_of(cal.loc[r.sample_id]), img.reshape(1, 2))[0])
        rows.append({"sample_id": r.sample_id, "timestamp_s": r.timestamp_s, "ball_status": r.status,
                     "confidence": r.confidence, "img_x": img[0], "img_y": img[1],
                     "x_m": pitch[0], "y_m": pitch[1], "calibration_status": status})
    return pd.DataFrame(rows, columns=["sample_id", "timestamp_s", "ball_status", "confidence", "img_x", "img_y",
                                       "x_m", "y_m", "calibration_status"])


def possession_table(players: pd.DataFrame, ball: pd.DataFrame, samples: pd.DataFrame,
                     p: ReplicaParams = ReplicaParams()) -> pd.DataFrame:
    """Per sample: nearest team player to the ball, the holder (within the radius) and the carried-over team."""
    pl = players[players["role"].isin(PLAYER_ROLES) & players["team_id"].isin(TEAMS)]
    by_sample = {s: g for s, g in pl.groupby("sample_id")}
    b = ball.set_index("sample_id")
    rows = []
    for rec in samples.itertuples():
        row = {"sample_id": rec.sample_id, "timestamp_s": rec.timestamp_s, "ball_available": False,
               "nearest_track": np.nan, "nearest_team": None, "nearest_dist_m": np.nan, "holder_team": None}
        if rec.sample_id in b.index and np.isfinite(b.at[rec.sample_id, "x_m"]) and rec.sample_id in by_sample:
            g = by_sample[rec.sample_id]
            if len(g) >= p.min_players_for_possession:
                d = np.hypot(g["x_m"].to_numpy() - b.at[rec.sample_id, "x_m"], g["y_m"].to_numpy() - b.at[rec.sample_id, "y_m"])
                k = int(np.argmin(d))
                row.update(ball_available=True, nearest_track=int(g["track_id"].iloc[k]),
                           nearest_team=g["team_id"].iloc[k], nearest_dist_m=float(d[k]))
                if d[k] <= p.possession_radius_m:
                    row["holder_team"] = g["team_id"].iloc[k]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["carried_team"] = out["holder_team"].ffill()
    out["carried"] = out["holder_team"].isna() & out["carried_team"].notna()
    return out


def possession_summary(poss: pd.DataFrame) -> dict:
    """Percent of frames per team, strict and with carry-over, with the counts behind them."""
    strict = poss["holder_team"].value_counts()
    carry = poss["carried_team"].value_counts()

    def pct(c: pd.Series) -> dict:
        total = int(c.sum())
        return {t: (round(100 * int(c.get(t, 0)) / total, 1) if total else None) for t in TEAMS}

    return {
        "samples": int(len(poss)), "ball_available": int(poss["ball_available"].sum()),
        "strict_frames": {t: int(strict.get(t, 0)) for t in TEAMS}, "strict_percent": pct(strict),
        "carry_frames": {t: int(carry.get(t, 0)) for t in TEAMS}, "carry_percent": pct(carry),
        "carried_share": round(float(poss["carried"].mean()), 3) if len(poss) else None,
    }


def coverage_summary(players: pd.DataFrame, traj: pd.DataFrame) -> dict:
    """How many of the 22 players the export actually observes, and how many are in view per sample."""
    pl = players[players["role"].isin(PLAYER_ROLES)]
    per_sample = pl.groupby("sample_id")["track_id"].nunique()
    seen = {t: int(pl[pl["team_id"] == t]["track_id"].nunique()) for t in (*TEAMS, "unknown")}
    return {
        "player_tracks_observed": int(pl["track_id"].nunique()), "tracks_by_team": seen,
        "players_in_view_per_sample": {"median": float(per_sample.median()), "max": int(per_sample.max()),
                                       "min": int(per_sample.min())} if len(per_sample) else {},
        "person_rows_total": int(traj["role"].isin(PLAYER_ROLES + ("referee",)).sum()),
        "person_rows_used": int(len(players)),
    }


def player_summary(players: pd.DataFrame) -> pd.DataFrame:
    """Per track: team, observed time, distance, speed statistics and flag counts."""
    pl = players[players["role"].isin(PLAYER_ROLES)]
    rows = []
    for tid, g in pl.groupby("track_id"):
        ok = g["speed_kmh"].dropna()
        span = float(g["timestamp_s"].max() - g["timestamp_s"].min())
        observed = float(len(g) * 0.2)
        rows.append({
            "track_id": int(tid), "team_id": g["team_id"].mode().iat[0], "role": g["role"].mode().iat[0],
            "first_s": float(g["timestamp_s"].min()), "last_s": float(g["timestamp_s"].max()), "rows": int(len(g)),
            "distance_m": round(float(g["cum_dist_m"].max()), 1), "unmeasured_s": round(max(span - observed, 0.0), 1),
            "speed_p50_kmh": round(float(ok.median()), 1) if len(ok) else np.nan,
            "speed_max_kmh": round(float(ok.max()), 1) if len(ok) else np.nan,
            "implausible_rows": int((g["speed_flag"] == "implausible").sum()),
        })
    return pd.DataFrame(rows)
