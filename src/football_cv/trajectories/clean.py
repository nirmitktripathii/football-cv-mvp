"""Trajectory cleaning: implausible-jump flags, light smoothing and speeds on observed points only.

No positions are invented. Smoothing is a centred rolling median over at most
three *observed* consecutive samples of the same track; a gap longer than
``MAX_GAP_S`` splits the track into runs that are smoothed separately, and
speed is only computed between consecutive observations of one run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP_S = 0.45  # > two 5 fps sample intervals means the track was not observed in between
BALL_MAX_SPEED_MPS = 40.0


def _flag_jumps(t: np.ndarray, x: np.ndarray, y: np.ndarray, vmax: float) -> np.ndarray:
    """Greedy pass: a point is a jump if reaching it from the last accepted point needs > vmax."""
    jump = np.zeros(len(t), dtype=bool)
    last = 0
    for i in range(1, len(t)):
        dt = t[i] - t[last]
        if dt <= 0 or np.hypot(x[i] - x[last], y[i] - y[last]) / dt > vmax:
            jump[i] = True
        else:
            last = i
    return jump


def _smooth_run(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["x_s"] = g["x_m"].rolling(3, center=True, min_periods=1).median()
    g["y_s"] = g["y_m"].rolling(3, center=True, min_periods=1).median()
    g["speed_mps"] = (np.hypot(g["x_s"].diff(), g["y_s"].diff()) / g["timestamp_s"].diff()).round(2)
    return g


def clean_trajectories(proj: pd.DataFrame, max_speed_mps: float, min_track_points: int) -> pd.DataFrame:
    """Usable (calibrated, on-pitch) rows with ``jump``, ``run_id``, ``x_s``, ``y_s``, ``speed_mps``, ``short_track``."""
    ok = proj[proj["calibration_status"].isin(["keyframe", "propagated"])]
    ok = ok.sort_values(["track_id", "timestamp_s"])
    parts = []
    for (tid, _), g in ok.groupby(["track_id", "segment_id"], sort=False):
        g = g.copy()
        vmax = BALL_MAX_SPEED_MPS if tid < 0 else max_speed_mps
        g["jump"] = _flag_jumps(g["timestamp_s"].to_numpy(), g["x_m"].to_numpy(), g["y_m"].to_numpy(), vmax)
        good = g[~g["jump"]].copy()
        good["run_id"] = (good["timestamp_s"].diff().fillna(0) > MAX_GAP_S).cumsum()
        good = pd.concat([_smooth_run(r) for _, r in good.groupby("run_id")]) if len(good) else good
        parts.append(g.join(good[["run_id", "x_s", "y_s", "speed_mps"]]) if len(good) else g)
    if not parts:
        return ok.assign(jump=False, run_id=np.nan, x_s=np.nan, y_s=np.nan, speed_mps=np.nan, short_track=False)
    out = pd.concat(parts)
    n_obs = out[~out["jump"]].groupby("track_id")["sample_id"].transform("size")
    out["short_track"] = False
    out.loc[n_obs.index, "short_track"] = (n_obs < min_track_points) & (out.loc[n_obs.index, "track_id"] >= 0)
    return out.sort_values(["timestamp_s", "track_id"]).reset_index(drop=True)


def usable(traj: pd.DataFrame) -> pd.DataFrame:
    """Rows fit for display and metrics: calibrated, not a jump, from a track long enough to trust."""
    return traj[~traj["jump"] & traj["x_s"].notna() & ~traj["short_track"]]


def track_summary(traj: pd.DataFrame) -> pd.DataFrame:
    """Per person track: team, span, observations, distance covered and speed statistics."""
    use = traj[(traj["track_id"] >= 0) & ~traj["jump"] & ~traj["short_track"]]
    rows = []
    for tid, g in use.groupby("track_id"):
        dist = sum(np.hypot(r["x_s"].diff(), r["y_s"].diff()).sum() for _, r in g.groupby("run_id"))
        has_speed = g["speed_mps"].notna().any()
        rows.append(
            {
                "track_id": tid, "team_id": g["team_id"].iloc[0], "start_s": g["timestamp_s"].min(),
                "end_s": g["timestamp_s"].max(), "n_obs": len(g), "distance_m": round(float(dist), 1),
                "median_speed_mps": round(float(g["speed_mps"].median()), 2) if has_speed else np.nan,
                "p95_speed_mps": round(float(g["speed_mps"].quantile(0.95)), 2) if has_speed else np.nan,
                "mean_x_m": round(float(g["x_s"].mean()), 1), "mean_y_m": round(float(g["y_s"].mean()), 1),
            }
        )
    return pd.DataFrame(rows)
