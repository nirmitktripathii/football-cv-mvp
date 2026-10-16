"""Team shape and ball-proximity metrics from observed, calibrated positions.

Every metric describes the VISIBLE SUBSET of a team: a broadcast camera shows
roughly a third to a half of the pitch, so width, depth and hull area are those
of the players in view, not the whole team. Frames with fewer than
``MIN_PLAYERS`` visible players of a team are skipped for that team.

``ball_proximity`` names the player nearest to the ball. It is a proxy and is
NOT possession: a defender closing down or a player the ball just passed are
nearest too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

from football_cv.trajectories.clean import usable

MIN_PLAYERS = 4
TEAMS = ("home", "away")


def _hull_area(pts: np.ndarray) -> float:
    try:
        return float(ConvexHull(pts).volume)  # in 2-D, "volume" is the area
    except QhullError:  # collinear players
        return 0.0


def _shape(pts: np.ndarray) -> dict:
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    return {
        "n_visible": len(pts),
        "centroid_x_m": pts[:, 0].mean(), "centroid_y_m": pts[:, 1].mean(),
        "width_m": np.ptp(pts[:, 1]), "depth_m": np.ptp(pts[:, 0]),
        "hull_area_m2": _hull_area(pts), "mean_nn_dist_m": d.min(axis=1).mean(),
    }


def team_shape(traj: pd.DataFrame, min_players: int = MIN_PLAYERS) -> pd.DataFrame:
    """One row per (sampled frame, team) with at least ``min_players`` visible outfield-coloured players."""
    use = usable(traj)
    use = use[(use["track_id"] >= 0) & use["team_id"].isin(TEAMS)]
    rows = []
    for (sid, team), g in use.groupby(["sample_id", "team_id"]):
        if len(g) < min_players:
            continue
        rows.append({"sample_id": sid, "timestamp_s": g["timestamp_s"].iloc[0], "team_id": team,
                     **_shape(g[["x_s", "y_s"]].to_numpy())})
    cols = ["sample_id", "timestamp_s", "team_id", "n_visible", "centroid_x_m", "centroid_y_m",
            "width_m", "depth_m", "hull_area_m2", "mean_nn_dist_m"]
    return pd.DataFrame(rows, columns=cols).round(2)


def ball_proximity(traj: pd.DataFrame) -> pd.DataFrame:
    """Nearest tracked player to each observed ball position (proxy, not possession)."""
    use = usable(traj)
    balls = use[use["track_id"] < 0]
    people = use[use["track_id"] >= 0]
    by_sample = {k: g for k, g in people.groupby("sample_id")}
    rows = []
    for b in balls.itertuples():
        g = by_sample.get(b.sample_id)
        if g is None or g.empty:
            continue
        d = np.hypot(g["x_s"].to_numpy() - b.x_s, g["y_s"].to_numpy() - b.y_s)
        i = int(d.argmin())
        rows.append({"sample_id": b.sample_id, "timestamp_s": b.timestamp_s,
                     "nearest_track_id": int(g["track_id"].iloc[i]), "nearest_team": g["team_id"].iloc[i],
                     "distance_m": round(float(d[i]), 2)})
    return pd.DataFrame(rows, columns=["sample_id", "timestamp_s", "nearest_track_id", "nearest_team", "distance_m"])


def shape_summary(shape: pd.DataFrame) -> pd.DataFrame:
    """Median shape per team across all frames where it was measurable."""
    if shape.empty:
        return shape
    agg = shape.groupby("team_id").agg(
        frames=("sample_id", "size"), median_visible=("n_visible", "median"),
        median_width_m=("width_m", "median"), median_depth_m=("depth_m", "median"),
        median_hull_area_m2=("hull_area_m2", "median"), median_nn_dist_m=("mean_nn_dist_m", "median"),
    )
    return agg.round(2).reset_index()
