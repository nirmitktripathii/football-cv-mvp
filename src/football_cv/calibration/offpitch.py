"""Remove what is not part of play from projected tracks: people standing off the pitch, balls that are not the match ball.

Both work on pitch coordinates (``x_m``, ``y_m``), so they run after calibration. Rows without a pitch position are never
removed: no position, no evidence.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEEP_ROLES = ("referee", "goalkeeper")  # referees and assistants work the touchline; keepers stand in or behind goal


def _outside_m(x: pd.Series, y: pd.Series, length: float, width: float) -> pd.Series:
    """Distance beyond the nearest line of the pitch rectangle (0 inside)."""
    return pd.concat([-x, x - length, -y, y - width], axis=1).max(axis=1).clip(lower=0.0)


def drop_off_pitch_people(
    proj: pd.DataFrame, length: float, width: float, outside_m: float = 2.0, min_rows: int = 3,
    keep_roles: tuple = KEEP_ROLES,
) -> tuple[pd.DataFrame, list[int]]:
    """Drop person tracks whose median position is more than ``outside_m`` beyond the lines.

    Coaches, substitutes warming up and ball boys stand there for the whole track, while a player who steps out for a
    throw-in has a median inside the lines. Tracks with fewer than ``min_rows`` positioned rows, and ``keep_roles``,
    are kept. Returns the remaining rows and the dropped track ids.
    """
    people = proj[(proj["object_class"] == "person") & proj["x_m"].notna() & ~proj["role"].isin(keep_roles)]
    beyond = _outside_m(people["x_m"], people["y_m"], length, width).groupby(people["track_id"])
    stats = pd.DataFrame({"median": beyond.median(), "n": beyond.size()})
    dropped = sorted(int(t) for t in stats.index[(stats["median"] > outside_m) & (stats["n"] >= min_rows)])
    if dropped:
        log.info("Off-pitch people: dropped %d tracks (> %.1f m outside the lines)", len(dropped), outside_m)
    return proj[~((proj["object_class"] == "person") & proj["track_id"].isin(dropped))], dropped


def drop_corner_flags(
    proj: pd.DataFrame, length: float, width: float, radius_m: float = 2.5, still_m: float = 0.6, min_rows: int = 2,
    keep_roles: tuple = KEEP_ROLES,
) -> tuple[pd.DataFrame, list[int]]:
    """Drop person tracks that are a corner flag: still, and standing on a pitch corner.

    A flag is a constant of the pitch. A track is dropped when it has at least ``min_rows`` positioned rows, its median
    foot position lies within ``radius_m`` of the nearest corner (wide, because a box over the upper pole projects its
    foot beyond the real base) and its x and y standard deviations are each at most ``still_m``. A corner taker runs up,
    so his spread is larger; ``keep_roles`` are never dropped. Returns the remaining rows and the dropped track ids.
    """
    people = proj[(proj["object_class"] == "person") & proj["x_m"].notna() & ~proj["role"].isin(keep_roles)]
    if people.empty:
        return proj, []
    corners = np.array([[0.0, 0.0], [length, 0.0], [0.0, width], [length, width]])
    by_track = people.groupby("track_id")
    stats = pd.DataFrame({"x": by_track["x_m"].median(), "y": by_track["y_m"].median(), "n": by_track.size(),
                          "sx": by_track["x_m"].std(ddof=0), "sy": by_track["y_m"].std(ddof=0)})
    near = np.linalg.norm(stats[["x", "y"]].to_numpy()[:, None, :] - corners[None], axis=2).min(axis=1)
    is_flag = (near <= radius_m) & (stats["sx"] <= still_m) & (stats["sy"] <= still_m) & (stats["n"] >= min_rows)
    dropped = sorted(int(t) for t in stats.index[is_flag.to_numpy()])
    if dropped:
        log.info("Corner flags: dropped %d tracks (still within %.1f m of a corner)", len(dropped), radius_m)
    return proj[~((proj["object_class"] == "person") & proj["track_id"].isin(dropped))], dropped


def _spots(length: float, width: float, penalty_m: float = 11.0) -> np.ndarray:
    return np.array([[length / 2, width / 2], [penalty_m, width / 2], [length - penalty_m, width / 2]])


def reject_unplayable_balls(
    proj: pd.DataFrame, length: float, width: float, outside_m: float = 2.0, link_s: float = 1.0,
    link_speed_mps: float = 35.0, spot_radius_m: float = 0.6, spot_moved_m: float = 3.0,
) -> pd.DataFrame:
    """Drop ball rows that cannot be the match ball in play.

    * A ball more than ``outside_m`` beyond the lines is kept only when a ball detection inside the lines lies within
      ``link_s`` and within what a kick can cover (``link_speed_mps``): a ball played out of play stays linked to the
      play, a spare ball by the advertising boards does not.
    * A ball within ``spot_radius_m`` of the centre or a penalty spot is dropped when no ball detection within ``link_s``
      lies more than ``spot_moved_m`` away, i.e. it stayed put: the spot mark itself. A ball passing over a spot moves.
    """
    ball = proj[(proj["object_class"] == "ball") & proj["x_m"].notna()]
    if ball.empty:
        return proj
    t = ball["timestamp_s"].to_numpy(float)
    xy = ball[["x_m", "y_m"]].to_numpy(float)
    beyond = _outside_m(ball["x_m"], ball["y_m"], length, width).to_numpy()
    spot_d = np.linalg.norm(xy[:, None, :] - _spots(length, width)[None], axis=2).min(axis=1)
    drop = np.zeros(len(ball), dtype=bool)
    for i in range(len(ball)):
        near = (np.abs(t - t[i]) <= link_s) & (np.arange(len(ball)) != i)
        dist = np.linalg.norm(xy[near] - xy[i], axis=1)
        if beyond[i] > outside_m:
            reach = link_speed_mps * np.abs(t[near] - t[i]) + 2.0
            drop[i] = not np.any((beyond[near] <= outside_m) & (dist <= reach))
        elif spot_d[i] <= spot_radius_m:
            drop[i] = not np.any(dist > spot_moved_m)
    if drop.any():
        log.info("Unplayable balls: dropped %d of %d ball rows", int(drop.sum()), len(ball))
    return proj.drop(index=ball.index[drop])
