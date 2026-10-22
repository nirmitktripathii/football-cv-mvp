"""Ball path selection and bounded gap filling, in stabilised (camera-pan-free) coordinates.

Step A, ``select_path``: instead of keeping the single most confident ball detection per sample, keep every
candidate and choose the chain of candidates (at most one per sample) that maximises detector evidence minus a
motion cost, by dynamic programming (Viterbi). A distant but confident false alarm loses to a weaker candidate that
continues the trajectory.

Step B, ``fill_gaps``: between two linked detections that are at most ``max_fill_samples`` apart, the missing
samples are filled with a constant-velocity Kalman filter and RTS smoother (``method="rts"``, the player models in
``reconstruction.motion_models``), or by linear interpolation or holding the last value for comparison. Nothing is
extrapolated beyond the first or last detection and nothing is filled across a motion segment (shot cut). Every
output row says whether it was ``detected`` or ``interpolated``; interpolated rows carry a per-axis position
uncertainty ``sigma_px`` (NaN for the comparison methods, which have no uncertainty model).

The motion model is 2-D in stabilised image pixels. A ball in the air is not on the pitch plane, so its pixel track
is still smooth but is not a metric pitch position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_cv.reconstruction import motion_models as mm

CANDIDATE_COLUMNS = ["sample_id", "timestamp_s", "segment_id", "confidence", "x", "y"]
TRACK_COLUMNS = ["sample_id", "timestamp_s", "segment_id", "x", "y", "status", "confidence", "sigma_px"]
DETECTED, INTERPOLATED = "detected", "interpolated"
METHODS = ("rts", "linear", "hold")


@dataclass(frozen=True)
class PathParams:
    """Fixed a-priori settings; none was fitted on the evaluation windows."""
    max_speed_px_s: float = 450.0      # stabilised px per second (a hard shot; 90 px per 0.2 s sample)
    max_link_s: float = 1.0            # detections further apart in time are never linked
    reference_conf: float = 0.10       # a candidate at this confidence adds nothing; higher adds log(c / ref)
    break_cost: float = 2.5            # nats charged when consecutive chosen candidates are not linked
    # nats credited to every chosen candidate. It must be at least ``break_cost``: log(conf / reference_conf) is at
    # most 2.3 for a confidence of 1, so without it any chain that breaks scores below the chain before the break
    # and every isolated detection is discarded. 0.0 reproduces that original (flawed) behaviour.
    detection_bonus: float = 2.5
    link_cost: float = 2.0             # nats charged for a link at maximum plausible speed (scales linearly)
    max_fill_samples: int = 3          # longest run of missing samples that is filled (0.6 s at 5 fps)
    accel_sigma_px_s2: float = 1500.0  # constant-velocity model process noise
    meas_sigma_px: float = 3.0         # ball box-centre jitter
    # initial velocity spread. Zero-mean, so it must be far wider than any real speed or it drags a fill toward a
    # stationary ball (225 px/s, i.e. max_speed / 2, biased a 200 px/s straight line by 4.6 px).
    velocity_prior_px_s: float = 5000.0
    method: str = "rts"


def candidates_from_detections(balls: pd.DataFrame, centres_stab: np.ndarray, segment_of: dict) -> pd.DataFrame:
    """Candidate table from ball detections and their stabilised centres."""
    out = pd.DataFrame({
        "sample_id": balls["sample_id"].to_numpy(), "timestamp_s": balls["timestamp_s"].to_numpy(dtype=float),
        "segment_id": [segment_of[s] for s in balls["sample_id"]],
        "confidence": balls["confidence"].to_numpy(dtype=float),
        "x": centres_stab[:, 0], "y": centres_stab[:, 1],
    })
    return out.sort_values(["timestamp_s", "confidence"], ascending=[True, False], ignore_index=True)[CANDIDATE_COLUMNS]


def _link_cost(a, b, p: PathParams) -> float | None:
    """Motion cost of linking candidate ``a`` to a later candidate ``b``; None when the link is not allowed."""
    dt = b.timestamp_s - a.timestamp_s
    if a.segment_id != b.segment_id or dt <= 0 or dt > p.max_link_s + 1e-9:
        return None
    u = float(np.hypot(b.x - a.x, b.y - a.y)) / (p.max_speed_px_s * dt)
    return None if u > 1.0 else p.link_cost * u


def _empty_path(cands: pd.DataFrame) -> pd.DataFrame:
    return cands.iloc[:0].assign(link_prev=pd.Series(dtype=bool))


def select_path(cands: pd.DataFrame, p: PathParams = PathParams()) -> pd.DataFrame:
    """Viterbi choice of at most one candidate per sample; adds ``link_prev`` (linked to the previous chosen one)."""
    if cands.empty:
        return _empty_path(cands)
    rows = list(cands.sort_values(["timestamp_s", "confidence"], ascending=[True, False]).itertuples(index=False))
    best, back, linked = np.zeros(len(rows)), np.full(len(rows), -1), np.zeros(len(rows), dtype=bool)
    # best chain ending in a strictly earlier sample (arg -1 = nothing chosen yet, starting a chain is free)
    prefix_val, prefix_arg, pending_val, pending_arg, prev_sample = 0.0, -1, 0.0, -1, None
    for i, c in enumerate(rows):
        if c.sample_id != prev_sample:
            prefix_val, prefix_arg, prev_sample = pending_val, pending_arg, c.sample_id
        top_val, top_arg, top_link = prefix_val - (p.break_cost if prefix_arg >= 0 else 0.0), prefix_arg, False
        for j in range(i - 1, -1, -1):
            if c.timestamp_s - rows[j].timestamp_s > p.max_link_s + 1e-9:
                break
            cost = _link_cost(rows[j], c, p) if rows[j].sample_id != c.sample_id else None
            if cost is not None and best[j] - cost > top_val:
                top_val, top_arg, top_link = best[j] - cost, j, True
        best[i] = float(np.log(max(c.confidence, 1e-6) / p.reference_conf)) + p.detection_bonus + top_val
        back[i], linked[i] = top_arg, top_link
        if best[i] > pending_val:
            pending_val, pending_arg = best[i], i
    return _trace(cands, rows, back, linked, pending_arg)


def _trace(cands: pd.DataFrame, rows: list, back: np.ndarray, linked: np.ndarray, end: int) -> pd.DataFrame:
    if end < 0:
        return _empty_path(cands)
    chosen, i = [], end
    while i >= 0:
        chosen.append(i)
        i = back[i]
    chosen.reverse()
    out = pd.DataFrame([rows[k] for k in chosen]).reset_index(drop=True)
    out["link_prev"] = [bool(linked[k]) for k in chosen]
    out.loc[0, "link_prev"] = False
    return out


def _fill_run(times: np.ndarray, anchors: np.ndarray, anchor_ix: np.ndarray, p: PathParams):
    """Positions and sigmas at all ``times`` of one run whose anchors sit at ``anchor_ix``."""
    z, observed = np.full((len(times), 2), np.nan), np.zeros(len(times), dtype=bool)
    z[anchor_ix], observed[anchor_ix] = anchors, True
    if p.method == "rts":
        _, _, mean, sd = mm.kalman_rts(times, z, observed, p.accel_sigma_px_s2, p.meas_sigma_px, p.velocity_prior_px_s)
        return mean, sd
    mean, sd = (mm.linear if p.method == "linear" else mm.hold)(times, z, observed)
    return mean, np.full(len(times), np.nan)


def _fillable(path: pd.DataFrame, ids: np.ndarray, p: PathParams) -> np.ndarray:
    ok = path["link_prev"].to_numpy().copy()
    seg = path["segment_id"].to_numpy()
    ok[1:] &= (ids[1:] - ids[:-1] - 1 <= p.max_fill_samples) & (seg[1:] == seg[:-1])
    ok[0] = False
    return ok


def fill_gaps(path: pd.DataFrame, samples: pd.DataFrame, p: PathParams = PathParams()) -> pd.DataFrame:
    """Detected rows plus interpolated rows for short gaps between linked detections.

    ``samples`` has one row per sample with ``sample_id``, ``timestamp_s`` and ``segment_id`` (the motion table).
    """
    if p.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    det = path.assign(status=DETECTED, sigma_px=0.0)[TRACK_COLUMNS]
    if len(path) < 2:
        return det.reset_index(drop=True)
    order = samples.sort_values("timestamp_s").reset_index(drop=True)
    pos = {s: i for i, s in enumerate(order["sample_id"])}
    ids = path["sample_id"].map(pos).to_numpy()
    xy_all = path[["x", "y"]].to_numpy(dtype=float)
    filled = []
    for run in np.split(np.arange(len(path)), np.flatnonzero(~_fillable(path, ids, p))):
        if len(run) < 2 or ids[run[-1]] - ids[run[0]] == len(run) - 1:
            continue
        span = order.iloc[ids[run[0]]: ids[run[-1]] + 1]
        times, a_ix = span["timestamp_s"].to_numpy(dtype=float), ids[run] - ids[run[0]]
        xy, sigma = _fill_run(times, xy_all[run], a_ix, p)
        gap = np.setdiff1d(np.arange(len(span)), a_ix)
        filled.append(pd.DataFrame({
            "sample_id": span["sample_id"].to_numpy()[gap], "timestamp_s": times[gap],
            "segment_id": span["segment_id"].to_numpy()[gap], "x": xy[gap, 0], "y": xy[gap, 1],
            "status": INTERPOLATED, "confidence": np.nan, "sigma_px": sigma[gap]}))
    return pd.concat([det] + filled, ignore_index=True).sort_values("timestamp_s", kind="stable", ignore_index=True)


def ball_track(cands: pd.DataFrame, samples: pd.DataFrame, p: PathParams = PathParams()) -> pd.DataFrame:
    """Candidates -> chosen path -> detected and interpolated ball rows (stabilised coordinates)."""
    return fill_gaps(select_path(cands, p), samples, p)
