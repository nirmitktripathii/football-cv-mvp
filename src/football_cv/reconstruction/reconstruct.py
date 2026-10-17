"""Evidence-gated reconstruction of missing person positions, with provenance.

Input: cleaned person trajectories (``trajectories/clean.py``), the per-sample shot table and the
per-frame calibration. Output, per identity and sample inside its life span:

* ``player_states``  one row per sample: published position (or none), the estimate behind it,
  1-sigma uncertainty, ``position_source``, ``state``, ``visibility``, ``confidence``,
  calibration status and time since the last observation.
* ``gaps``           one row per missing interval, including intervals that were *not* bridged
  (camera cut, uncalibrated frames, too long).

Rules, in order of precedence:

1. An observation is published as ``OBSERVED`` / ``VISIBLE_ON_PITCH``. Nothing is overwritten.
2. A missing sample on an uncalibrated frame (replay, close-up, rejected calibration) gets no
   coordinates: ``UNKNOWN_MISSING``. Gaps are never bridged across a camera cut.
3. On a calibrated frame the gap is filled by a motion model: ``INTERPOLATED`` between two
   observations of one shot, ``PREDICTED`` (history only) for at most ``predict_horizon_s`` after
   the last one. The position is withheld when the gap is longer than ``max_gap_s`` or the model's
   1-sigma exceeds ``max_sigma_m``; the estimate and an uncertainty radius are still reported.
4. State is decided from geometry, not from the gap alone. The estimate and its 2-sigma disc are
   projected into the frame with the inverse homography: wholly inside the image and the gap is
   short -> ``OCCLUDED``; wholly outside -> ``OUTSIDE_CAMERA_VIEW``; straddling the border, or an
   in-view gap longer than ``max_occlusion_s`` (more likely a tracker break) -> ``UNKNOWN_MISSING``.
5. ``OFF_PITCH_INJURY_TREATMENT``, ``RETURNING_TO_PLAY`` and ``SUBSTITUTED_CONFIRMED`` are assigned
   only from manual annotations (``load_annotations``), never from a gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from football_cv.calibration.motion import apply_h
from football_cv.calibration.projection import H_of
from football_cv.reconstruction import motion_models as mm
from football_cv.reconstruction.states import ANNOTATION_ONLY_STATES, PlayerState, PositionSource

DEFAULTS = {
    "interpolate": "rts",        # method between two observations: rts | linear
    "predict": "cv",             # after the last observation: cv | kalman | hold (cv won the masking eval at <= 2 s)
    "max_gap_s": 8.0,            # longer bracketed gaps are not filled
    "predict_horizon_s": 2.0,    # history-only predictions stop this long after the last observation
    "max_occlusion_s": 3.0,      # an in-view gap longer than this is UNKNOWN_MISSING, not OCCLUDED
    "max_sigma_m": 5.0,          # estimates less certain than this are withheld
    "view_margin_px": 20.0,      # tolerance around the image border for the in-view test
    "image_size": [1280, 720],   # (width, height) of the sampled frames
}
CALIBRATED = ("keyframe", "propagated")
STATE_COLUMNS = [
    "sample_id", "timestamp_s", "identity_id", "track_id", "team_id", "shot_id", "x_m", "y_m", "x_est_m",
    "y_est_m", "sigma_m", "position_source", "state", "visibility", "confidence", "calibration_status",
    "keyframe_dt_s", "time_since_obs_s", "time_to_next_obs_s", "gap_s", "method", "reason",
]


def settings(cfg: dict | None) -> dict:
    return DEFAULTS | ((cfg or {}).get("reconstruction") or {})


def load_annotations(path: str | Path | None) -> list[dict]:
    """Manual events: ``{identity_id, state, start_s, end_s, [x_m, y_m], [note]}``; ``[]`` if absent."""
    if path is None or not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)
    for a in items:
        if PlayerState(a["state"]) not in ANNOTATION_ONLY_STATES:
            raise ValueError(f"annotation state must be one of {sorted(s.value for s in ANNOTATION_ONLY_STATES)}")
        if a["end_s"] < a["start_s"]:
            raise ValueError(f"annotation ends before it starts: {a}")
    return items


def view_state(H: np.ndarray, centre: np.ndarray, sigma_m: float, image_size, margin_px: float) -> PlayerState:
    """Where the 2-sigma disc around a pitch point falls in the frame of homography ``H`` (image -> pitch)."""
    w, h = image_size
    r = 2.0 * sigma_m
    pts = centre + np.array([[0, 0], [r, 0], [-r, 0], [0, r], [0, -r]], float)
    img = apply_h(np.linalg.inv(H), pts)
    if not np.all(np.isfinite(img)):
        return PlayerState.UNKNOWN_MISSING
    inside = (
        (img[:, 0] >= -margin_px) & (img[:, 0] <= w + margin_px)
        & (img[:, 1] >= -margin_px) & (img[:, 1] <= h + margin_px)
    )
    if inside.all():
        return PlayerState.OCCLUDED
    if not inside.any():
        return PlayerState.OUTSIDE_CAMERA_VIEW
    return PlayerState.UNKNOWN_MISSING


def _neighbours(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the last observation at/before and the first at/after each position (-1 if none)."""
    n = len(observed)
    pos = np.arange(n)
    prev = np.maximum.accumulate(np.where(observed, pos, -1))
    nxt = np.minimum.accumulate(np.where(observed, pos, n)[::-1])[::-1]
    return prev, np.where(nxt == n, -1, nxt)


def _blank(sample, ident, team, shot_id, cal) -> dict:
    return {
        "sample_id": int(sample.sample_id), "timestamp_s": float(sample.timestamp_s), "identity_id": ident,
        "track_id": ident, "team_id": team, "shot_id": int(shot_id),
        "calibration_status": cal["calibration_status"] if cal is not None else "not_wide_view",
        "keyframe_dt_s": float(cal["keyframe_dt_s"]) if cal is not None else np.nan,
    }


def _segment_rows(ident, team, obs: pd.DataFrame, shot_tl: pd.DataFrame, cal: pd.DataFrame, rejected: set, s: dict):
    """Rows for one identity inside one shot (first observation -> last observation + horizon)."""
    shot_tl = shot_tl.sort_values("sample_id").reset_index(drop=True)
    sid = shot_tl["sample_id"].to_numpy()
    ts = shot_tl["timestamp_s"].to_numpy()
    i0 = int(np.searchsorted(sid, obs["sample_id"].min()))
    i_last = int(np.searchsorted(sid, obs["sample_id"].max()))
    i1 = int(np.searchsorted(ts, ts[i_last] + s["predict_horizon_s"], side="right"))
    seg = shot_tl.iloc[i0:i1].reset_index(drop=True)
    t = seg["timestamp_s"].to_numpy()
    by_sid = obs.set_index("sample_id")
    observed = seg["sample_id"].isin(by_sid.index).to_numpy()
    z = np.full((len(seg), 2), np.nan)
    z[observed] = by_sid.loc[seg["sample_id"][observed], ["x_s", "y_s"]].to_numpy()
    models = mm.reconstruct_methods(t, z, observed)
    prev, nxt = _neighbours(observed)
    rows = []
    for i, sample in enumerate(seg.itertuples()):
        c = cal.loc[sample.sample_id] if sample.sample_id in cal.index else None
        row = _blank(sample, ident, team, sample.shot_id, c)
        if observed[i]:
            o = by_sid.loc[sample.sample_id]
            rows.append(row | {
                "x_m": z[i, 0], "y_m": z[i, 1], "x_est_m": z[i, 0], "y_est_m": z[i, 1], "sigma_m": mm.SIGMA_MEAS,
                "position_source": PositionSource.OBSERVED.value, "state": PlayerState.VISIBLE_ON_PITCH.value,
                "visibility": "visible", "confidence": float(o["confidence"]), "time_since_obs_s": 0.0,
                "time_to_next_obs_s": 0.0, "gap_s": 0.0, "method": "observed", "reason": "",
            })
            continue
        bracketed = nxt[i] >= 0
        gap_s = float(t[nxt[i]] - t[prev[i]]) if bracketed else float(t[i] - t[prev[i]])
        row |= {
            "time_since_obs_s": float(t[i] - t[prev[i]]),
            "time_to_next_obs_s": float(t[nxt[i]] - t[i]) if bracketed else np.nan, "gap_s": gap_s,
            "visibility": "not_visible", "x_m": np.nan, "y_m": np.nan, "x_est_m": np.nan, "y_est_m": np.nan,
            "sigma_m": np.nan, "position_source": "", "method": "", "confidence": 0.0,
            "state": PlayerState.UNKNOWN_MISSING.value,
        }
        if c is None or c["calibration_status"] not in CALIBRATED:
            rows.append(row | {"reason": f"uncalibrated frame ({row['calibration_status']})"})
            continue
        if sample.sample_id in rejected:
            rows.append(row | {"reason": "detected but rejected as an implausible jump"})
            continue
        method = s["interpolate"] if bracketed else s["predict"]
        mean, std = models[method]
        est, sigma = mean[i], float(std[i])
        if not np.isfinite(est).all() or not np.isfinite(sigma):
            rows.append(row | {"reason": f"{method} gave no estimate"})
            continue
        state = view_state(H_of(c), est, sigma, s["image_size"], s["view_margin_px"])
        if state is PlayerState.OCCLUDED and gap_s > s["max_occlusion_s"]:
            state = PlayerState.UNKNOWN_MISSING
        reason = ""
        if bracketed and gap_s > s["max_gap_s"]:
            reason = f"gap {gap_s:.1f}s exceeds max_gap_s"
        elif sigma > s["max_sigma_m"]:
            reason = f"sigma {sigma:.1f}m exceeds max_sigma_m"
        row |= {
            "x_est_m": float(est[0]), "y_est_m": float(est[1]), "sigma_m": sigma, "method": method,
            "state": state.value, "reason": reason,
        }
        if not reason:
            src = PositionSource.INTERPOLATED if bracketed else PositionSource.PREDICTED
            row |= {
                "x_m": float(est[0]), "y_m": float(est[1]), "position_source": src.value,
                "confidence": float(np.clip(1.0 - sigma / s["max_sigma_m"], 0.0, 1.0)),
            }
        rows.append(row)
    return rows


def _cut_gaps(ident, shots_seen: list[tuple[int, float, float]]) -> list[dict]:
    """Intervals between consecutive shots in which one identity was observed (never bridged)."""
    out = []
    for (sa, _, end_a), (sb, start_b, _) in zip(shots_seen, shots_seen[1:]):
        out.append({
            "identity_id": ident, "start_s": end_a, "end_s": start_b, "duration_s": start_b - end_a,
            "kind": "across_shots", "state": PlayerState.UNKNOWN_MISSING.value, "filled": False,
            "reason": f"camera cut or replay between shot {sa} and shot {sb}; not bridged",
        })
    return out


def summarise_gaps(states: pd.DataFrame) -> pd.DataFrame:
    """One row per maximal run of non-observed samples of an identity within a shot."""
    miss = states[states["position_source"] != PositionSource.OBSERVED.value].copy()
    if miss.empty:
        return pd.DataFrame()
    miss = miss.sort_values(["identity_id", "shot_id", "sample_id"])
    brk = (miss.groupby(["identity_id", "shot_id"])["sample_id"].diff() != 1).cumsum()
    rows = []
    for _, g in miss.groupby(brk):
        filled = g["position_source"] != ""
        rows.append({
            "identity_id": g["identity_id"].iloc[0], "start_s": g["timestamp_s"].iloc[0],
            "end_s": g["timestamp_s"].iloc[-1], "duration_s": g["gap_s"].max(),
            "kind": "interior" if g["time_to_next_obs_s"].notna().all() else "trailing",
            "state": g["state"].mode().iloc[0], "filled": bool(filled.any()),
            "reason": "; ".join(sorted(set(g["reason"]) - {""})),
        })
    return pd.DataFrame(rows)


def apply_annotations(states: pd.DataFrame, timeline: pd.DataFrame, annotations: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Overlay manual events on non-observed samples; observed samples win and the conflict is reported."""
    warnings: list[str] = []
    if not annotations:
        return states, warnings
    states = states.copy()
    extra = []
    for a in annotations:
        sel = timeline[(timeline["timestamp_s"] >= a["start_s"]) & (timeline["timestamp_s"] <= a["end_s"])]
        mine = states["identity_id"] == a["identity_id"]
        manual = {"x_m", "y_m"} <= a.keys() and a["x_m"] is not None and a["y_m"] is not None
        for sample in sel.itertuples():
            hit = mine & (states["sample_id"] == sample.sample_id)
            if hit.any() and (states.loc[hit, "position_source"] == PositionSource.OBSERVED.value).any():
                warnings.append(f"identity {a['identity_id']} observed at t={sample.timestamp_s:.1f}s inside annotation")
                continue
            patch = {
                "state": a["state"], "visibility": "not_visible", "reason": a.get("note", "manual annotation"),
                "method": "manual", "x_m": a["x_m"] if manual else np.nan, "y_m": a["y_m"] if manual else np.nan,
                "position_source": PositionSource.MANUALLY_ANNOTATED.value if manual else "",
                "confidence": 1.0 if manual else 0.0,
            }
            if hit.any():
                for k, v in patch.items():
                    states.loc[hit, k] = v
            else:
                extra.append({
                    "sample_id": int(sample.sample_id), "timestamp_s": float(sample.timestamp_s),
                    "identity_id": a["identity_id"], "track_id": a["identity_id"], "shot_id": int(sample.shot_id),
                    "calibration_status": "n/a", "keyframe_dt_s": np.nan,
                } | patch)
    if extra:
        states = pd.concat([states, pd.DataFrame(extra)], ignore_index=True)
    return states.sort_values(["identity_id", "sample_id"]).reset_index(drop=True), warnings


def reconstruct(traj: pd.DataFrame, timeline: pd.DataFrame, calib: pd.DataFrame, cfg: dict | None = None,
                annotations: list[dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """``(player_states, gaps, warnings)`` for every person identity in ``traj``.

    ``traj`` needs ``track_id, sample_id, timestamp_s, shot_id, team_id, x_s, y_s, confidence, jump``
    (and ``identity_id`` if tracklets have been linked; otherwise ``track_id`` is the identity).
    ``timeline`` needs ``sample_id, timestamp_s, shot_id``; ``calib`` is ``frame_calibration``.
    """
    s = settings(cfg)
    people = traj[traj["track_id"] >= 0].copy()
    if "identity_id" not in people:
        people["identity_id"] = people["track_id"]
    good = people[~people["jump"] & people["x_s"].notna() & ~people["short_track"]]
    cal = calib.set_index("sample_id")
    rejected_by_ident = people[people["jump"]].groupby("identity_id")["sample_id"].apply(set).to_dict()
    rows: list[dict] = []
    gaps: list[dict] = []
    for ident, g in good.groupby("identity_id"):
        team = g["team_id"].mode().iloc[0]
        seen = []
        for shot_id, gs in g.groupby("shot_id"):
            shot_tl = timeline[timeline["shot_id"] == shot_id]
            rows += _segment_rows(ident, team, gs, shot_tl, cal, rejected_by_ident.get(ident, set()), s)
            seen.append((int(shot_id), float(gs["timestamp_s"].min()), float(gs["timestamp_s"].max())))
        gaps += _cut_gaps(ident, sorted(seen, key=lambda r: r[1]))
    states = pd.DataFrame(rows, columns=STATE_COLUMNS) if rows else pd.DataFrame(columns=STATE_COLUMNS)
    states, warnings = apply_annotations(states, timeline, annotations or [])
    all_gaps = pd.concat([summarise_gaps(states), pd.DataFrame(gaps)], ignore_index=True)
    return states, all_gaps, warnings
