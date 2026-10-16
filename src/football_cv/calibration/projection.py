"""Per-frame calibration and projection of tracked foot points onto the pitch.

``calibrate_keyframes`` fits and refines each manual keyframe. ``frame_calibrations``
propagates keyframes through their motion segment and validates every frame.
``project_tracks`` maps each track's bottom-centre point with its frame's
homography and labels it with ``calibration_status``:

* ``keyframe`` / ``propagated``  - valid transform, point inside pitch + margin
* ``outside_pitch``              - valid transform, point beyond the margin (kept, not used)
* ``rejected:<reason>``          - transform failed validation; no coordinates
* ``no_keyframe``                - wide-view segment that was never annotated; no coordinates

Frames that are not the tactical wide view never get coordinates (they have
no motion row and no track rows).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pandas as pd

from football_cv.calibration import homography as hg
from football_cv.calibration.motion import G_of, apply_h
from football_cv.calibration.shots import grass_mask

log = logging.getLogger(__name__)

H_COLS = [f"h{i}{j}" for i in range(3) for j in range(3)]


def calibrate_keyframes(
    keyframes: list[hg.Keyframe], frames: pd.DataFrame, frames_root, motion: pd.DataFrame, refine: bool = True
) -> list[hg.Keyframe]:
    """Fit, optionally line-refine and validate each keyframe in place."""
    paths = frames.set_index("sample_id")["path"]
    mot = motion.set_index("sample_id")
    for kf in keyframes:
        if kf.sample_id not in mot.index:
            raise ValueError(f"keyframe {kf.sample_id} is not a wide-view frame with motion data")
        img = cv2.imread(str(frames_root / paths[kf.sample_id]))
        lines = (kf.line_img, kf.line_axis, kf.line_val)
        H, rms = hg.fit_from_correspondences(kf.image_pts, kf.pitch_pts, *lines)
        diag = {
            "n_points": len(kf.names), "n_line_points": len(kf.line_names), "manual_rms_m": round(rms, 3),
            "segment_id": int(mot.at[kf.sample_id, "segment_id"]),
        }
        grass = grass_mask(img)
        if refine:
            H, rdiag = hg.refine_with_lines(H, img, grass, kf.image_pts, kf.pitch_pts, anchor_lines=lines)
            diag |= rdiag
        ok, why = hg.validate(H, img.shape[:2], grass)
        err = np.linalg.norm(apply_h(H, kf.image_pts) - kf.pitch_pts, axis=1)
        diag |= {"valid": ok, "validation": why, "final_point_errors_m": dict(zip(kf.names, err.round(2).tolist()))}
        if kf.line_names:
            lerr = hg.line_errors(H, *lines).round(2).tolist()
            diag["final_line_errors_m"] = [[n, e] for n, e in zip(kf.line_names, lerr)]
        kf.H, kf.diagnostics = (H if ok else None), diag
        log.info("Keyframe %d: %s", kf.sample_id, diag)
    return keyframes


def drift_checks(keyframes: list[hg.Keyframe], motion: pd.DataFrame, image_shape) -> list[dict]:
    """Propagate each keyframe to the next one in the same segment and compare (metres).

    ``landmark_*`` measures the propagated transform at the next keyframe's own
    manual landmarks (where the truth is known): motion-chain drift plus click
    noise. ``probe_*`` compares both transforms on a fixed grid in the lower
    image, which also includes extrapolation disagreement between two
    keyframes whose landmarks lie elsewhere in the image.
    """
    mot = motion.set_index("sample_id")
    h, w = image_shape
    probe = np.array([[u, v] for v in (h * 0.45, h * 0.7, h * 0.95) for u in (w * 0.1, w * 0.5, w * 0.9)])
    valid = sorted((k for k in keyframes if k.H is not None), key=lambda k: k.sample_id)
    out = []
    for a, b in zip(valid, valid[1:]):
        if mot.at[a.sample_id, "segment_id"] != mot.at[b.sample_id, "segment_id"]:
            continue
        Hb_pred = hg.frame_homography(a.H, G_of(mot.loc[a.sample_id]), G_of(mot.loc[b.sample_id]))
        d = np.linalg.norm(apply_h(Hb_pred, probe) - apply_h(b.H, probe), axis=1)
        lm = np.linalg.norm(apply_h(Hb_pred, b.image_pts) - b.pitch_pts, axis=1)
        out.append(
            {
                "from_sample": a.sample_id, "to_sample": b.sample_id,
                "dt_s": round(float(mot.at[b.sample_id, "timestamp_s"] - mot.at[a.sample_id, "timestamp_s"]), 1),
                "landmark_median_error_m": round(float(np.median(lm)), 2),
                "landmark_max_error_m": round(float(lm.max()), 2),
                "probe_median_drift_m": round(float(np.median(d)), 2), "probe_max_drift_m": round(float(d.max()), 2),
            }
        )
    return out


def frame_calibrations(
    keyframes: list[hg.Keyframe], motion: pd.DataFrame, frames: pd.DataFrame, frames_root,
    max_keyframe_dt_s: float = np.inf,
) -> pd.DataFrame:
    """One row per wide-view frame with its image->pitch homography (h00..h22) and status.

    Each frame uses the nearest valid keyframe of its motion segment. Frames more
    than ``max_keyframe_dt_s`` from it are rejected: motion-chain drift grows
    with time and is only measured where two keyframes share a segment.
    """
    mot = motion.set_index("sample_id")
    by_segment: dict[int, list[hg.Keyframe]] = {}
    for k in keyframes:
        if k.H is not None:
            by_segment.setdefault(int(mot.at[k.sample_id, "segment_id"]), []).append(k)
    paths = frames.set_index("sample_id")["path"]
    rows = []
    for m in motion.itertuples():
        kfs = by_segment.get(int(m.segment_id))
        row = {"sample_id": m.sample_id, "timestamp_s": m.timestamp_s, "segment_id": m.segment_id}
        if not kfs:
            rows.append(row | {"calibration_status": "no_keyframe", "keyframe_id": -1, "keyframe_dt_s": np.nan})
            continue
        k = min(kfs, key=lambda kf: abs(mot.at[kf.sample_id, "timestamp_s"] - m.timestamp_s))
        dt = float(m.timestamp_s - mot.at[k.sample_id, "timestamp_s"])
        if abs(dt) > max_keyframe_dt_s:
            rows.append(row | {"calibration_status": "rejected:too far from keyframe", "keyframe_id": k.sample_id,
                               "keyframe_dt_s": round(dt, 2)})
            continue
        H = k.H if k.sample_id == m.sample_id else hg.frame_homography(k.H, G_of(mot.loc[k.sample_id]), G_of(m))
        grass = grass_mask(cv2.imread(str(frames_root / paths[m.sample_id])))
        ok, why = hg.validate(H, grass.shape, grass)
        status = ("keyframe" if k.sample_id == m.sample_id else "propagated") if ok else f"rejected:{why}"
        row |= {
            "calibration_status": status, "keyframe_id": k.sample_id,
            "keyframe_dt_s": round(dt, 2),
        }
        if ok:
            row |= dict(zip(H_COLS, H.ravel().astype(float)))
        rows.append(row)
    out = pd.DataFrame(rows)
    log.info("Frame calibration status: %s", out["calibration_status"].value_counts().to_dict())
    return out


def H_of(row) -> np.ndarray:
    """3x3 image -> pitch homography from a calibration row (Series)."""
    return np.array([row[c] for c in H_COLS], dtype=float).reshape(3, 3)


def project_tracks(
    tracks: pd.DataFrame, calib: pd.DataFrame, margin_m: float, length: float, width: float
) -> pd.DataFrame:
    """Add ``x_m, y_m, calibration_status`` to every track row."""
    out = tracks.copy()
    out["x_m"] = np.nan
    out["y_m"] = np.nan
    out["calibration_status"] = "not_wide_view"
    cal = calib.set_index("sample_id")
    for sid, idx in out.groupby("sample_id").groups.items():
        if sid not in cal.index:
            continue
        c = cal.loc[sid]
        status = c["calibration_status"]
        if status not in ("keyframe", "propagated"):
            out.loc[idx, "calibration_status"] = status
            continue
        p = apply_h(H_of(c), out.loc[idx, ["foot_x", "foot_y"]].to_numpy())
        inside = (
            (p[:, 0] >= -margin_m) & (p[:, 0] <= length + margin_m)
            & (p[:, 1] >= -margin_m) & (p[:, 1] <= width + margin_m)
        )
        out.loc[idx, "x_m"] = p[:, 0].round(3)
        out.loc[idx, "y_m"] = p[:, 1].round(3)
        out.loc[idx, "calibration_status"] = np.where(inside, status, "outside_pitch")
    return out
