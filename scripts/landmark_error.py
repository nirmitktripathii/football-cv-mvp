"""Metre error of the pitch mapping at hand-labelled points on painted lines.

Each label is an image point (u, v) that lies on a straight pitch marking. Mapped through the frame's
homography it should land on that marking; the distance to it, perpendicular to the line, is the
error in metres. Labels live in configs/overlay_labels.json so anyone can add their own.

    python scripts/landmark_error.py [--labels configs/overlay_labels.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from football_cv import config as C
from football_cv.calibration import homography as hg
from football_cv.calibration.motion import apply_h
from football_cv.calibration.projection import H_of

L, W = hg.L, hg.W
# line name -> (pitch axis it fixes, allowed values)
LINES = {
    "halfway": (0, (L / 2,)),
    "touch_near": (1, (0.0,)),
    "touch_far": (1, (W,)),
    "pen_front": (0, (hg.PEN_D, L - hg.PEN_D)),
}


def point_error_px(H: np.ndarray, line: str, u: float, v: float) -> float:
    """Signed pixels from the labelled point to the projected marking, perpendicular to it in the image."""
    axis, values = LINES[line]
    q = apply_h(H, np.array([[u, v]], dtype=float))[0]
    val = min(values, key=lambda x: abs(q[axis] - x))
    span = (0.0, W) if axis == 0 else (0.0, L)
    ends = np.array([[val, span[0]], [val, span[1]]] if axis == 0 else [[span[0], val], [span[1], val]])
    a, b = apply_h(np.linalg.inv(H), ends)
    d = (b - a) / np.linalg.norm(b - a)
    return float((np.array([u, v]) - a) @ np.array([-d[1], d[0]]))


def point_error(H: np.ndarray, line: str, u: float, v: float) -> float:
    """Signed metres from the labelled point, mapped to the pitch, to the nearest allowed line."""
    axis, values = LINES[line]
    q = apply_h(H, np.array([[u, v]], dtype=float))[0]
    return float(min((q[axis] - val for val in values), key=abs))


# The far touchline is read where the board hides it and a pixel there is ~0.5 m, so a few pixels of
# reading error swamp the result; it is left out of the figures unless asked for.
EXCLUDED_BY_DEFAULT = ("touch_far",)


def evaluate(labels: dict, calib: pd.DataFrame, exclude: tuple[str, ...] = EXCLUDED_BY_DEFAULT) -> pd.DataFrame:
    rows = []
    for fr in labels["frames"]:
        c = calib[calib.sample_id == fr["sample_id"]].iloc[0]
        H = H_of(c)
        pts = [p for p in fr["points"] if p[0] not in exclude]
        if not pts:
            continue
        errs = [point_error(H, ln, u, v) for ln, u, v in pts]
        px = [point_error_px(H, ln, u, v) for ln, u, v in pts]
        rows.append({
            "sample_id": fr["sample_id"], "timestamp_s": c.timestamp_s, "status": c.calibration_status,
            "keyframe_dt_s": abs(c.keyframe_dt_s), "n_points": len(errs),
            "mean_abs_m": float(np.mean(np.abs(errs))), "max_abs_m": float(np.max(np.abs(errs))),
            "mean_signed_m": float(np.mean(errs)),
            "mean_abs_px": float(np.mean(np.abs(px))), "max_abs_px": float(np.max(np.abs(px))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--labels", default="configs/overlay_labels.json")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    calib = pd.read_parquet(C.run_dir(cfg) / "frame_calibration.parquet")
    table = evaluate(json.loads(C.resolve(args.labels).read_text()), calib)
    print(table.round(2).to_string(index=False))
    out = C.artifacts_dir(cfg) / "calibration_validation"
    table.to_csv(out / "landmark_error.csv", index=False)
    for name, sub in (("keyframes (fitted on these lines)", table[table.status == "keyframe"]),
                      ("propagated (held out)", table[table.status == "propagated"])):
        n = sub.n_points.sum()
        print(f"{name}: {len(sub)} frames, {n} points, point-weighted mean |error| "
              f"{(sub.mean_abs_m * sub.n_points).sum() / n:.2f} m, median frame {sub.mean_abs_m.median():.2f} m, "
              f"worst frame {sub.mean_abs_m.max():.2f} m")


if __name__ == "__main__":
    main()
