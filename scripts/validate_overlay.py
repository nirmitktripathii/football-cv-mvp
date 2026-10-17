"""Overlay validation of the pitch mapping on frames that were not used to fit it.

For every calibrated frame the reference pitch lines are projected into the image with that frame's
homography and compared with the painted-line pixels found in the frame (white, thin, low-saturation
structures on grass). Keyframes were fitted on those lines, so they are reported separately from
propagated frames, which are the held-out test. Output: a CSV of per-frame figures, a JSON summary
by distance from the nearest keyframe, and contact sheets of overlays for visual review.

    python scripts/validate_overlay.py [--config configs/default.yaml] [--sheet-frames 12]

The line mask is a detector, not ground truth: ad boards and shadows add false line pixels, so the
distance is a lower bound on the true error where a wrong line sits next to a real one.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_cv import config as C
from football_cv.calibration import homography as hg
from football_cv.calibration.projection import H_COLS, H_of
from football_cv.calibration.shots import grass_mask

log = logging.getLogger("validate_overlay")

CALIBRATED = ("keyframe", "propagated")
INLIER_PX = 12.0
DT_BINS_S = (0.0, 2.0, 4.0, 8.0, 12.01)
MIN_VISIBLE_SAMPLES = 50


def metres_per_pixel(H_img2pitch: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Local pitch metres covered by one image pixel at each point (mean of the two axes)."""
    here = hg.apply_h(H_img2pitch, pts)
    dx = hg.apply_h(H_img2pitch, pts + [1.0, 0.0])
    dy = hg.apply_h(H_img2pitch, pts + [0.0, 1.0])
    return 0.5 * (np.linalg.norm(dx - here, axis=1) + np.linalg.norm(dy - here, axis=1))


def frame_figures(image: np.ndarray, H: np.ndarray) -> dict:
    """Distance from projected pitch lines to the nearest painted-line pixel, in px and metres."""
    cost = hg.LineCost(image, grass_mask(image), truncate_px=INLIER_PX)
    pts = cost.visible(H)
    if len(pts) < MIN_VISIBLE_SAMPLES:
        return {"n_visible": len(pts), "inlier_frac": np.nan, "median_px": np.nan, "median_m": np.nan}
    d = cost.dist[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    near = d <= INLIER_PX
    scale = metres_per_pixel(H, pts)
    return {
        "n_visible": len(pts),
        "inlier_frac": float(near.mean()),
        "median_px": float(np.median(d[near])) if near.any() else np.nan,
        "median_m": float(np.median(d[near] * scale[near])) if near.any() else np.nan,
    }


def summarise(table: pd.DataFrame) -> dict:
    """Per-group medians; propagated frames binned by seconds from their keyframe."""
    def stats(df: pd.DataFrame) -> dict:
        return {
            "frames": int(len(df)),
            "median_px": float(df["median_px"].median()),
            "p90_px": float(df["median_px"].quantile(0.9)),
            "median_m": float(df["median_m"].median()),
            "p90_m": float(df["median_m"].quantile(0.9)),
            "median_inlier_frac": float(df["inlier_frac"].median()),
        }

    ok = table.dropna(subset=["median_px"])
    out = {"keyframes_in_sample": stats(ok[ok.calibration_status == "keyframe"])}
    prop = ok[ok.calibration_status == "propagated"]
    for lo, hi in zip(DT_BINS_S[:-1], DT_BINS_S[1:]):
        sub = prop[(prop.keyframe_dt_s >= lo) & (prop.keyframe_dt_s < hi)]
        if len(sub):
            out[f"propagated_{lo:g}-{min(hi, 12):g}s"] = stats(sub)
    out["propagated_all"] = stats(prop)
    return out


def contact_sheet(rows: pd.DataFrame, frames_dir: Path, out_path: Path, cols: int = 3) -> None:
    tiles = []
    for _, r in rows.iterrows():
        img = cv2.imread(str(frames_dir / f"{int(r.sample_id):06d}.jpg"))
        img = hg.draw_pitch_overlay(img, H_of(r), colour=(0, 0, 255), thickness=2)
        label = f"t={r.timestamp_s:.1f}s {r.calibration_status} dt={r.keyframe_dt_s:.1f}s err~{r.median_m:.2f}m"
        cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (640, 360)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(grid))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sheet-frames", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = C.load_config(args.config)
    processed = C.run_dir(cfg)
    calib = pd.read_parquet(processed / "frame_calibration.parquet")
    calib = calib[calib.calibration_status.isin(CALIBRATED)].reset_index(drop=True)
    frames_dir = processed / "frames"

    rows = []
    for _, r in calib.iterrows():
        img = cv2.imread(str(frames_dir / f"{int(r.sample_id):06d}.jpg"))
        rows.append({**r.to_dict(), **frame_figures(img, H_of(r))})
    table = pd.DataFrame(rows)

    out_dir = C.artifacts_dir(cfg) / "calibration_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.drop(columns=H_COLS).to_csv(out_dir / "overlay_error.csv", index=False)
    summary = summarise(table)
    (out_dir / "overlay_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(json.dumps(summary, indent=2))

    # Contact sheets: propagated frames spread evenly over keyframe distance, and the worst six.
    prop = table[table.calibration_status == "propagated"].dropna(subset=["median_m"]).sort_values("keyframe_dt_s")
    pick = prop.iloc[np.linspace(0, len(prop) - 1, args.sheet_frames).astype(int)]
    contact_sheet(pick, frames_dir, out_dir / "overlay_sheet.jpg")
    contact_sheet(prop.sort_values("median_m", ascending=False).head(6), frames_dir, out_dir / "overlay_worst.jpg")
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
