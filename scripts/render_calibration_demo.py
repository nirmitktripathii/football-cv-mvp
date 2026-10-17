"""Render a 1-2 minute broadcast excerpt that shows how the pitch mapping is used and how good it is.

    python scripts/render_calibration_demo.py [--start 49] [--end 149] [--out docs/media/calibration_demo.mp4]

Per frame: player boxes and track ids, the pitch boundary (thick yellow), inner markings (red), the projected
landmark fix points (yellow dots), the pitch mini-map, and a status bar with the calibration state, seconds from
the nearest manual keyframe and the measured mapping error. Manual keyframes are held for about 1.6 s and show the
hand-clicked points (green) against where the model projects them (red). Frames without a calibrated view get
no overlay and say so. The error figures are the hand-checked ones in docs/evaluation.md section 10.
"""
from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np
import pandas as pd

from football_cv import config as C
from football_cv.calibration.homography import apply_h, landmarks, load_keyframes, pitch_line_samples
from football_cv.calibration.projection import H_of
from football_cv.visualization import broadcast as B

log = logging.getLogger("calibration_demo")
BOUNDARY = (0, 215, 255)      # BGR yellow: touchlines and goal lines
MARKINGS = (80, 80, 255)      # BGR red: inner markings
FIX_DOT, CLICK, PROJ = (0, 255, 255), (0, 220, 0), (60, 60, 255)
KEYFRAME_HOLD = 8             # repeated frames (1.6 s at 5 fps) so a keyframe can be read
CRF = 28
KEYFRAME_ERR_M, PROPAGATED_MEAN_M, PROPAGATED_WORST_M = 0.18, 4.1, 8.1   # docs/evaluation.md section 10
BAR_H = 48
LABELS = (("corner", "corner"), ("halfway", "halfway"), ("centre_spot", "centre"), ("circle", "circle"),
          ("_goalline", "box"), ("_corner", "box"), ("_arc", "D"), ("_spot", "pen spot"))


def short_label(name: str) -> str:
    if name.startswith("ga_"):
        return "6yd"
    return next((lab for key, lab in LABELS if key in name), name)


def _in_frame(pts: np.ndarray, w: int, h: int, margin: int = 8) -> np.ndarray:
    return (np.isfinite(pts).all(axis=1) & (pts[:, 0] >= margin) & (pts[:, 0] < w - margin)
            & (pts[:, 1] >= margin) & (pts[:, 1] < h - BAR_H - margin))


def _project(Hinv: np.ndarray, pitch: np.ndarray, w: int, h: int) -> np.ndarray:
    px = apply_h(Hinv, pitch)
    px[~(np.abs(px) < 4 * max(w, h)).all(axis=1)] = np.nan
    return px


def _text(img: np.ndarray, s: str, org: tuple[int, int], colour, scale: float = 0.4) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_mapping(img: np.ndarray, H: np.ndarray) -> None:
    """Pitch markings, boundary and landmark fix points projected through ``H`` (image -> pitch)."""
    h, w = img.shape[:2]
    Hinv = np.linalg.inv(H)
    for k, poly in enumerate(pitch_line_samples()):
        px = _project(Hinv, poly, w, h)
        px = px[np.isfinite(px).all(axis=1)]
        if len(px) > 1:
            boundary = k < 4
            cv2.polylines(img, [px.astype(np.int32).reshape(-1, 1, 2)], False, BOUNDARY if boundary else MARKINGS,
                          3 if boundary else 1, cv2.LINE_AA)
    marks = landmarks()
    names = list(marks)
    px = _project(Hinv, np.array([marks[n] for n in names]), w, h)
    for n, p in zip(names, px):
        if _in_frame(p[None], w, h)[0]:
            c = (int(p[0]), int(p[1]))
            cv2.circle(img, c, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, 4, FIX_DOT, -1, cv2.LINE_AA)
            _text(img, short_label(n), (c[0] + 7, c[1] - 6), FIX_DOT, 0.38)


def draw_keyframe_check(img: np.ndarray, H: np.ndarray, kf) -> None:
    """Hand-clicked landmarks (green +) against the model's projection of the same landmarks (red o)."""
    Hinv = np.linalg.inv(H)
    proj = apply_h(Hinv, kf.pitch_pts)
    for name, clicked, p in zip(kf.names, kf.image_pts, proj):
        c, q = (int(clicked[0]), int(clicked[1])), (int(p[0]), int(p[1]))
        cv2.line(img, c, q, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(img, c, CLICK, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.circle(img, q, 7, PROJ, 2, cv2.LINE_AA)
        _text(img, short_label(name), (c[0] + 9, c[1] + 14), CLICK, 0.4)
    for clicked in kf.line_img:
        cv2.drawMarker(img, (int(clicked[0]), int(clicked[1])), CLICK, cv2.MARKER_DIAMOND, 10, 1, cv2.LINE_AA)


def draw_legend(img: np.ndarray, keyframe: bool) -> None:
    items = [(BOUNDARY, "pitch boundary"), (MARKINGS, "pitch markings"), (FIX_DOT, "projected fix points"),
             (B.BGR["home"], "Bengaluru FC"), (B.BGR["away"], "Sporting Club Delhi"), (B.BGR["other"], "referee / keeper")]
    if keyframe:
        items += [(CLICK, "+ hand-clicked landmark"), (PROJ, "o model projection")]
    w = img.shape[1]
    x0, y0 = w - 232, 10
    cv2.rectangle(img, (x0 - 8, y0 - 4), (w - 6, y0 + 20 * len(items) + 2), (0, 0, 0), -1)
    for i, (colour, text) in enumerate(items):
        cv2.rectangle(img, (x0, y0 + 20 * i + 3), (x0 + 14, y0 + 20 * i + 15), colour, -1)
        cv2.putText(img, text, (x0 + 22, y0 + 20 * i + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def status_text(row, keyframe: bool) -> tuple[str, str]:
    status = row.calibration_status
    if status == "keyframe" or keyframe:
        return "MANUAL KEYFRAME", f"mapping error at held-out landmarks ~{KEYFRAME_ERR_M} m"
    if status == "propagated":
        return (f"propagated, {abs(row.keyframe_dt_s):.1f} s from keyframe {int(row.keyframe_id)}",
                f"mapping error on propagated frames: mean ~{PROPAGATED_MEAN_M:g} m, worst measured {PROPAGATED_WORST_M:g} m")
    return f"NO CALIBRATED VIEW ({str(status).replace('_', ' ')})", "no overlay drawn: nothing is projected without a valid mapping"


def draw_bar(img: np.ndarray, header: str, line1: str, line2: str, calibrated: bool) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - BAR_H), (w, h), (0, 0, 0), -1)
    cv2.putText(img, f"{header}   {line1}", (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255) if calibrated else (80, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(img, line2, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)


def build_tracks(run) -> pd.DataFrame:
    tracks = pd.read_parquet(run / "tracks.parquet")
    traj = pd.read_parquet(run / "trajectories.parquet")
    keys = ["sample_id", "track_id"]
    out = tracks.merge(traj[keys + ["x_s", "y_s", "jump", "short_track"]], on=keys, how="left")
    hide = out["jump"].fillna(False).astype(bool) | out["short_track"].fillna(False).astype(bool)
    out.loc[hide, ["x_s", "y_s"]] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--start", type=float, default=49.0)
    ap.add_argument("--end", type=float, default=149.0)
    ap.add_argument("--out", default="docs/media/calibration_demo.mp4")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = C.load_config(args.config)
    run = C.run_dir(cfg)
    frames = pd.read_parquet(run / "frames.parquet")
    frames = frames[(frames.timestamp_s >= args.start) & (frames.timestamp_s <= args.end)]
    calib = pd.read_parquet(run / "frame_calibration.parquet").set_index("sample_id")
    by_sample = {k: g for k, g in build_tracks(run).groupby("sample_id")}
    keyframes = {k.sample_id: k for k in load_keyframes(C.resolve(cfg["calibration"]["keyframes_file"]))}
    base = B._inset_base()
    fps = float(cfg["sampling"]["fps"])
    writer, n_key, n_cal = None, 0, 0
    for r in frames.itertuples():
        img = cv2.imread(str(run / r.path))
        h, w = img.shape[:2]
        if writer is None:
            writer = B.FfmpegWriter(C.resolve(args.out), w, h, fps, crf=CRF)
        row = calib.loc[r.sample_id] if r.sample_id in calib.index else None
        is_cal = row is not None and row.calibration_status in ("keyframe", "propagated")
        is_key = is_cal and r.sample_id in keyframes
        out = img.copy()
        rows = by_sample.get(r.sample_id, pd.DataFrame())
        if is_cal:
            H = H_of(row)
            draw_mapping(out, H)
            if is_key:
                draw_keyframe_check(out, H, keyframes[r.sample_id])
        if len(rows):
            B.draw_boxes(out, rows)
            if is_cal:
                B._draw_inset(out, base, rows)
        draw_legend(out, is_key)
        info = row if row is not None else pd.Series({"calibration_status": "not_wide_view"})
        line1, line2 = status_text(info, is_key)
        clock = f"match {r.match_clock}" if getattr(r, "match_clock", None) else ""
        draw_bar(out, f"t={r.timestamp_s:.1f}s  {clock}", line1, line2, is_cal)
        for _ in range(KEYFRAME_HOLD if is_key else 1):
            writer.write(out)
        n_key += int(is_key)
        n_cal += int(is_cal)
    writer.close()
    log.info("%d frames, %d calibrated, %d manual keyframes -> %s", len(frames), n_cal, n_key, args.out)


if __name__ == "__main__":
    main()
