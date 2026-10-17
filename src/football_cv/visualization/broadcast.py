"""Annotated broadcast video: boxes, track ids, team colours, ball, timestamps and a pitch inset.

Frames are piped to ffmpeg as H.264 (yuv420p) so the result plays in browsers
and in the Streamlit dashboard. Only data that exists is drawn: a frame with no
tracks shows the raw broadcast plus its shot / calibration status.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_cv.calibration.homography import draw_pitch_overlay, pitch_line_samples
from football_cv.calibration.projection import H_of

log = logging.getLogger(__name__)

BGR = {
    "home": (191, 95, 31), "away": (244, 244, 244), "other": (0, 196, 245),
    "unknown": (154, 154, 154), "ball": (14, 127, 255),
}
PX_PER_M = 3
INSET_W, INSET_H, INSET_PAD = 105 * PX_PER_M, 68 * PX_PER_M, 10


class FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 23):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not found on PATH")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding")


def _inset_base() -> np.ndarray:
    img = np.full((INSET_H, INSET_W, 3), (50, 125, 46), np.uint8)
    for poly in pitch_line_samples():
        pts = np.c_[poly[:, 0] * PX_PER_M, (68 - poly[:, 1]) * PX_PER_M].astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _draw_inset(frame: np.ndarray, base: np.ndarray, rows: pd.DataFrame) -> None:
    inset = base.copy()
    for r in rows.itertuples():
        if not np.isfinite(r.x_s):
            continue
        colour = BGR["ball"] if r.track_id < 0 else BGR.get(r.team_id, BGR["unknown"])
        radius = 3 if r.track_id < 0 else 4
        p = (int(r.x_s * PX_PER_M), int((68 - r.y_s) * PX_PER_M))
        cv2.circle(inset, p, radius, colour, -1, cv2.LINE_AA)
        cv2.circle(inset, p, radius, (0, 0, 0), 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    y0, x0 = h - INSET_H - INSET_PAD - 26, w - INSET_W - INSET_PAD
    region = frame[y0 : y0 + INSET_H, x0 : x0 + INSET_W]
    frame[y0 : y0 + INSET_H, x0 : x0 + INSET_W] = cv2.addWeighted(region, 0.15, inset, 0.85, 0)


def draw_boxes(out: np.ndarray, rows: pd.DataFrame) -> None:
    """Player boxes with track ids and the ball marker, drawn in place."""
    for r in rows.itertuples():
        colour = BGR["ball"] if r.track_id < 0 else BGR.get(r.team_id, BGR["unknown"])
        p1, p2 = (int(r.bbox_x1), int(r.bbox_y1)), (int(r.bbox_x2), int(r.bbox_y2))
        if r.track_id < 0:
            centre = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.circle(out, centre, max(6, p2[0] - p1[0]), colour, 2, cv2.LINE_AA)
            continue
        cv2.rectangle(out, p1, p2, colour, 2)
        cv2.putText(out, str(r.track_id), (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def annotate_frame(img: np.ndarray, rows: pd.DataFrame, header: str, status: str, H=None, lines: bool = False) -> np.ndarray:
    out = draw_pitch_overlay(img, H, colour=(80, 80, 255), thickness=1) if (lines and H is not None) else img.copy()
    draw_boxes(out, rows)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(out, f"{header}   calibration: {status}", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_broadcast_video(
    traj_all: pd.DataFrame, frames: pd.DataFrame, frames_root: Path, calib: pd.DataFrame, out_path: Path,
    start_s: float | None = None, end_s: float | None = None, fps: float = 5.0, draw_lines: bool = True,
) -> Path:
    """``traj_all``: every tracked row with bbox columns plus cleaned ``x_s, y_s`` (NaN when uncalibrated)."""
    fr = frames
    if start_s is not None:
        fr = fr[fr["timestamp_s"] >= start_s]
    if end_s is not None:
        fr = fr[fr["timestamp_s"] <= end_s]
    by_sample = {k: g for k, g in traj_all.groupby("sample_id")}
    empty = traj_all.iloc[0:0]
    cal = calib.set_index("sample_id")
    base = _inset_base()
    writer = None
    for r in fr.itertuples():
        img = cv2.imread(str(frames_root / r.path))
        if writer is None:
            writer = FfmpegWriter(out_path, img.shape[1], img.shape[0], fps)
        rows = by_sample.get(r.sample_id, empty)
        status = cal.at[r.sample_id, "calibration_status"] if r.sample_id in cal.index else "not wide view"
        H = H_of(cal.loc[r.sample_id]) if status in ("keyframe", "propagated") else None
        clock = f"  match {r.match_clock}" if getattr(r, "match_clock", None) else ""
        out = annotate_frame(img, rows, f"t={r.timestamp_s:.1f}s  frame {r.frame_id}{clock}", status, H, draw_lines)
        if H is not None:
            _draw_inset(out, base, rows)
        writer.write(out)
    if writer is not None:
        writer.close()
    log.info("Annotated broadcast video (%d frames) -> %s", len(fr), out_path)
    return Path(out_path)
