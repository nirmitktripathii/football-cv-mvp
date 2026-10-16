"""Video probing and timestamp-preserving frame sampling.

Frames are decoded sequentially with OpenCV and a frame is kept when its own
presentation timestamp (CAP_PROP_POS_MSEC) reaches the next sample time. That
keeps the true source timestamp even for variable-frame-rate footage, instead
of assuming ``frame_index / fps``.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampledFrame:
    sample_id: int
    frame_id: int
    timestamp_s: float
    image: np.ndarray


def probe_video(path: Path) -> dict:
    """Container and stream metadata from ffprobe, with an OpenCV fallback."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    meta: dict = {"path": str(path), "size_bytes": Path(path).stat().st_size}
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True, timeout=60,
        )
        data = json.loads(out.stdout)
        stream = data["streams"][0]
        r_fps = float(Fraction(stream["r_frame_rate"]))
        avg_fps = float(Fraction(stream["avg_frame_rate"])) if stream["avg_frame_rate"] != "0/0" else r_fps
        meta.update(
            codec=stream.get("codec_name"),
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=avg_fps,
            r_frame_rate=r_fps,
            nb_frames=int(stream["nb_frames"]) if stream.get("nb_frames") else None,
            duration_s=float(data["format"]["duration"]),
            variable_frame_rate=abs(r_fps - avg_fps) > 0.01,
            probe="ffprobe",
        )
    except (FileNotFoundError, subprocess.SubprocessError, KeyError, ValueError) as exc:
        log.warning("ffprobe unavailable or failed (%s); falling back to OpenCV metadata", exc)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {path}") from exc
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta.update(
            codec=None,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps, r_frame_rate=fps, nb_frames=n,
            duration_s=n / fps if fps else None,
            variable_frame_rate=None, probe="opencv",
        )
        cap.release()
    return meta


def sample_frames(
    path: Path, fps: float, start_s: float = 0.0, end_s: float | None = None
) -> Iterator[SampledFrame]:
    """Yield frames at ``fps`` samples per second, in temporal order, with source timestamps.

    Decoding errors mid-stream are logged and end the iteration early rather
    than silently skipping a gap.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1.0 / fps
    next_t = start_s
    frame_id = -1
    sample_id = 0
    last_ts = -1.0
    try:
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
            frame_id = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        while True:
            if not cap.grab():
                break
            frame_id += 1
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0 and frame_id > 0:
                ts = frame_id / native_fps  # some backends report 0 for every frame
            if ts < last_ts:
                log.warning("Non-monotonic timestamp at frame %d (%.3f < %.3f)", frame_id, ts, last_ts)
            last_ts = ts
            if end_s is not None and ts > end_s:
                break
            # Half a source frame of tolerance so 5 fps on 30 fps lands on every 6th frame.
            if ts + 0.5 / native_fps < next_t:
                continue
            ok, image = cap.retrieve()
            if not ok:
                log.error("Decode failure at frame %d (t=%.3fs); stopping", frame_id, ts)
                break
            yield SampledFrame(sample_id, frame_id, round(ts, 4), image)
            sample_id += 1
            next_t += step
            while next_t + 0.5 / native_fps <= ts:  # catch up after a timestamp jump
                next_t += step
    finally:
        cap.release()


def extract_samples(
    video_path: Path,
    out_dir: Path,
    fps: float,
    start_s: float = 0.0,
    end_s: float | None = None,
    replay_offset_s: float = 0.0,
    kickoff_replay_s: float | None = None,
) -> pd.DataFrame:
    """Write sampled frames as JPEG and return the frame index table.

    Re-running with the same settings reuses the cached index when every frame
    file still exists.
    """
    frames_dir = out_dir / "frames"
    index_path = out_dir / "frames.parquet"
    settings_path = out_dir / "frames_settings.json"
    settings = {"video": str(video_path), "fps": fps, "start_s": start_s, "end_s": end_s}
    if index_path.exists() and settings_path.exists():
        cached = json.loads(settings_path.read_text(encoding="utf-8"))
        idx = pd.read_parquet(index_path)
        if cached == settings and all((out_dir / p).exists() for p in idx["path"]):
            log.info("Reusing %d cached sampled frames", len(idx))
            return idx

    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sf in sample_frames(video_path, fps, start_s, end_s):
        rel = f"frames/{sf.sample_id:06d}.jpg"
        cv2.imwrite(str(out_dir / rel), sf.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        replay_s = replay_offset_s + sf.timestamp_s
        rows.append(
            {
                "sample_id": sf.sample_id,
                "frame_id": sf.frame_id,
                "timestamp_s": sf.timestamp_s,
                "replay_s": round(replay_s, 4),
                "match_clock": match_clock(replay_s, kickoff_replay_s),
                "path": rel,
            }
        )
    idx = pd.DataFrame(rows)
    if idx.empty:
        raise RuntimeError(f"No frames decoded from {video_path}")
    idx.to_parquet(index_path, index=False)
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    log.info("Sampled %d frames at %.2f fps -> %s", len(idx), fps, frames_dir)
    return idx


def match_clock(replay_s: float, kickoff_replay_s: float | None) -> str | None:
    """On-screen-style match clock label (MM:SS) for a replay timestamp, if kickoff is known."""
    if kickoff_replay_s is None:
        return None
    t = max(0.0, replay_s - kickoff_replay_s)
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def contact_sheet(out_dir: Path, idx: pd.DataFrame, every_s: float, dest: Path, cols: int = 4) -> Path:
    """Grid of representative frames with their timestamps burned in."""
    picks = []
    next_t = idx["timestamp_s"].min()
    for row in idx.itertuples():
        if row.timestamp_s >= next_t:
            picks.append(row)
            next_t += every_s
    tiles = []
    for row in picks:
        img = cv2.imread(str(out_dir / row.path))
        img = cv2.resize(img, (480, 270))
        label = f"t={row.timestamp_s:.1f}s  frame={row.frame_id}  clock={row.match_clock}"
        cv2.rectangle(img, (0, 0), (480, 22), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), grid)
    return dest
