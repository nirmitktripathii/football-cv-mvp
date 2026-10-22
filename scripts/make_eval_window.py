"""Run the tracker on one evaluation window and export a CVAT pre-fill for hand correction (phase 0).

    python scripts/make_eval_window.py --config configs/inter_napoli.yaml --window w1_open_t600
    python scripts/make_eval_window.py --config configs/inter_napoli.yaml --all

Windows are listed under ``evaluation.windows`` in the config. Each window runs as its own clip,
``<clip.id>_<window>``, through ingest, detect, shots, motion and track in one process. It writes:

- ``data/processed/<clip>/cvat_frames.zip``: the sampled frames, ``000000.jpg`` ..., so CVAT frame = sample_id;
- ``data/annotations/<clip>/prefill_cvat.xml``: tracker output as CVAT for video 1.1, to import and correct;
- ``data/annotations/<clip>/labels.json``: the label set, for CVAT's Raw label editor;
- ``data/annotations/<clip>/window.json``: what was run, so the corrected file can be scored later.

The corrected export is saved by hand as ``data/annotations/<clip>/annotations.xml``. This script never
writes that file. See docs/annotation_guide.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from football_cv import config as C
from football_cv import pipeline
from football_cv.evaluation import cvat

log = logging.getLogger("make_eval_window")
STAGES = ("ingest", "detect", "shots", "motion", "track")


def window_config(config_path: str, window: str, extra: list[str]) -> dict:
    base = C.load_config(config_path)
    spec = base.get("evaluation", {}).get("windows", {}).get(window)
    if spec is None:
        raise SystemExit(f"window {window!r} is not under evaluation.windows in {config_path}")
    if any(e.replace(" ", "").startswith("clip.id=") for e in extra):
        raise SystemExit("do not override clip.id: each window runs as <clip.id>_<window>; set clip.id in the config instead")
    sets = [f"clip.id={base['clip']['id']}_{window}", f"sampling.start_s={spec['start_s']}",
            f"sampling.end_s={spec['end_s']}", *extra]
    return C.load_config(config_path, sets)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def zip_frames(run: Path, frames: pd.DataFrame) -> Path:
    dest = run / "cvat_frames.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as zf:
        for row in frames.itertuples():
            zf.write(run / row.path, arcname=f"{row.sample_id:06d}.jpg")
    return dest


def export(cfg: dict, window: str, spec: dict) -> dict:
    run = C.run_dir(cfg)
    frames = pd.read_parquet(run / "frames.parquet").sort_values("sample_id", ignore_index=True)
    tracks = pd.read_parquet(run / "tracks.parquet")
    n = len(frames)
    if frames["sample_id"].tolist() != list(range(n)):
        raise RuntimeError(f"sample ids in {run / 'frames.parquet'} are not 0..{n - 1}; CVAT frames would not line up")
    h, w = cv2.imread(str(run / frames["path"].iloc[0])).shape[:2]
    clip = cfg["clip"]["id"]
    ann = C.resolve("data/annotations") / clip
    ann.mkdir(parents=True, exist_ok=True)
    (ann / "prefill_cvat.xml").write_text(cvat.to_cvat_video_xml(tracks, n, w, h, clip), encoding="utf-8")
    C.write_json(ann / "labels.json", cvat.label_spec())
    people = tracks[(tracks["object_class"] == cvat.PERSON) & (tracks["track_id"] >= 0)]
    meta = {
        "clip_id": clip, "window": window, "note": spec.get("note"), "video_path": cfg["clip"]["video_path"],
        "start_s": spec["start_s"], "end_s": spec["end_s"], "sampling_fps": cfg["sampling"]["fps"],
        "num_frames": n, "frame_size": [w, h],
        "first_timestamp_s": float(frames["timestamp_s"].iloc[0]),
        "last_timestamp_s": float(frames["timestamp_s"].iloc[-1]),
        "prefill": {"person_tracks": int(people["track_id"].nunique()), "person_boxes": int(len(people)),
                    "ball_boxes": int((tracks["object_class"] == cvat.BALL).sum())},
        "frames_zip": zip_frames(run, frames).relative_to(C.REPO_ROOT).as_posix(),
        "git_commit": git_commit(), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    C.write_json(ann / "window.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/inter_napoli.yaml")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--window", action="append", help="window name; repeatable")
    group.add_argument("--all", action="store_true", help="every window in the config")
    ap.add_argument("--export-only", action="store_true", help="skip the pipeline, re-export existing tracks")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    windows = list(C.load_config(args.config)["evaluation"]["windows"]) if args.all else args.window
    for window in windows:
        cfg = window_config(args.config, window, args.overrides)
        C.setup_logging(C.run_dir(cfg) / "logs" / "pipeline.log")
        for stage in () if args.export_only else STAGES:
            t0 = time.perf_counter()
            log.info("=== %s: stage %s ===", window, stage)
            pipeline.STAGES[stage](cfg)
            log.info("=== %s: stage %s done in %.1fs ===", window, stage, time.perf_counter() - t0)
        meta = export(cfg, window, cfg["evaluation"]["windows"][window])
        log.info("%s: %d frames, pre-fill %s", window, meta["num_frames"], json.dumps(meta["prefill"]))


if __name__ == "__main__":
    main()
