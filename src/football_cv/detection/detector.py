"""Detection of people and the ball on sampled frames with an Ultralytics model.

Two class sets are understood, chosen from the model's own class names:

- COCO (pretrained YOLO11): 0 = person (players, goalkeepers, referees, staff) and 32 = sports ball.
  The detector does not know who is a player, so ``role`` stays empty.
- Football 4-class (ball, player, goalkeeper, referee, e.g. a fine-tuned model): all three people classes
  become ``object_class`` "person" with their class in ``role``; one box per person (class-agnostic NMS).

Missed objects are never filled in: a frame with no ball detection simply
has no ball row, and coverage is reported downstream.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_BALL = 32
FOOTBALL_CLASSES = {"ball", "player", "goalkeeper", "referee"}

DETECTION_COLUMNS = [
    "sample_id", "frame_id", "timestamp_s", "det_id", "object_class", "role", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "model",
]


def class_roles(names: dict[int, str]) -> dict[int, tuple[str, str]]:
    """Map each model class index to ``(object_class, role)``; raise if the class set is not understood."""
    by_name = {str(v).lower(): int(k) for k, v in names.items()}
    if FOOTBALL_CLASSES <= set(by_name):
        return {
            by_name["ball"]: ("ball", ""),
            by_name["player"]: ("person", "player"),
            by_name["goalkeeper"]: ("person", "goalkeeper"),
            by_name["referee"]: ("person", "referee"),
        }
    if names.get(COCO_PERSON) == "person" and names.get(COCO_BALL) == "sports ball":
        return {COCO_PERSON: ("person", ""), COCO_BALL: ("ball", "")}
    raise ValueError(f"Unrecognised detector classes {dict(names)}; expected COCO or {sorted(FOOTBALL_CLASSES)}")


def detect_frames(
    frames: pd.DataFrame,
    frames_root: Path,
    model_name: str,
    imgsz: int,
    conf_person: float,
    conf_ball: float,
    device: int | str,
    half: bool,
    batch: int,
) -> pd.DataFrame:
    """Run the detector over every sampled frame listed in ``frames``."""
    from ultralytics import YOLO  # heavy import kept local so tests can import this module

    model = YOLO(model_name)
    roles = class_roles(model.names)
    football = any(role for _, role in roles.values())
    provenance = f"ultralytics:{Path(model_name).name}:imgsz{imgsz}"
    min_conf = min(conf_person, conf_ball)
    rows: list[dict] = []
    records = frames.to_dict("records")
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        results = model.predict(
            [str(frames_root / r["path"]) for r in chunk],
            imgsz=imgsz, conf=min_conf, classes=sorted(roles), agnostic_nms=football,
            device=device, half=half, verbose=False,
        )
        for rec, res in zip(chunk, results):
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            if not np.isfinite(xyxy).all():
                raise RuntimeError(
                    f"Non-finite boxes on {rec['path']}: half-precision inference is unreliable on some GPUs "
                    "(GTX 16xx); set detection.half=false"
                )
            det_id = 0
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, classes):
                object_class, role = roles[k]
                if c < (conf_person if object_class == "person" else conf_ball):
                    continue
                rows.append(
                    {
                        "sample_id": rec["sample_id"], "frame_id": rec["frame_id"],
                        "timestamp_s": rec["timestamp_s"], "det_id": det_id,
                        "object_class": object_class, "role": role, "confidence": float(c),
                        "bbox_x1": float(x1), "bbox_y1": float(y1),
                        "bbox_x2": float(x2), "bbox_y2": float(y2),
                        "model": provenance,
                    }
                )
                det_id += 1
        if (start // batch) % 25 == 0:
            log.info("Detected %d/%d frames", min(start + batch, len(records)), len(records))
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def detection_summary(dets: pd.DataFrame, n_frames: int) -> dict:
    """Per-class counts and the share of sampled frames with at least one detection."""
    out: dict = {"n_frames": int(n_frames)}
    for cls in ("person", "ball"):
        sub = dets[dets["object_class"] == cls]
        per_frame = sub.groupby("sample_id").size()
        out[cls] = {
            "detections": int(len(sub)),
            "frames_with_detection": int(per_frame.size),
            "frame_coverage": round(per_frame.size / n_frames, 4) if n_frames else 0.0,
            "mean_per_frame_when_present": round(float(per_frame.mean()), 2) if len(per_frame) else 0.0,
            "median_confidence": round(float(sub["confidence"].median()), 3) if len(sub) else None,
        }
    if "role" in dets and (dets["role"] != "").any():
        out["person"]["by_role"] = {str(k): int(v) for k, v in dets.loc[dets["object_class"] == "person", "role"].value_counts().items()}
    return out
