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


def frame_rows(
    rec: dict, xyxy: np.ndarray, confs: np.ndarray, classes: np.ndarray,
    roles: dict[int, tuple[str, str]], conf_person: float, conf_ball: float, provenance: str,
) -> list[dict]:
    """Detection rows of one frame from model output, dropping boxes below their object class's confidence floor."""
    if not np.isfinite(xyxy).all():
        raise RuntimeError(
            f"Non-finite boxes on {rec.get('path')}: half-precision inference is unreliable on some GPUs "
            "(GTX 16xx); set detection.half=false"
        )
    rows: list[dict] = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, classes):
        object_class, role = roles[int(k)]
        if c < (conf_person if object_class == "person" else conf_ball):
            continue
        rows.append(
            {
                "sample_id": rec["sample_id"], "frame_id": rec["frame_id"],
                "timestamp_s": rec["timestamp_s"], "det_id": len(rows),
                "object_class": object_class, "role": role, "confidence": float(c),
                "bbox_x1": float(x1), "bbox_y1": float(y1), "bbox_x2": float(x2), "bbox_y2": float(y2),
                "model": provenance,
            }
        )
    return rows


def _ultralytics_predictor(model_name: str, imgsz: int, min_conf: float, device, half: bool):
    """``(predict(paths), roles, provenance)`` for an Ultralytics model."""
    from ultralytics import YOLO  # heavy import kept local so tests can import this module

    model = YOLO(model_name)
    roles = class_roles(model.names)
    football = any(role for _, role in roles.values())

    def predict(paths: list[str]):
        results = model.predict(
            paths, imgsz=imgsz, conf=min_conf, classes=sorted(roles), agnostic_nms=football,
            device=device, half=half, verbose=False,
        )
        return [
            (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))
            for r in results
        ]

    return predict, roles, f"ultralytics:{Path(model_name).name}:imgsz{imgsz}"


def _yolox_predictor(weights: str, yolox_repo: Path, test_size: tuple[int, int], min_conf: float, device, half: bool):
    """``(predict(paths), roles, provenance)`` for the fine-tuned YOLOX-m checkpoint (see ``yolox_backend``)."""
    from football_cv.detection import yolox_backend  # imports torch and the YOLOX checkout only when used

    if half:
        raise ValueError("the YOLOX backend runs in FP32; set detection.half=false")
    predictor = yolox_backend.YoloxPredictor(Path(weights), yolox_repo, device, test_size, conf_floor=min_conf)
    return predictor.predict, class_roles(yolox_backend.CLASS_NAMES), predictor.provenance


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
    backend: str = "ultralytics",
    yolox_repo: Path | None = None,
    test_size: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Run the detector over every sampled frame listed in ``frames``.

    ``backend`` is "ultralytics" (``model_name`` is a weights file or name, ``imgsz`` the input size) or "yolox"
    (``model_name`` is a checkpoint path, ``yolox_repo`` the pinned YOLOX checkout, ``test_size`` the input ``(h, w)``).
    """
    min_conf = min(conf_person, conf_ball)
    if backend == "ultralytics":
        predict, roles, provenance = _ultralytics_predictor(model_name, imgsz, min_conf, device, half)
    elif backend == "yolox":
        if yolox_repo is None or test_size is None:
            raise ValueError("the yolox backend needs yolox_repo and test_size")
        predict, roles, provenance = _yolox_predictor(model_name, Path(yolox_repo), test_size, min_conf, device, half)
    else:
        raise ValueError(f"unknown detection backend {backend!r}; expected 'ultralytics' or 'yolox'")
    rows: list[dict] = []
    records = frames.to_dict("records")
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        for rec, (xyxy, confs, classes) in zip(chunk, predict([str(frames_root / r["path"]) for r in chunk])):
            rows += frame_rows(rec, xyxy, confs, classes, roles, conf_person, conf_ball, provenance)
        if (start // batch) % 25 == 0:
            log.info("Detected %d/%d frames", min(start + batch, len(records)), len(records))
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def merge_ball_source(people: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """People from one detector and the ball from another, with ``det_id`` renumbered within each sample.

    The ``model`` column keeps each row's own provenance, so a mixed table still says where every box came from.
    """
    out = pd.concat([people[people["object_class"] == "person"], balls[balls["object_class"] == "ball"]],
                    ignore_index=True)
    out = out.sort_values(["sample_id", "object_class", "det_id"], ascending=[True, False, True], kind="stable")
    out["det_id"] = out.groupby("sample_id").cumcount()
    return out[DETECTION_COLUMNS].reset_index(drop=True)


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
