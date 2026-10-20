"""Multi-object tracking of on-pitch people with ByteTrack in camera-stabilized coordinates.

Broadcast pans move every player box by tens of pixels between 5 fps samples,
which breaks IoU association. Each box is therefore first mapped into its
motion segment's reference frame with ``G_t`` (see ``calibration.motion``), so
the tracker only sees the players' own movement.

The tracker is reset at every motion segment (shot cut or failed motion
estimate) and track ids are made globally unique, so an id never spans two
camera views. Re-identification across cuts is out of scope and documented.

The ball is not tracked by ByteTrack: per frame the single highest-confidence
on-pitch ball detection is kept (``track_id = -1``). Missing ball frames stay
missing.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from football_cv.calibration.motion import G_of, apply_h

log = logging.getLogger(__name__)

BALL_TRACK_ID = -1
TRACK_COLUMNS = [
    "sample_id", "frame_id", "timestamp_s", "shot_id", "segment_id", "track_id", "object_class", "role",
    "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "foot_x", "foot_y", "stab_x", "stab_y",
]


def stabilize_boxes(boxes: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Axis-aligned boxes after mapping all four corners of each box with ``G``."""
    if len(boxes) == 0:
        return np.empty((0, 4))
    x1, y1, x2, y2 = boxes.T
    corners = np.stack([np.c_[x1, y1], np.c_[x2, y1], np.c_[x1, y2], np.c_[x2, y2]], axis=1)
    mapped = apply_h(G, corners.reshape(-1, 2)).reshape(-1, 4, 2)
    return np.c_[mapped[..., 0].min(1), mapped[..., 1].min(1), mapped[..., 0].max(1), mapped[..., 1].max(1)]


def _new_tracker(params: dict, fps: float):
    import supervision as sv

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # ByteTrack deprecated in supervision 0.28, pinned <0.31
        # supervision computes max_time_lost = frame_rate / 30 * lost_track_buffer
        return sv.ByteTrack(
            track_activation_threshold=float(params["track_activation_threshold"]),
            lost_track_buffer=int(round(float(params["lost_track_buffer_s"]) * 30)),
            minimum_matching_threshold=float(params["minimum_matching_threshold"]),
            frame_rate=int(round(fps)),
            minimum_consecutive_frames=int(params["minimum_consecutive_frames"]),
        )


def _update(tracker, ppl: pd.DataFrame, G: np.ndarray):
    import supervision as sv

    boxes = ppl[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=float)
    det = sv.Detections(
        xyxy=stabilize_boxes(boxes, G),
        confidence=ppl["confidence"].to_numpy(dtype=float),
        class_id=np.zeros(len(ppl), dtype=int),
        data={"row": ppl.index.to_numpy()},
    )
    return tracker.update_with_detections(det)


def _rows(dets: pd.DataFrame, track_ids, G: np.ndarray, m) -> list[dict]:
    feet = np.c_[(dets["bbox_x1"] + dets["bbox_x2"]) / 2, dets["bbox_y2"]]
    stab = apply_h(G, feet)
    out = []
    for d, tid, (fx, fy), (sx, sy) in zip(dets.itertuples(), track_ids, feet, stab):
        out.append(
            {
                "sample_id": d.sample_id, "frame_id": d.frame_id, "timestamp_s": d.timestamp_s,
                "shot_id": m.shot_id, "segment_id": m.segment_id, "track_id": int(tid),
                "object_class": d.object_class, "role": getattr(d, "role", "") or "", "confidence": d.confidence,
                "bbox_x1": d.bbox_x1, "bbox_y1": d.bbox_y1, "bbox_x2": d.bbox_x2, "bbox_y2": d.bbox_y2,
                "foot_x": fx, "foot_y": fy, "stab_x": sx, "stab_y": sy,
            }
        )
    return out


def track(detections: pd.DataFrame, motion: pd.DataFrame, params: dict, fps: float) -> pd.DataFrame:
    """Track on-pitch people per motion segment; return one row per tracked detection."""
    on = detections[detections["on_pitch"]]
    people = {k: g for k, g in on[on["object_class"] == "person"].groupby("sample_id")}
    balls = {k: g.nlargest(1, "confidence") for k, g in on[on["object_class"] == "ball"].groupby("sample_id")}
    rows: list[dict] = []
    id_offset = 0
    for _, seg in motion.groupby("segment_id", sort=True):
        tracker = _new_tracker(params, fps)
        seg_max = 0
        for m in seg.sort_values("timestamp_s").itertuples():
            G = G_of(m)
            ppl = people.get(m.sample_id)
            if ppl is not None and len(ppl):
                result = _update(tracker, ppl, G)
                if len(result):
                    tracked = ppl.loc[result.data["row"]]
                    rows += _rows(tracked, result.tracker_id + id_offset, G, m)
                    seg_max = max(seg_max, int(result.tracker_id.max()))
            ball = balls.get(m.sample_id)
            if ball is not None:
                rows += _rows(ball, [BALL_TRACK_ID], G, m)
        id_offset += seg_max
    out = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    log.info(
        "Tracking: %d person rows in %d tracks over %d segments; ball in %d frames",
        int((out["object_class"] == "person").sum()), out.loc[out["track_id"] >= 0, "track_id"].nunique(),
        motion["segment_id"].nunique(), int((out["object_class"] == "ball").sum()),
    )
    return out
