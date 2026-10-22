"""Multi-object tracking of on-pitch people with ByteTrack in camera-stabilized coordinates.

Broadcast pans move every player box by tens of pixels between 5 fps samples,
which breaks IoU association. Each box is therefore first mapped into its
motion segment's reference frame with ``G_t`` (see ``calibration.motion``), so
the tracker only sees the players' own movement.

The tracker is reset at every motion segment (shot cut or failed motion
estimate) and track ids are made globally unique, so an id never spans two
camera views. Re-identification across cuts is out of scope and documented.

The ball is not tracked by ByteTrack: per frame the single highest-confidence
on-pitch ball detection is kept (``track_id = -1``), after dropping detections
that sit in the same stabilised spot in several nearby samples (spare balls,
the penalty spot, line marks: see ``static_ball_mask``). Missing ball frames
stay missing; nothing is interpolated.
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


def buffer_boxes(boxes: np.ndarray, buffer: float) -> np.ndarray:
    """Boxes grown by ``buffer`` times their width and height on every side (C-BIoU matching space).

    At 5 fps a running player's box can move more than its own width between samples, so plain IoU with the
    predicted box is often zero; growing both boxes for matching only keeps such pairs comparable.
    """
    if buffer <= 0 or len(boxes) == 0:
        return boxes
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    return np.c_[boxes[:, 0] - buffer * w, boxes[:, 1] - buffer * h, boxes[:, 2] + buffer * w, boxes[:, 3] + buffer * h]


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


def _update(tracker, ppl: pd.DataFrame, G: np.ndarray, buffer: float = 0.0):
    import supervision as sv

    boxes = ppl[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=float)
    det = sv.Detections(
        xyxy=buffer_boxes(stabilize_boxes(boxes, G), buffer),
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


RECOVER_DEFAULTS = {"enabled": False, "iou": 0.3, "keep_confidence": 0.5, "duplicate_iou": 0.7}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    import supervision as sv

    return sv.box_iou_batch(a, b)


def _greedy_pairs(iou: np.ndarray, iou_min: float) -> list[tuple[int, int]]:
    """One-to-one (row, column) pairs by descending IoU, at least ``iou_min``."""
    pairs, used_r, used_c = [], set(), set()
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < iou_min:
            break
        if r not in used_r and c not in used_c:
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    return pairs


def recover_swallowed(
    boxes: list[np.ndarray], conf: list[np.ndarray], ids: list[np.ndarray], next_id: int, iou_min: float,
    keep_confidence: float, duplicate_iou: float = 0.7,
) -> tuple[list[np.ndarray], int]:
    """Give back detections ByteTrack swallowed: the first detection of every track and tracks seen once.

    ByteTrack only outputs a track once it has been matched again, so a person who appears (or reappears after a
    pan) loses at least one frame, and a person seen in a single sample is never output. ``boxes``, ``conf`` and
    ``ids`` are per-sample arrays in time order for one motion segment (``ids`` is -1 for a swallowed detection).
    Returns new id arrays and the next free id; the inputs are not modified. Two steps:

    1. a track is extended onto the swallowed detection in the neighbouring sample (first backwards, then forwards)
       that overlaps its box by at least ``iou_min`` (one-to-one, best overlap first), repeatedly, so a track's first
       detection and one-sample gaps are filled;
    2. the remaining swallowed detections of confidence at least ``keep_confidence`` become tracks of their own
       (chained across consecutive samples by ``iou_min``), unless they sit on an already tracked box
       (``duplicate_iou``, a doubled detection).
    """
    out = [i.copy() for i in ids]
    # Backwards first (a track's first detection), then forwards (a gap inside or after a track).
    for src_order, step in ((range(len(out) - 1, 0, -1), -1), (range(len(out) - 1), 1)):
        for t in src_order:
            to = t + step
            absent = [k for k, tid in enumerate(out[t]) if tid >= 0 and int(tid) not in set(out[to].tolist())]
            free = np.flatnonzero(out[to] < 0)
            if not absent or not free.size:
                continue
            for h, c in _greedy_pairs(_iou(boxes[t][absent], boxes[to][free]), iou_min):
                out[to][free[c]] = out[t][absent[h]]

    prev_free: np.ndarray = np.empty(0, dtype=int)
    prev_new: dict[int, int] = {}
    for t in range(len(out)):
        tracked = boxes[t][out[t] >= 0]
        orphan = [
            k for k in np.flatnonzero(out[t] < 0)
            if conf[t][k] >= keep_confidence and (not len(tracked) or _iou(boxes[t][[k]], tracked).max() < duplicate_iou)
        ]
        cur_new: dict[int, int] = {}
        matched: set[int] = set()
        if orphan and len(prev_free):
            for r, c in _greedy_pairs(_iou(boxes[t][orphan], boxes[t - 1][prev_free]), iou_min):
                cur_new[orphan[r]] = prev_new[int(prev_free[c])]
                matched.add(r)
        for r, k in enumerate(orphan):
            if r not in matched:
                cur_new[k] = next_id
                next_id += 1
        for k, tid in cur_new.items():
            out[t][k] = tid
        prev_free, prev_new = np.array(sorted(cur_new), dtype=int), cur_new
    return out, next_id



BALL_DEFAULTS = {"static_radius_px": 6.0, "static_window_s": 2.0, "static_min_frames": 3, "min_confidence": 0.0}


def static_ball_mask(balls: pd.DataFrame, xy: np.ndarray, params: dict) -> np.ndarray:
    """True for ball detections that recur in place.

    A detection is static when at least ``static_min_frames`` other samples, within ``static_window_s`` and the same
    motion segment, have a ball detection within ``static_radius_px`` of it in stabilised coordinates ``xy``. These are
    spare balls, the penalty spot and line marks. A real ball waiting for a set piece is static too and is dropped as
    well: position alone cannot tell it from a spare ball.
    """
    radius, window, need = float(params["static_radius_px"]), float(params["static_window_s"]), int(params["static_min_frames"])
    t = balls["timestamp_s"].to_numpy(dtype=float)
    sample = balls["sample_id"].to_numpy()
    segment = balls["segment_id"].to_numpy()
    mask = np.zeros(len(balls), dtype=bool)
    for seg in np.unique(segment):
        idx = np.flatnonzero(segment == seg)
        idx = idx[np.argsort(t[idx], kind="stable")]
        lo = np.searchsorted(t[idx], t[idx] - window - 1e-9, side="left")
        hi = np.searchsorted(t[idx], t[idx] + window + 1e-9, side="right")
        for k, i in enumerate(idx):
            near = idx[lo[k]:hi[k]]
            close = near[(np.hypot(*(xy[near] - xy[i]).T) <= radius) & (sample[near] != sample[i])]
            mask[i] = len(np.unique(sample[close])) >= need
    return mask


def select_balls(balls: pd.DataFrame, xy: np.ndarray, params: dict) -> pd.DataFrame:
    """The highest-confidence non-static ball detection per sample, at least ``min_confidence``."""
    p = {**BALL_DEFAULTS, **params}
    keep = ~static_ball_mask(balls, xy, p) & (balls["confidence"].to_numpy(dtype=float) >= float(p["min_confidence"]))
    best = balls[keep].sort_values("confidence", ascending=False, kind="stable").groupby("sample_id").head(1)
    return best.sort_values("sample_id")


def _pick_balls(ball_dets: pd.DataFrame, motion: pd.DataFrame, params: dict) -> dict[int, pd.DataFrame]:
    """Per sample, the single ball row to keep (empty when nothing survives)."""
    mot = motion.set_index("sample_id")
    d = ball_dets[ball_dets["sample_id"].isin(mot.index)].copy()
    if d.empty:
        return {}
    d["segment_id"] = d["sample_id"].map(mot["segment_id"])
    centres = np.c_[(d["bbox_x1"] + d["bbox_x2"]) / 2, (d["bbox_y1"] + d["bbox_y2"]) / 2]
    xy = np.empty_like(centres)
    for sid, ix in d.reset_index(drop=True).groupby("sample_id").indices.items():
        xy[ix] = apply_h(G_of(mot.loc[sid]), centres[ix])
    picked = select_balls(d, xy, params)
    log.info("Ball: %d detections, %d samples kept after dropping static ones", len(d), len(picked))
    return {k: g for k, g in picked.groupby("sample_id")}


def track(detections: pd.DataFrame, motion: pd.DataFrame, params: dict, fps: float) -> pd.DataFrame:
    """Track on-pitch people per motion segment; return one row per tracked detection."""
    on = detections[detections["on_pitch"]]
    people = {k: g for k, g in on[on["object_class"] == "person"].groupby("sample_id")}
    balls = _pick_balls(on[on["object_class"] == "ball"], motion, {**BALL_DEFAULTS, **params.get("ball", {})})
    recover = {**RECOVER_DEFAULTS, **params.get("recover", {})}
    rows: list[dict] = []
    id_offset = 0
    for _, seg in motion.groupby("segment_id", sort=True):
        tracker = _new_tracker(params, fps)
        steps = []  # (motion row, G, people, track id per person; -1 when ByteTrack did not output it)
        seg_max = 0
        for m in seg.sort_values("timestamp_s").itertuples():
            G = G_of(m)
            ppl = people.get(m.sample_id)
            ids = np.full(0 if ppl is None else len(ppl), -1, dtype=int)
            if ppl is not None and len(ppl):
                result = _update(tracker, ppl, G, float(params.get("box_buffer", 0.0)))
                if len(result):
                    ids[ppl.index.get_indexer(result.data["row"])] = result.tracker_id + id_offset
                    seg_max = max(seg_max, int(result.tracker_id.max()))
            steps.append((m, G, ppl, ids))
        if recover["enabled"]:
            boxes = [stabilize_boxes(p[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=float), G)
                     if p is not None and len(p) else np.empty((0, 4)) for _, G, p, _ in steps]
            conf = [p["confidence"].to_numpy(dtype=float) if p is not None else np.empty(0) for _, _, p, _ in steps]
            new_ids, next_id = recover_swallowed(
                boxes, conf, [i for *_, i in steps], id_offset + seg_max + 1, float(recover["iou"]),
                float(recover["keep_confidence"]), float(recover["duplicate_iou"]),
            )
            steps = [(m, G, p, i) for (m, G, p, _), i in zip(steps, new_ids)]
            seg_max = next_id - id_offset - 1
        for m, G, ppl, ids in steps:
            if (ids >= 0).any():
                rows += _rows(ppl[ids >= 0], ids[ids >= 0], G, m)
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
