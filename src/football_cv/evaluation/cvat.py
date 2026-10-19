"""Exchange tracks with CVAT for hand-corrected ground truth (phase 0, docs/annotation_guide.md).

``to_cvat_video_xml`` turns tracker output into a "CVAT for video 1.1" file. It is imported into a task
created from the window's sampled frames, so CVAT frame ``i`` is ``sample_id`` ``i``. Every box is a
keyframe, so CVAT never interpolates between them. A track that disappears gets an ``outside`` box on the
first frame it is missing, so it is not stretched over the gap.

``parse_cvat_xml`` reads the corrected export back as one row per visible box.

Labels:

- ``person``, with immutable ``role`` and ``team`` attributes;
- ``ball``;
- ``ignore``, for people outside the evaluation scope (staff, substitutes warming up). Predictions on
  ``ignore`` boxes are dropped before scoring rather than counted as false positives.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PERSON, BALL, IGNORE = "person", "ball", "ignore"
ROLES = ("player", "goalkeeper", "referee", "unknown")
TEAMS = ("home", "away", "none", "unknown")
# Pre-fill guesses from the pipeline's team_id. Kit colour cannot tell a goalkeeper from a referee, so
# anything not clearly home/away starts as role "unknown" for the annotator to set.
_PREFILL_ROLE = {"home": "player", "away": "player"}
_PREFILL_TEAM = {"home": "home", "away": "away"}
GT_COLUMNS = ["frame", "gt_id", "label", "x1", "y1", "x2", "y2", "occluded", "role", "team"]


def label_spec() -> list[dict]:
    """Labels for CVAT's "Raw" label editor when creating the task (paste the JSON)."""
    return [
        {
            "name": PERSON, "color": "#33ddff", "type": "rectangle",
            "attributes": [
                {"name": "role", "mutable": False, "input_type": "select", "default_value": "unknown", "values": list(ROLES)},
                {"name": "team", "mutable": False, "input_type": "select", "default_value": "unknown", "values": list(TEAMS)},
            ],
        },
        {"name": BALL, "color": "#ffcc00", "type": "rectangle", "attributes": []},
        {"name": IGNORE, "color": "#888888", "type": "rectangle", "attributes": []},
    ]


def _runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted frame numbers into contiguous [start, end] runs."""
    breaks = np.flatnonzero(np.diff(frames) > 1)
    starts = np.r_[frames[0], frames[breaks + 1]]
    ends = np.r_[frames[breaks], frames[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _box(parent: ET.Element, frame: int, xyxy, outside: bool, attrs: dict[str, str]) -> None:
    x1, y1, x2, y2 = (f"{v:.2f}" for v in xyxy)
    el = ET.SubElement(parent, "box", frame=str(frame), keyframe="1", outside=str(int(outside)), occluded="0",
                       xtl=x1, ytl=y1, xbr=x2, ybr=y2, z_order="0")
    for name, value in attrs.items():
        ET.SubElement(el, "attribute", name=name).text = value


def _add_track(root: ET.Element, xml_id: int, label: str, rows: pd.DataFrame, num_frames: int, attrs: dict[str, str]) -> None:
    rows = rows.sort_values("sample_id").drop_duplicates("sample_id")
    track = ET.SubElement(root, "track", id=str(xml_id), label=label, source="auto")
    boxes = rows[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    frames = rows["sample_id"].to_numpy(int)
    for start, end in _runs(frames):
        for i in np.flatnonzero((frames >= start) & (frames <= end)):
            _box(track, int(frames[i]), boxes[i], False, attrs)
        if end + 1 < num_frames:
            _box(track, end + 1, boxes[np.flatnonzero(frames == end)[0]], True, attrs)


def to_cvat_video_xml(tracks: pd.DataFrame, num_frames: int, width: int, height: int, name: str) -> str:
    """CVAT for video 1.1 annotations pre-filled from ``tracks.parquet`` rows.

    Person tracks keep their tracker id order. The ball (``track_id == -1``, one detection per frame) becomes
    a single ``ball`` track.
    """
    t = tracks[(tracks["sample_id"] >= 0) & (tracks["sample_id"] < num_frames)].copy()
    t[["bbox_x1", "bbox_x2"]] = t[["bbox_x1", "bbox_x2"]].clip(0, width)
    t[["bbox_y1", "bbox_y2"]] = t[["bbox_y1", "bbox_y2"]].clip(0, height)
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "name").text = name
    ET.SubElement(task, "size").text = str(num_frames)
    ET.SubElement(task, "mode").text = "annotation"
    size = ET.SubElement(task, "original_size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)

    xml_id = 0
    people = t[(t["object_class"] == PERSON) & (t["track_id"] >= 0)]
    for _, rows in people.groupby("track_id", sort=True):
        team_id = str(rows["team_id"].mode().iloc[0]) if "team_id" in rows else "unknown"
        attrs = {"role": _PREFILL_ROLE.get(team_id, "unknown"), "team": _PREFILL_TEAM.get(team_id, "unknown")}
        _add_track(root, xml_id, PERSON, rows, num_frames, attrs)
        xml_id += 1
    ball = t[t["object_class"] == BALL]
    if len(ball):
        _add_track(root, xml_id, BALL, ball, num_frames, {})
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def parse_cvat_xml(text: str) -> pd.DataFrame:
    """Visible boxes from a CVAT for video 1.1 export, one row per (frame, track)."""
    root = ET.fromstring(text)
    if root.find("image") is not None and root.find("track") is None:
        raise ValueError("this is a 'CVAT for images' export; export the task as 'CVAT for video 1.1' to keep track ids")
    rows = []
    for track in root.iter("track"):
        gt_id, label = int(track.get("id")), track.get("label")
        for box in track.iter("box"):
            if box.get("outside") == "1":
                continue
            attrs = {a.get("name"): (a.text or "").strip() for a in box.iter("attribute")}
            rows.append({
                "frame": int(box.get("frame")), "gt_id": gt_id, "label": label,
                "x1": float(box.get("xtl")), "y1": float(box.get("ytl")),
                "x2": float(box.get("xbr")), "y2": float(box.get("ybr")),
                "occluded": box.get("occluded") == "1",
                "role": attrs.get("role", ""), "team": attrs.get("team", ""),
            })
    gt = pd.DataFrame(rows, columns=GT_COLUMNS)
    dup = gt.duplicated(["frame", "gt_id"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (frame, track) boxes in the export")
    return gt.sort_values(["frame", "gt_id"], ignore_index=True)
