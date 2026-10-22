"""Shot segmentation and wide-view classification for broadcast footage.

Broadcasts cut between the elevated tactical camera, close-ups, replays and
crowd shots. Pitch calibration and trajectories are only meaningful on the
wide tactical camera, so every sampled frame gets:

* ``shot_id``   - increments at a hard cut (colour-histogram correlation drop),
* ``wide_view`` - the frame looks like the tactical camera: mostly grass, several
  small people standing on it.

The grass mask is also used to drop detections in the stands (spectators are
"person" detections too).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

# HSV range for broadcast pitch grass (OpenCV hue is 0-179).
GRASS_LOW = np.array([30, 40, 40], dtype=np.uint8)
GRASS_HIGH = np.array([90, 255, 255], dtype=np.uint8)
# Inside a person crop, grass is green AND dim. Referee shirts are lime (hue 32-40, inside the grass range) but bright
# (value ~235 against ~125 for turf, 99th percentile 167), so a brightness cap keeps them. Used for kit colours only.
PERSON_GRASS_HIGH = np.array([90, 255, 190], dtype=np.uint8)
MIN_COMPONENT_SHARE = 0.25   # a green component this large relative to the biggest is still pitch


def grass_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of the playing surface, holes from players and lines filled.

    Takes the green components that are at least ``MIN_COMPONENT_SHARE`` of the largest one and fills their joint
    convex hull, so white lines and players standing on the grass count as pitch, while stands, advertising boards and
    the athletics track do not. Several components are kept because a wide halfway line splits the grass in two, and
    keeping only the largest left one half of the pitch (and every player on it) outside the mask.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    mask = np.zeros_like(green)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= MIN_COMPONENT_SHARE * areas.max())
    # The hull of a region equals the hull of its outer contour; hulling every pixel took ~120 ms per frame.
    contours, _ = cv2.findContours(np.isin(labels, keep).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.concatenate(contours))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def kit_pixel_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of the crop's pixels that belong to a person's kit rather than the pitch behind them."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, GRASS_LOW, PERSON_GRASS_HIGH) == 0


def green_ratio(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.count_nonzero(cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH))) / (image.shape[0] * image.shape[1])


def colour_hist(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def feet_on_pitch(dets: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Boolean per detection: is the bottom-centre of the box inside the grass mask?"""
    if dets.empty:
        return np.zeros(0, dtype=bool)
    h, w = mask.shape
    fx = ((dets["bbox_x1"] + dets["bbox_x2"]) / 2).clip(0, w - 1).astype(int).to_numpy()
    fy = dets["bbox_y2"].clip(0, h - 1).astype(int).to_numpy()
    return mask[fy, fx] > 0


def classify_frames(
    frames: pd.DataFrame,
    frames_root,
    detections: pd.DataFrame,
    min_green_ratio: float,
    max_median_person_height_px: float,
    min_persons: int,
    cut_hist_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-frame shot table, all detections annotated with ``on_pitch``)."""
    dets = detections.copy()
    dets["on_pitch"] = False
    by_sample = {k: v.index for k, v in dets.groupby("sample_id")}
    rows = []
    prev_hist = None
    shot_id = 0
    for rec in frames.itertuples():
        img = cv2.imread(str(frames_root / rec.path))
        if img is None:
            raise FileNotFoundError(frames_root / rec.path)
        hist = colour_hist(img)
        corr = 1.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
        is_cut = prev_hist is not None and corr < cut_hist_threshold
        if is_cut:
            shot_id += 1
        prev_hist = hist
        mask = grass_mask(img)
        idx = by_sample.get(rec.sample_id)
        heights = np.array([])
        if idx is not None:
            on = feet_on_pitch(dets.loc[idx], mask)
            dets.loc[idx, "on_pitch"] = on
            sub = dets.loc[idx][on & (dets.loc[idx, "object_class"] == "person").to_numpy()]
            heights = (sub["bbox_y2"] - sub["bbox_y1"]).to_numpy()
        g = green_ratio(img)
        med_h = float(np.median(heights)) if heights.size else float("nan")
        wide = bool(
            g >= min_green_ratio
            and heights.size >= min_persons
            and med_h <= max_median_person_height_px
        )
        rows.append(
            {
                "sample_id": rec.sample_id, "timestamp_s": rec.timestamp_s,
                "green_ratio": round(g, 4), "hist_corr": round(corr, 4), "is_cut": is_cut,
                "shot_id": shot_id, "n_persons_on_pitch": int(heights.size),
                "median_person_h": round(med_h, 1), "wide_view_raw": wide,
            }
        )
    shots = pd.DataFrame(rows)
    shots["wide_view"] = smooth_wide_flags(shots)
    return shots, dets


def smooth_wide_flags(shots: pd.DataFrame, min_shot_wide_share: float = 0.6) -> pd.Series:
    """A frame is wide only if its whole shot is mostly wide; removes one-frame flickers."""
    share = shots.groupby("shot_id")["wide_view_raw"].transform("mean")
    return (share >= min_shot_wide_share) & shots["wide_view_raw"]


def shot_table(shots: pd.DataFrame) -> pd.DataFrame:
    """One row per shot with its time span and whether it is usable for calibration."""
    g = shots.groupby("shot_id")
    return pd.DataFrame(
        {
            "start_s": g["timestamp_s"].min(),
            "end_s": g["timestamp_s"].max(),
            "n_frames": g.size(),
            "wide_share": g["wide_view"].mean().round(3),
            "median_green": g["green_ratio"].median().round(3),
        }
    ).reset_index()
