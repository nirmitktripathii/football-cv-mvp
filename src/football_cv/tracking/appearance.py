"""Per-track kit appearance descriptors for the identity linker.

For each person crop three horizontal bands of the bounding box are sampled (shirt, shorts, socks),
grass-coloured pixels are dropped, and the mean CIELAB colour of each band is taken. A track's
descriptor is the median over its crops and ``spread`` is the median distance of its crops from that
median, so noisy tracks carry a wide tolerance.

What this can and cannot do: players of one team wear the same kit, so these colours cannot tell
two team-mates apart. They are a *veto* (a referee, a goalkeeper, or a differently-coloured sock is
not the same person) and never positive evidence that two fragments match. Output columns are
``track_id``, ``d_shirt_*``, ``d_shorts_*``, ``d_socks_*`` (L, a, b each), ``spread`` and ``n``.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from football_cv.calibration.shots import GRASS_HIGH, GRASS_LOW

# (name, top, bottom) as fractions of the box height
BANDS = (("shirt", 0.15, 0.40), ("shorts", 0.45, 0.62), ("socks", 0.78, 0.95))
MIN_BAND_PIXELS = 8
MIN_CROPS = 3
DESCRIPTOR_COLUMNS = [f"d_{name}_{ch}" for name, _, _ in BANDS for ch in "Lab"]
APPEARANCE_COLUMNS = ["track_id", *DESCRIPTOR_COLUMNS, "spread", "n"]


def band_colours(image: np.ndarray, box) -> np.ndarray | None:
    """Mean Lab of each band as a flat (9,) vector; None if any band has too few non-grass pixels."""
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = y2 - y1, x2 - x1
    H, W = image.shape[:2]
    tx1, tx2 = max(0, int(x1 + 0.25 * w)), min(W, int(x2 - 0.25 * w))
    out = []
    for _, top, bottom in BANDS:
        ty1, ty2 = max(0, int(y1 + top * h)), min(H, int(y1 + bottom * h))
        if ty2 - ty1 < 2 or tx2 - tx1 < 2:
            return None
        crop = image[ty1:ty2, tx1:tx2]
        not_grass = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), GRASS_LOW, GRASS_HIGH) == 0
        if not_grass.sum() < MIN_BAND_PIXELS:
            return None
        out.append(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[not_grass].mean(axis=0))
    return np.concatenate(out)


def sample_bands(tracks: pd.DataFrame, frames: pd.DataFrame, frames_root, min_box_h: float) -> pd.DataFrame:
    """One row per usable person crop: ``track_id`` plus the nine band colours."""
    ppl = tracks[(tracks["object_class"] == "person") & ((tracks["bbox_y2"] - tracks["bbox_y1"]) >= min_box_h)]
    paths = frames.set_index("sample_id")["path"]
    rows = []
    for sid, grp in ppl.groupby("sample_id"):
        img = cv2.imread(str(frames_root / paths[sid]))
        if img is None:
            continue
        for r in grp.itertuples():
            v = band_colours(img, (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2))
            if v is not None:
                rows.append([r.track_id, *v])
    return pd.DataFrame(rows, columns=["track_id", *DESCRIPTOR_COLUMNS])


def describe_tracks(samples: pd.DataFrame, min_crops: int = MIN_CROPS) -> pd.DataFrame:
    """Median descriptor and spread per track; tracks with fewer than ``min_crops`` crops get none."""
    rows = []
    for tid, g in samples.groupby("track_id"):
        if len(g) < min_crops:
            continue
        X = g[DESCRIPTOR_COLUMNS].to_numpy(float)
        med = np.median(X, axis=0)
        spread = float(np.median(np.linalg.norm(X - med, axis=1)))
        rows.append({"track_id": tid, **dict(zip(DESCRIPTOR_COLUMNS, med)), "spread": spread, "n": len(g)})
    return pd.DataFrame(rows, columns=APPEARANCE_COLUMNS)
