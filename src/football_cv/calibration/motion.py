"""Broadcast camera-motion estimation by chained inter-frame homographies.

The main broadcast camera pans, tilts and zooms about a fixed position. For a
camera rotating about its optical centre every static scene point (pitch,
stands, boards) moves by the same 3x3 homography between two frames, so ORB
features matched with RANSAC give that homography directly.

For every sampled frame ``t`` in a shot we keep ``G_t``: the homography that
maps frame-t pixels into the pixels of the segment's *reference frame* (its
first frame). Players are masked out because they move independently, and the
scoreboard overlay is masked because it does not move with the scene.

A chain breaks (new ``segment_id``) at a shot cut, when too few inliers
survive, or when the estimated motion is implausible for 0.2 s of broadcast
camera movement. Downstream code never crosses a segment boundary with ``G``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WORK_SCALE = 0.5  # features are computed on a half-resolution image
SCOREBOARD_BOX = (0, 0, 560, 100)  # x1, y1, x2, y2 in full-res pixels (top-left overlay)
MIN_SCALE_STEP, MAX_SCALE_STEP = 0.8, 1.25  # plausible zoom change between samples
MAX_PERSPECTIVE = 2e-3  # |h31|, |h32| at full resolution


@dataclass(frozen=True)
class MotionStep:
    ok: bool
    H: np.ndarray | None  # maps current frame -> previous frame (full-res pixels)
    inliers: int
    reason: str


def static_mask(shape: tuple[int, int], boxes: np.ndarray, pad: int = 6) -> np.ndarray:
    """uint8 mask (255 = usable) at full resolution, excluding people and the scoreboard."""
    h, w = shape
    mask = np.full((h, w), 255, np.uint8)
    x1, y1, x2, y2 = SCOREBOARD_BOX
    mask[y1:y2, x1:x2] = 0
    for bx1, by1, bx2, by2 in boxes:
        mask[max(0, int(by1) - pad) : min(h, int(by2) + pad), max(0, int(bx1) - pad) : min(w, int(bx2) + pad)] = 0
    return mask


def _scaled(H_small: np.ndarray) -> np.ndarray:
    S = np.diag([WORK_SCALE, WORK_SCALE, 1.0])
    H = np.linalg.inv(S) @ H_small @ S
    return H / H[2, 2]


def plausible(H: np.ndarray) -> tuple[bool, str]:
    """Reject homographies that cannot come from 0.2 s of broadcast camera motion."""
    det = float(np.linalg.det(H[:2, :2]))
    if det <= 0:
        return False, "orientation flip"
    scale = np.sqrt(det)
    if not MIN_SCALE_STEP <= scale <= MAX_SCALE_STEP:
        return False, f"zoom step {scale:.2f}"
    if np.abs(H[2, :2]).max() > MAX_PERSPECTIVE:
        return False, "perspective"
    return True, ""


class MotionEstimator:
    def __init__(self, n_features: int = 3000, min_inliers: int = 60, ransac_px: float = 2.0):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_inliers = min_inliers
        self.ransac_px = ransac_px

    def features(self, image: np.ndarray, mask: np.ndarray):
        small = cv2.resize(image, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        return self.orb.detectAndCompute(gray, m)

    def step(self, prev_feats, cur_feats) -> MotionStep:
        (kp0, d0), (kp1, d1) = prev_feats, cur_feats
        if d0 is None or d1 is None or len(kp0) < self.min_inliers or len(kp1) < self.min_inliers:
            return MotionStep(False, None, 0, "too few features")
        pairs = self.matcher.knnMatch(d1, d0, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < self.min_inliers:
            return MotionStep(False, None, len(good), "too few matches")
        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp0[m.trainIdx].pt for m in good])
        H_small, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        n_in = int(inl.sum()) if inl is not None else 0
        if H_small is None or n_in < self.min_inliers:
            return MotionStep(False, None, n_in, "too few inliers")
        H = _scaled(H_small)
        ok, why = plausible(H)
        return MotionStep(ok, H if ok else None, n_in, why)


def estimate_motion(
    frames: pd.DataFrame,
    frames_root,
    shots: pd.DataFrame,
    persons: pd.DataFrame,
    n_features: int,
    min_inliers: int,
    max_chain_s: float,
) -> pd.DataFrame:
    """One row per wide-view frame: segment id, inliers and ``G`` (frame -> segment reference).

    Non-wide frames are skipped entirely; they never receive pitch coordinates.
    ``max_chain_s`` starts a new segment after that long so accumulated drift
    stays bounded and each segment can be anchored by its own keyframe.
    """
    est = MotionEstimator(n_features, min_inliers)
    wide = shots.set_index("sample_id")[["shot_id", "wide_view"]]
    boxes_by_sample = {
        k: g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy() for k, g in persons.groupby("sample_id")
    }
    rows = []
    segment_id = -1
    prev = None  # (shot_id, features, G, segment start time)
    for rec in frames.itertuples():
        if not bool(wide.at[rec.sample_id, "wide_view"]):
            prev = None
            continue
        shot_id = int(wide.at[rec.sample_id, "shot_id"])
        img = cv2.imread(str(frames_root / rec.path))
        boxes = boxes_by_sample.get(rec.sample_id, np.empty((0, 4)))
        feats = est.features(img, static_mask(img.shape[:2], boxes))
        G, inliers, reason = None, 0, "no previous wide frame"
        if prev is not None and prev[0] == shot_id and rec.timestamp_s - prev[3] <= max_chain_s:
            st = est.step(prev[1], feats)
            inliers, reason = st.inliers, st.reason
            if st.ok:
                G = prev[2] @ st.H
                G = G / G[2, 2]
        elif prev is not None:
            reason = "shot change" if prev[0] != shot_id else "max chain length"
        if G is None:
            segment_id += 1
            G = np.eye(3)
            seg_start = rec.timestamp_s
            if prev is not None:
                log.info("Motion chain break at t=%.1fs (%s, %d inliers)", rec.timestamp_s, reason, inliers)
        else:
            seg_start = prev[3]
        rows.append(
            {
                "sample_id": rec.sample_id, "timestamp_s": rec.timestamp_s, "shot_id": shot_id,
                "segment_id": segment_id, "inliers": inliers,
                **{f"g{i}{j}": float(G[i, j]) for i in range(3) for j in range(3)},
            }
        )
        prev = (shot_id, feats, G, seg_start)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["is_segment_start"] = ~out["segment_id"].duplicated()
    return out


def G_of(row) -> np.ndarray:
    """3x3 frame -> segment-reference homography from a motion row (Series or namedtuple)."""
    get = row.__getitem__ if isinstance(row, pd.Series) else (lambda k: getattr(row, k))
    return np.array([[get(f"g{i}{j}") for j in range(3)] for i in range(3)])


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
