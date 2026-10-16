"""Aid for manual calibration: render a keyframe with a pixel grid and candidate line intersections.

    python scripts/annotate_keyframe.py 120 455 900

For every sample id it writes ``artifacts/<clip>/calibration/annotate_<id>.jpg``
(grid every 50 px, detected pitch-line pixels in magenta, numbered candidate
intersections of straight line segments) and prints the candidates' pixel
coordinates. A human then decides which candidates are which pitch landmarks
and records them in ``configs/calibration_keyframes.json``; nothing is
assigned automatically.
"""
from __future__ import annotations

import argparse
import itertools

import cv2
import numpy as np
import pandas as pd

from football_cv import config as C
from football_cv.calibration.homography import line_mask
from football_cv.calibration.shots import grass_mask


def segments(mask: np.ndarray) -> np.ndarray:
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60, minLineLength=80, maxLineGap=12)
    return np.empty((0, 4)) if lines is None else lines[:, 0, :].astype(float)


def _near(seg: np.ndarray, q: np.ndarray) -> bool:
    """Is q on the segment, allowing a short extension past either end?"""
    v = seg[2:] - seg[:2]
    u = np.clip(np.dot(q - seg[:2], v) / np.dot(v, v), -0.15, 1.15)
    return bool(np.linalg.norm(seg[:2] + u * v - q) < 6)


def intersections(segs: np.ndarray, shape, min_angle_deg: float = 12.0) -> np.ndarray:
    h, w = shape
    pts = []
    for a, b in itertools.combinations(segs, 2):
        d1, d2 = a[2:] - a[:2], b[2:] - b[:2]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9) < np.sin(np.radians(min_angle_deg)):
            continue
        t = ((b[0] - a[0]) * d2[1] - (b[1] - a[1]) * d2[0]) / cross
        p = a[:2] + t * d1
        if 0 <= p[0] < w and 0 <= p[1] < h and _near(a, p) and _near(b, p):
            pts.append(p)
    merged: list[list] = []
    for p in pts:  # greedy merge of near-duplicates
        for m in merged:
            if np.linalg.norm(m[0] / m[1] - p) < 10:
                m[0] = m[0] + p
                m[1] += 1
                break
        else:
            merged.append([p.copy(), 1])
    return np.array([m[0] / m[1] for m in merged]).reshape(-1, 2)


def render(image: np.ndarray, mask: np.ndarray, cands: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[mask > 0] = (255, 0, 255)
    h, w = image.shape[:2]
    for x in range(0, w, 50):
        cv2.line(out, (x, 0), (x, h), (255, 255, 255) if x % 100 == 0 else (150, 150, 150), 1)
        if x % 100 == 0:
            cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for y in range(0, h, 50):
        cv2.line(out, (0, y), (w, y), (255, 255, 255) if y % 100 == 0 else (150, 150, 150), 1)
        if y % 100 == 0:
            cv2.putText(out, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for i, (x, y) in enumerate(cands):
        cv2.circle(out, (int(x), int(y)), 5, (0, 255, 255), 2)
        cv2.putText(out, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample_ids", nargs="+", type=int)
    ap.add_argument("--config", default=str(C.DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg = C.load_config(args.config, [])
    run = C.run_dir(cfg)
    frames = pd.read_parquet(run / "frames.parquet").set_index("sample_id")
    dest = C.artifacts_dir(cfg) / "calibration"
    dest.mkdir(parents=True, exist_ok=True)
    for sid in args.sample_ids:
        img = cv2.imread(str(run / frames.at[sid, "path"]))
        mask = line_mask(img, grass_mask(img))
        cands = intersections(segments(mask), img.shape[:2])
        path = dest / f"annotate_{sid:06d}.jpg"
        cv2.imwrite(str(path), render(img, mask, cands))
        print(f"sample {sid} t={frames.at[sid, 'timestamp_s']:.1f}s -> {path}")
        for i, (x, y) in enumerate(cands):
            print(f"  [{i}] ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
