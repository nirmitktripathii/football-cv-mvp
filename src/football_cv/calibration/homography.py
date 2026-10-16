"""Image -> pitch homographies from manual keyframes, line-alignment refinement and propagation.

Pitch coordinates are metres on a 105 x 68 reference pitch, origin at the
bottom-left corner as seen in the broadcast: ``x`` runs along the touchline
(0 = left goal line, 105 = right goal line), ``y`` across it (0 = near
touchline at the bottom of the image, 68 = far touchline).

Workflow
1. A human picks a wide-view keyframe and lists image points whose pitch
   position is known from the markings (corners, box corners, halfway line,
   centre circle) in ``configs/calibration_keyframes.json``. A point may also
   be tied to a straight marking only (``"on_line": "touchline_near"``): it
   fixes one pitch coordinate, which anchors views where a touchline is the
   only marking in part of the image.
2. ``fit_from_correspondences`` solves the homography (normalised DLT over all
   point and line constraints) and reports the reprojection error in metres.
3. ``refine_with_lines`` nudges it so the projected pitch markings sit on the
   white line pixels (truncated chamfer distance, 4-point parameterisation).
   The refinement is kept only if it lowers the line cost *and* stays within
   ``max_shift_m`` of every manual correspondence.
4. ``frame_homography`` propagates a keyframe to any frame of the same motion
   segment: ``H_t = H_k @ inv(G_k) @ G_t``.
5. ``validate`` rejects transforms that are degenerate, mirror the pitch,
   put the pitch horizon inside the lower image, or project the visible grass
   far outside the pitch.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from football_cv.calibration.motion import apply_h

log = logging.getLogger(__name__)

L, W = 105.0, 68.0
PEN_D, PEN_HALF_W = 16.5, 20.16
GOAL_AREA_D, GOAL_AREA_HALF_W = 5.5, 9.16
CIRCLE_R = 9.15
PEN_SPOT = 11.0


def landmarks(length: float = L, width: float = W) -> dict[str, tuple[float, float]]:
    """Named pitch points usable as manual correspondences (metres)."""
    cy = width / 2
    arc_dy = float(np.sqrt(CIRCLE_R**2 - (PEN_D - PEN_SPOT) ** 2))
    pts = {
        "corner_bl": (0, 0), "corner_tl": (0, width), "corner_br": (length, 0), "corner_tr": (length, width),
        "halfway_bottom": (length / 2, 0), "halfway_top": (length / 2, width), "centre_spot": (length / 2, cy),
        "circle_bottom": (length / 2, cy - CIRCLE_R), "circle_top": (length / 2, cy + CIRCLE_R),
        "circle_left": (length / 2 - CIRCLE_R, cy), "circle_right": (length / 2 + CIRCLE_R, cy),
    }
    for side, x0, sgn in (("l", 0.0, 1), ("r", length, -1)):
        pts |= {
            f"pen_{side}_goalline_bottom": (x0, cy - PEN_HALF_W),
            f"pen_{side}_goalline_top": (x0, cy + PEN_HALF_W),
            f"pen_{side}_corner_bottom": (x0 + sgn * PEN_D, cy - PEN_HALF_W),
            f"pen_{side}_corner_top": (x0 + sgn * PEN_D, cy + PEN_HALF_W),
            f"pen_{side}_arc_bottom": (x0 + sgn * PEN_D, cy - arc_dy),
            f"pen_{side}_arc_top": (x0 + sgn * PEN_D, cy + arc_dy),
            f"pen_{side}_spot": (x0 + sgn * PEN_SPOT, cy),
            f"ga_{side}_goalline_bottom": (x0, cy - GOAL_AREA_HALF_W),
            f"ga_{side}_goalline_top": (x0, cy + GOAL_AREA_HALF_W),
            f"ga_{side}_corner_bottom": (x0 + sgn * GOAL_AREA_D, cy - GOAL_AREA_HALF_W),
            f"ga_{side}_corner_top": (x0 + sgn * GOAL_AREA_D, cy + GOAL_AREA_HALF_W),
        }
    return {k: (float(x), float(y)) for k, (x, y) in pts.items()}


def pitch_lines(length: float = L, width: float = W) -> dict[str, tuple[int, float]]:
    """Named straight markings as (axis, value): a point on one fixes x (axis 0) or y (axis 1)."""
    return {
        "touchline_near": (1, 0.0), "touchline_far": (1, width),
        "goalline_left": (0, 0.0), "goalline_right": (0, length), "halfway_line": (0, length / 2),
        "pen_l_front": (0, PEN_D), "pen_r_front": (0, length - PEN_D),
    }


def pitch_line_samples(length: float = L, width: float = W, step: float = 0.5) -> list[np.ndarray]:
    """Pitch markings as polylines of points every ``step`` metres."""
    def seg(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
        return np.linspace(a, b, n)

    def arc(cx, cy, r, t0, t1):
        n = max(8, int(abs(t1 - t0) * r / step))
        t = np.linspace(t0, t1, n)
        return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]

    cy = width / 2
    lines = [
        seg((0, 0), (length, 0)), seg((0, width), (length, width)),
        seg((0, 0), (0, width)), seg((length, 0), (length, width)),
        seg((length / 2, 0), (length / 2, width)),
        arc(length / 2, cy, CIRCLE_R, 0, 2 * np.pi),
    ]
    half_angle = np.arccos((PEN_D - PEN_SPOT) / CIRCLE_R)
    for x0, sgn in ((0.0, 1), (length, -1)):
        px = x0 + sgn * PEN_D
        gx = x0 + sgn * GOAL_AREA_D
        lines += [
            seg((x0, cy - PEN_HALF_W), (px, cy - PEN_HALF_W)), seg((px, cy - PEN_HALF_W), (px, cy + PEN_HALF_W)),
            seg((px, cy + PEN_HALF_W), (x0, cy + PEN_HALF_W)),
            seg((x0, cy - GOAL_AREA_HALF_W), (gx, cy - GOAL_AREA_HALF_W)),
            seg((gx, cy - GOAL_AREA_HALF_W), (gx, cy + GOAL_AREA_HALF_W)),
            seg((gx, cy + GOAL_AREA_HALF_W), (x0, cy + GOAL_AREA_HALF_W)),
        ]
        centre_angle = 0.0 if sgn == 1 else np.pi
        lines.append(arc(x0 + sgn * PEN_SPOT, cy, CIRCLE_R, centre_angle - half_angle, centre_angle + half_angle))
    return lines


@dataclass
class Keyframe:
    sample_id: int
    names: list[str]
    image_pts: np.ndarray  # (N, 2) pixels
    pitch_pts: np.ndarray  # (N, 2) metres
    note: str = ""
    line_names: list[str] = field(default_factory=list)
    line_img: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))  # (M, 2) pixels on a marking
    line_axis: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))  # 0: fixes x, 1: fixes y
    line_val: np.ndarray = field(default_factory=lambda: np.empty(0))  # the fixed coordinate, metres
    H: np.ndarray | None = None  # image -> pitch
    diagnostics: dict = field(default_factory=dict)


def load_keyframes(path: Path) -> list[Keyframe]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing; annotate calibration keyframes first (see docs/architecture.md)")
    marks, lines = landmarks(), pitch_lines()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for kf in data["keyframes"]:
        pts = [c for c in kf["points"] if "landmark" in c]
        on_line = [c for c in kf["points"] if "on_line" in c]
        names, line_names = [c["landmark"] for c in pts], [c["on_line"] for c in on_line]
        unknown = [n for n in names if n not in marks] + [n for n in line_names if n not in lines]
        if unknown or len(pts) + len(on_line) != len(kf["points"]):
            raise ValueError(f"keyframe {kf['sample_id']}: unknown or untyped points {unknown}")
        out.append(
            Keyframe(
                sample_id=int(kf["sample_id"]), names=names,
                image_pts=np.array([c["image"] for c in pts], dtype=float).reshape(-1, 2),
                pitch_pts=np.array([marks[n] for n in names], dtype=float).reshape(-1, 2),
                note=kf.get("note", ""), line_names=line_names,
                line_img=np.array([c["image"] for c in on_line], dtype=float).reshape(-1, 2),
                line_axis=np.array([lines[n][0] for n in line_names], dtype=int),
                line_val=np.array([lines[n][1] for n in line_names], dtype=float),
            )
        )
    return out


def _similarity(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centroid to the origin, mean distance sqrt(2)."""
    c = pts.mean(axis=0)
    s = np.sqrt(2) / max(np.linalg.norm(pts - c, axis=1).mean(), 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def _dlt_with_lines(img, pitch, line_img, line_axis, line_val) -> np.ndarray:
    """Normalised DLT: each point gives two equations, each point-on-line one."""
    Ti = _similarity(np.vstack([img, line_img]))
    Tp = _similarity(pitch) if len(pitch) >= 2 else np.eye(3)
    u = apply_h(Ti, np.vstack([img, line_img]))
    p = apply_h(Tp, pitch) if len(pitch) else pitch
    rows = []
    for (a, b), (X, Y) in zip(u[: len(img)], p):
        rows.append([a, b, 1, 0, 0, 0, -X * a, -X * b, -X])
        rows.append([0, 0, 0, a, b, 1, -Y * a, -Y * b, -Y])
    for (a, b), axis, val in zip(u[len(img):], line_axis, line_val):
        c = Tp[axis, axis] * val + Tp[axis, 2]  # the line's constant in normalised pitch units
        rows.append([a, b, 1, 0, 0, 0, -c * a, -c * b, -c] if axis == 0 else [0, 0, 0, a, b, 1, -c * a, -c * b, -c])
    _, sv, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    if sv[-2] < 1e-9 * sv[0]:
        raise ValueError("degenerate correspondences (collinear points?)")
    H = np.linalg.inv(Tp) @ vt[-1].reshape(3, 3) @ Ti
    return H / H[2, 2]


def line_errors(H: np.ndarray, line_img: np.ndarray, line_axis: np.ndarray, line_val: np.ndarray) -> np.ndarray:
    """Distance (m) of point-on-line clicks from their marking after projection."""
    if not len(line_img):
        return np.empty(0)
    proj = apply_h(H, line_img)
    return np.abs(proj[np.arange(len(proj)), line_axis] - line_val)


def fit_from_correspondences(
    image_pts: np.ndarray, pitch_pts: np.ndarray, line_img: np.ndarray | None = None,
    line_axis: np.ndarray | None = None, line_val: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """DLT homography image -> pitch from all constraints, and the RMS point error in metres."""
    n_lines = 0 if line_img is None else len(line_img)
    if 2 * len(image_pts) + n_lines < 8 or len(image_pts) + n_lines < 4:
        raise ValueError("need at least 4 correspondences (8 equations)")
    if n_lines == 0:
        H, _ = cv2.findHomography(image_pts.astype(np.float64), pitch_pts.astype(np.float64), 0)
        if H is None:
            raise ValueError("degenerate correspondences (collinear points?)")
        H = H / H[2, 2]
    else:
        H = _dlt_with_lines(image_pts, pitch_pts, line_img, line_axis, line_val)
    err = np.linalg.norm(apply_h(H, image_pts) - pitch_pts, axis=1) if len(image_pts) else np.zeros(1)
    return H, float(np.sqrt(np.mean(err**2)))


def line_mask(image: np.ndarray, grass: np.ndarray | None = None) -> np.ndarray:
    """uint8 mask of bright, thin, low-saturation structures (painted lines) on the grass."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    mask = ((tophat > 25) & (sat < 90)).astype(np.uint8) * 255
    if grass is not None:
        mask &= grass
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


class LineCost:
    """Truncated mean distance (px) from projected pitch-line samples to the nearest line pixel."""

    def __init__(self, image: np.ndarray, grass: np.ndarray, truncate_px: float = 12.0):
        self.mask = line_mask(image, grass)
        self.dist = cv2.distanceTransform(255 - self.mask, cv2.DIST_L2, 5)
        self.grass = grass
        self.truncate = truncate_px
        self.samples = np.vstack(pitch_line_samples())
        self.h, self.w = image.shape[:2]

    def visible(self, H_img2pitch: np.ndarray) -> np.ndarray:
        pts = apply_h(np.linalg.inv(H_img2pitch), self.samples)
        ok = np.isfinite(pts).all(1) & (pts[:, 0] >= 0) & (pts[:, 0] < self.w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < self.h - 1)
        pts = pts[ok]
        inside = self.grass[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0
        return pts[inside]

    def __call__(self, H_img2pitch: np.ndarray) -> tuple[float, int]:
        pts = self.visible(H_img2pitch)
        if len(pts) < 50:
            return float(self.truncate), len(pts)
        d = self.dist[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        return float(np.minimum(d, self.truncate).mean()), len(pts)


CONTROL_PX = np.array([[240.0, 300.0], [1040.0, 300.0], [1040.0, 660.0], [240.0, 660.0]])


def refine_with_lines(
    H: np.ndarray, image: np.ndarray, grass: np.ndarray, anchors_img: np.ndarray | None = None,
    anchors_pitch: np.ndarray | None = None, max_shift_m: float = 2.5, anchor_lines: tuple | None = None,
) -> tuple[np.ndarray, dict]:
    """Snap projected markings onto image lines; returns (H, diagnostics). Falls back to ``H``."""
    cost = LineCost(image, grass)
    c0, n0 = cost(H)
    ctrl_pitch = apply_h(H, CONTROL_PX)

    def build(delta: np.ndarray) -> np.ndarray | None:
        Hn, _ = cv2.findHomography(CONTROL_PX, ctrl_pitch + delta.reshape(4, 2), 0)
        return None if Hn is None else Hn / Hn[2, 2]

    def objective(delta: np.ndarray) -> float:
        Hn = build(delta)
        if Hn is None:
            return 1e3
        c, n = cost(Hn)
        return c if n >= 0.5 * n0 else cost.truncate  # do not "win" by projecting lines off-screen

    res = minimize(objective, np.zeros(8), method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 4000})
    Hr = build(res.x)
    diag = {"line_cost_before_px": round(c0, 3), "line_samples": int(n0)}
    if Hr is None:
        return H, diag | {"refined": False, "reason": "degenerate"}
    c1, n1 = cost(Hr)
    diag |= {"line_cost_after_px": round(c1, 3), "line_samples_after": int(n1)}
    if anchors_img is not None and len(anchors_img):
        shift = np.linalg.norm(apply_h(Hr, anchors_img) - anchors_pitch, axis=1)
        if anchor_lines is not None:
            shift = np.r_[shift, line_errors(Hr, *anchor_lines)]
        diag["max_anchor_error_after_m"] = round(float(shift.max()), 3)
        if shift.max() > max_shift_m:
            return H, diag | {"refined": False, "reason": "moved away from manual points"}
    if c1 >= c0:
        return H, diag | {"refined": False, "reason": "no improvement"}
    return Hr, diag | {"refined": True}


def validate(
    H: np.ndarray, image_shape: tuple[int, int], grass: np.ndarray | None = None
) -> tuple[bool, str]:
    """Reject numerically or geometrically implausible image -> pitch transforms."""
    if H is None or not np.all(np.isfinite(H)):
        return False, "non-finite"
    if abs(np.linalg.det(H)) < 1e-12:
        return False, "singular"
    h, w = image_shape
    grid = np.array([[u, v] for v in np.linspace(h * 0.25, h - 1, 4) for u in np.linspace(0, w - 1, 5)])
    denom = (H[2, :2] @ grid.T) + H[2, 2]
    if not (np.all(denom > 0) or np.all(denom < 0)):
        return False, "pitch horizon inside lower image"
    # Main camera sits on the near side: image-right is pitch +x, image-down is towards y = 0.
    c = apply_h(H, np.array([[w / 2, h * 0.6], [w / 2 + 50, h * 0.6], [w / 2, h * 0.6 + 50]]))
    if c[1, 0] <= c[0, 0] or c[2, 1] >= c[0, 1]:
        return False, "mirrored or rotated pitch"
    if grass is not None:
        ys, xs = np.nonzero(grass[::20, ::20])
        if len(xs):
            p = apply_h(H, np.c_[xs * 20, ys * 20].astype(float))
            outside = (p[:, 0] < -15) | (p[:, 0] > L + 15) | (p[:, 1] < -15) | (p[:, 1] > W + 15)
            if outside.mean() > 0.25:
                return False, "visible grass projects far outside pitch"
    return True, "ok"


def frame_homography(H_k: np.ndarray, G_k: np.ndarray, G_t: np.ndarray) -> np.ndarray:
    """Propagate keyframe image->pitch ``H_k`` to frame t of the same motion segment."""
    H = H_k @ np.linalg.inv(G_k) @ G_t
    return H / H[2, 2]


def draw_pitch_overlay(image: np.ndarray, H_img2pitch: np.ndarray, colour=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Project the reference pitch markings into the image (visual calibration check)."""
    out = image.copy()
    Hinv = np.linalg.inv(H_img2pitch)
    h, w = image.shape[:2]
    for poly in pitch_line_samples():
        pts = apply_h(Hinv, poly)
        ok = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 4 * max(h, w)).all(axis=1)
        pts = pts[ok]
        if len(pts) > 1:
            cv2.polylines(out, [pts.astype(np.int32).reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)
    return out
