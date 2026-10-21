"""Role labels from appearance and pitch position: SRITrack probe plus a goal-centre keeper prior (docs/role_probe.md).

The probe is a multinomial logistic regression on the SRITrack ``fc`` embedding of a person crop (player, goalkeeper,
referee). It labels referees well but misses near-black goalkeepers, so goalkeepers also come from a position prior:
per frame and per goal end, the on-pitch non-referee person nearest the goal centre inside the penalty area is the
keeper, with the probe keeper probability breaking ties within ``tie_m``. Both parts are deterministic given the model
file; the model is saved as plain arrays (no pickle).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_cv.calibration.motion import apply_h
from football_cv.calibration.projection import H_of

log = logging.getLogger(__name__)

CLASSES = ("player", "goalkeeper", "referee")
USABLE_STATUS = ("keyframe", "propagated", "propagated_unrefined")
KEEPER_RULES = ("prior_or_probe", "prior_only", "prior_or_probe_in_area")
SEED = 0


@dataclass(frozen=True)
class RoleModel:
    """Standardiser + multinomial logistic regression over ``CLASSES`` (in that column order)."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # (3, d)
    intercept: np.ndarray  # (3,)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, seed: int = SEED) -> "RoleModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(x)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(sc.transform(x), y)
        order = [list(clf.classes_).index(c) for c in CLASSES]
        return cls(sc.mean_, sc.scale_, clf.coef_[order], clf.intercept_[order])

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.coef.T + self.intercept
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept,
                 classes=np.array(CLASSES))

    @classmethod
    def load(cls, path: str | Path) -> "RoleModel":
        z = np.load(path, allow_pickle=False)
        if tuple(z["classes"]) != CLASSES:
            raise ValueError(f"{path}: classes {tuple(z['classes'])} do not match {CLASSES}")
        return cls(z["mean"], z["scale"], z["coef"], z["intercept"])


def pitch_positions(tracks: pd.DataFrame, calib: pd.DataFrame) -> np.ndarray:
    """(n, 2) foot position in metres per row; NaN where the frame has no usable calibration."""
    xy = np.full((len(tracks), 2), np.nan)
    cal = calib.set_index("sample_id")
    sid = tracks["sample_id"].to_numpy()
    feet = tracks[["foot_x", "foot_y"]].to_numpy(float)
    for s in np.unique(sid):
        if s not in cal.index or cal.at[s, "calibration_status"] not in USABLE_STATUS:
            continue
        rows = np.flatnonzero(sid == s)
        xy[rows] = apply_h(H_of(cal.loc[s]), feet[rows])
    return xy


def keeper_prior(frame: np.ndarray, xy: np.ndarray, eligible: np.ndarray, p_keeper: np.ndarray,
                 pitch: tuple[float, float], pen_depth: float, pen_half_width: float, tie_m: float) -> np.ndarray:
    """True for the box chosen as keeper at each goal end of each frame (see module docstring)."""
    length, width = pitch
    fires = np.zeros(len(frame), bool)
    on_pitch = (xy[:, 0] >= 0) & (xy[:, 0] <= length) & (xy[:, 1] >= 0) & (xy[:, 1] <= width)
    ok = eligible & on_pitch
    for idx in pd.Series(np.arange(len(frame))).groupby(frame).indices.values():
        idx = idx[ok[idx]]
        for end_x in (0.0, length):
            near_end = (np.abs(xy[idx, 0] - end_x) <= pen_depth) & (np.abs(xy[idx, 1] - width / 2) <= pen_half_width)
            cand = idx[near_end]
            if not len(cand):
                continue
            d = np.hypot(xy[cand, 0] - end_x, xy[cand, 1] - width / 2)
            tied = cand[d <= d.min() + tie_m]
            fires[tied[int(np.argmax(p_keeper[tied]))]] = True
    return fires


def in_penalty_area(xy: np.ndarray, pitch: tuple[float, float], pen_depth: float, pen_half_width: float) -> np.ndarray:
    """True where the position lies inside the penalty-area rectangle of either goal (NaN positions are False)."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        near_goal = np.minimum(xy[:, 0], length - xy[:, 0]) <= pen_depth
        inside = (xy[:, 0] >= 0) & (xy[:, 0] <= length) & (np.abs(xy[:, 1] - width / 2) <= pen_half_width)
    return near_goal & inside


def assign_roles(proba: np.ndarray, fires: np.ndarray, rule: str, in_area: np.ndarray | None = None) -> np.ndarray:
    """Final role per row from probe probabilities and the prior: referee, goalkeeper, else player.

    ``prior_or_probe_in_area`` accepts a probe-only keeper call only where ``in_area`` is True.
    """
    if rule not in KEEPER_RULES:
        raise ValueError(f"keeper_rule must be one of {KEEPER_RULES}, got {rule!r}")
    top = proba.argmax(axis=1)
    probe_keeper = top == CLASSES.index("goalkeeper")
    if rule == "prior_or_probe_in_area":
        if in_area is None:
            raise ValueError("keeper_rule prior_or_probe_in_area needs in_area")
        keeper = fires | (probe_keeper & in_area)
    else:
        keeper = fires | probe_keeper if rule == "prior_or_probe" else fires
    return np.where(top == CLASSES.index("referee"), "referee", np.where(keeper, "goalkeeper", "player"))


def crops_of(tracks: pd.DataFrame, rows: np.ndarray, frames: pd.DataFrame, frames_root, min_side: int,
             edge_margin: int | None = None, min_height: int | None = None):
    """Crops for ``rows`` (positional) as (crop list, positions that produced a crop).

    ``edge_margin`` skips a box whose raw coordinates come within that many px of the frame edge (a cut-off person);
    ``min_height`` skips boxes shorter than that. Both are off when None.
    """
    paths = frames.set_index("sample_id")["path"]
    crops, kept = [], []
    sid = tracks["sample_id"].to_numpy()
    box = tracks[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    for s in np.unique(sid[rows]):
        img = cv2.imread(str(Path(frames_root) / paths[s]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for i in rows[sid[rows] == s]:
            if edge_margin is not None and (box[i, 0] <= edge_margin or box[i, 1] <= edge_margin
                                            or box[i, 2] >= w - edge_margin or box[i, 3] >= h - edge_margin):
                continue
            if min_height is not None and box[i, 3] - box[i, 1] < min_height:
                continue
            x1, y1, x2, y2 = max(int(box[i, 0]), 0), max(int(box[i, 1]), 0), min(int(box[i, 2]), w), min(int(box[i, 3]), h)
            if x2 - x1 >= min_side and y2 - y1 >= min_side:
                crops.append(img[y1:y2, x1:x2].copy())
                kept.append(i)
    return crops, np.array(kept, dtype=int)


def relabel(tracks: pd.DataFrame, calib: pd.DataFrame, frames: pd.DataFrame, frames_root, model: RoleModel, embedder,
            kit_team: dict[int, str], s: dict, pitch: tuple[float, float]) -> tuple[pd.DataFrame, dict]:
    """Return ``tracks`` with probe roles. ``role_pipeline`` keeps the detector role, ``role_source`` says what decided.

    A person row is relabelled only when it has a crop (at least ``min_side_px`` on each side) and a usable calibration.
    Every other row, and all non-person rows, keeps its detector role: the probe alone produced 20 of the 22 false
    keepers in stage 1d, and the prior needs a position, so neither is trusted without both.
    """
    out = tracks.copy().reset_index(drop=True)
    out["role_pipeline"] = out["role"]
    out["role_source"] = "detector"
    person = ((out["object_class"] == "person") & (out["track_id"] >= 0)).to_numpy()
    crops, kept = crops_of(out, np.flatnonzero(person), frames, frames_root, int(s["min_side_px"]),
                           s.get("edge_margin_px"), s.get("min_height_px"))
    proba = np.full((len(out), len(CLASSES)), np.nan)
    if len(crops):
        proba[kept] = model.proba(embedder.embed(crops)[s.get("feature", "fc")])
    xy = pitch_positions(out, calib)
    has = ~np.isnan(proba[:, 0])
    scorable = has & ~np.isnan(xy[:, 0])
    top_ref = np.zeros(len(out), bool)
    top_ref[has] = proba[has].argmax(axis=1) == CLASSES.index("referee")
    p_keeper = np.nan_to_num(proba[:, CLASSES.index("goalkeeper")])
    fires = keeper_prior(out["sample_id"].to_numpy(), xy, scorable & ~top_ref, p_keeper, pitch,
                         float(s["penalty_depth_m"]), float(s["penalty_half_width_m"]), float(s["tie_m"]))
    new_role = np.array(out["role"], dtype=object)
    in_area = in_penalty_area(xy, pitch, float(s["penalty_depth_m"]), float(s["penalty_half_width_m"]))
    new_role[scorable] = assign_roles(proba[scorable], fires[scorable], str(s["keeper_rule"]), in_area[scorable])
    changed = scorable & (new_role != out["role"].to_numpy(object))
    out["role"] = new_role
    out.loc[scorable, "role_source"] = np.where(fires, "prior", "probe")[scorable]
    out["role_p_goalkeeper"] = proba[:, CLASSES.index("goalkeeper")]
    out["role_p_referee"] = proba[:, CLASSES.index("referee")]
    team = out["team_id"].astype(object).to_numpy()
    to_ref, to_gk, to_pl = (changed & (new_role == r) for r in ("referee", "goalkeeper", "player"))
    team[to_ref], team[to_gk] = "other", "unknown"
    team[to_pl] = [kit_team.get(int(t), "unknown") for t in out.loc[to_pl, "track_id"]]
    out["team_id"] = team
    report = {
        "person_rows": int(person.sum()), "scored_rows": int(scorable.sum()),
        "no_crop_rows": int(person.sum() - has.sum()), "no_calibration_rows": int((has & ~scorable).sum()),
        "changed_rows": int(changed.sum()), "prior_fired_rows": int(fires.sum()), "keeper_rule": s["keeper_rule"],
        "roles_before": {k: int(v) for k, v in out.loc[person, "role_pipeline"].value_counts().items()},
        "roles_after": {k: int(v) for k, v in out.loc[person, "role"].value_counts().items()},
    }
    return out, report
