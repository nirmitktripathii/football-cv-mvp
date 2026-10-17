"""Team assignment from kit colour.

For every tracked person the torso region (rows 15-50 % of the box, the middle
half of its width) is sampled, grass-coloured pixels are dropped, and the mean
CIELAB colour is taken. Each track gets the median of its samples. Tracks are
clustered with k-means (k=2) into the two outfield kits; a track whose colour
is far from its centroid (referee, assistant, goalkeepers) is labelled
``other``. Clusters are named by comparing their centroids with the configured
kit colours, so ``home``/``away`` follow the broadcast, not cluster order.

This is unsupervised and unvalidated: no per-player ground truth exists for
the clip, so the assignment is reported with its cluster distances and a
manual spot-check, never as a measured accuracy.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from football_cv.calibration.shots import GRASS_HIGH, GRASS_LOW

log = logging.getLogger(__name__)

# Approximate kit colours in OpenCV 8-bit Lab (L, a, b with a/b offset by 128).
KIT_LAB = {
    "white": (235, 128, 128),
    "blue": (70, 140, 85),
    "red": (110, 180, 160),
    "yellow": (220, 115, 200),
    "black": (30, 128, 128),
    "orange": (150, 165, 190),
    "green": (120, 90, 150),
}
MIN_TORSO_PIXELS = 12
MIN_SAMPLES_PER_TRACK = 3
MIN_TEAM_AGREEMENT = 0.8     # share of a track's crops that must agree with its label to call it confident
MIN_CONFIDENT_SAMPLES = 5


def torso_colour(image: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = y2 - y1, x2 - x1
    H, W = image.shape[:2]
    ty1, ty2 = max(0, int(y1 + 0.15 * h)), min(H, int(y1 + 0.5 * h))
    tx1, tx2 = max(0, int(x1 + 0.25 * w)), min(W, int(x2 - 0.25 * w))
    if ty2 - ty1 < 2 or tx2 - tx1 < 2:
        return None
    crop = image[ty1:ty2, tx1:tx2]
    not_grass = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), GRASS_LOW, GRASS_HIGH) == 0
    if not_grass.sum() < MIN_TORSO_PIXELS:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[not_grass].mean(axis=0)


def sample_colours(tracks: pd.DataFrame, frames: pd.DataFrame, frames_root, min_box_h: float) -> pd.DataFrame:
    """Torso Lab colour per tracked-person row (rows with too small or grassy crops omitted)."""
    ppl = tracks[(tracks["object_class"] == "person") & ((tracks["bbox_y2"] - tracks["bbox_y1"]) >= min_box_h)]
    paths = frames.set_index("sample_id")["path"]
    out = []
    for sid, grp in ppl.groupby("sample_id"):
        img = cv2.imread(str(frames_root / paths[sid]))
        for r in grp.itertuples():
            c = torso_colour(img, (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2))
            if c is not None:
                out.append({"track_id": r.track_id, "L": c[0], "a": c[1], "b": c[2]})
    return pd.DataFrame(out, columns=["track_id", "L", "a", "b"])


def add_team_confidence(table: pd.DataFrame, colours: pd.DataFrame, centres: np.ndarray, names: dict) -> pd.DataFrame:
    """Add ``agreement`` (share of a track's crops whose nearest kit cluster matches its label) and ``team_status``.

    ``confident`` needs agreement >= ``MIN_TEAM_AGREEMENT`` and >= ``MIN_CONFIDENT_SAMPLES`` crops; other
    home/away labels are ``uncertain``. Labels themselves are not changed.
    """
    d = np.linalg.norm(colours[["L", "a", "b"]].to_numpy()[:, None, :] - centres[None], axis=2)
    crop_team = pd.Series([names[k] for k in d.argmin(1)], index=colours.index)
    label = colours["track_id"].map(table.set_index("track_id")["team_id"])
    agree = (crop_team == label).groupby(colours["track_id"]).mean()
    out = table.copy()
    outfield = out["team_id"].isin(["home", "away"])
    out["agreement"] = out["track_id"].map(agree).where(outfield).round(3)
    sure = (out["agreement"] >= MIN_TEAM_AGREEMENT) & (out["n_colour_samples"] >= MIN_CONFIDENT_SAMPLES)
    out["team_status"] = np.select(
        [out["team_id"] == "unknown", out["team_id"] == "other", sure], ["unknown", "other", "confident"], "uncertain"
    )
    return out


def assign_teams(
    colours: pd.DataFrame, home_kit: str, away_kit: str, other_distance_factor: float, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """Return (track_id -> team_id table, diagnostics)."""
    per_track = colours.groupby("track_id").agg(
        L=("L", "median"), a=("a", "median"), b=("b", "median"), n=("L", "size")
    )
    fit_set = per_track[per_track["n"] >= MIN_SAMPLES_PER_TRACK]
    if len(fit_set) < 4:
        raise RuntimeError(f"Only {len(fit_set)} tracks with enough colour samples; cannot cluster kits")
    X = fit_set[["L", "a", "b"]].to_numpy()
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)

    refs = np.array([KIT_LAB[home_kit], KIT_LAB[away_kit]], dtype=float)
    cost = np.linalg.norm(km.cluster_centers_[:, None, :] - refs[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    names = {int(r): ("home" if c == 0 else "away") for r, c in zip(rows, cols)}

    d_all = np.linalg.norm(per_track[["L", "a", "b"]].to_numpy()[:, None, :] - km.cluster_centers_[None], axis=2)
    nearest, dist = d_all.argmin(1), d_all.min(1)
    d_fit = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    medians = {k: float(np.median(d_fit[km.labels_ == k])) for k in (0, 1)}
    limit = np.array([other_distance_factor * max(medians[k], 1.0) for k in nearest])
    team = np.where(dist > limit, "other", [names[k] for k in nearest])
    team = np.where(per_track["n"].to_numpy() < MIN_SAMPLES_PER_TRACK, "unknown", team)
    table = pd.DataFrame(
        {
            "track_id": per_track.index, "team_id": team, "colour_distance": dist.round(1),
            "n_colour_samples": per_track["n"].to_numpy(),
        }
    )
    table = add_team_confidence(table, colours, km.cluster_centers_, names)
    diag = {
        "centroids_lab": {names[k]: km.cluster_centers_[k].round(1).tolist() for k in (0, 1)},
        "kit_hint": {"home": home_kit, "away": away_kit},
        "centroid_to_hint_distance": {names[int(r)]: float(cost[r, c].round(1)) for r, c in zip(rows, cols)},
        "median_member_distance": {names[k]: round(medians[k], 1) for k in (0, 1)},
        "tracks_by_team": {str(k): int(v) for k, v in table["team_id"].value_counts().items()},
    }
    log.info("Team clustering: %s", diag)
    return table, diag
