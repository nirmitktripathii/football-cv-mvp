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

from football_cv.calibration.shots import kit_pixel_mask

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
MIN_KIT_SHARE = 0.15       # each team's shirt covers far more of the crops than this
MAX_KIT_REFITS = 3


def torso_colour(image: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = y2 - y1, x2 - x1
    H, W = image.shape[:2]
    ty1, ty2 = max(0, int(y1 + 0.15 * h)), min(H, int(y1 + 0.5 * h))
    tx1, tx2 = max(0, int(x1 + 0.25 * w)), min(W, int(x2 - 0.25 * w))
    if ty2 - ty1 < 2 or tx2 - tx1 < 2:
        return None
    crop = image[ty1:ty2, tx1:tx2]
    not_grass = kit_pixel_mask(crop)
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
                out.append({"track_id": r.track_id, "sample_id": sid, "L": c[0], "a": c[1], "b": c[2]})
    return pd.DataFrame(out, columns=["track_id", "sample_id", "L", "a", "b"])


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


def _fit_kits(crops: np.ndarray, seed: int) -> KMeans:
    """2-means over crop colours; a cluster holding < MIN_KIT_SHARE of the crops is an outlier group (referee,
    glare, a third kit), so its crops are dropped and the fit repeated."""
    keep = np.ones(len(crops), bool)
    for _ in range(MAX_KIT_REFITS):
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(crops[keep])
        share = np.bincount(km.labels_, minlength=2) / keep.sum()
        if share.min() >= MIN_KIT_SHARE:
            break
        keep[np.flatnonzero(keep)[km.labels_ == share.argmin()]] = False
    return km


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
    # Fit on individual crops, not per-track medians: a track that switches between opponents has a median between
    # the two kits, and enough of those (or a few bright referee tracks) pulled the per-track clusters off the kits.
    crops = colours[colours["track_id"].isin(fit_set.index)][["L", "a", "b"]].to_numpy()
    km = _fit_kits(crops, seed)
    X = fit_set[["L", "a", "b"]].to_numpy()
    fit_labels = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None], axis=2).argmin(1)

    refs = np.array([KIT_LAB[home_kit], KIT_LAB[away_kit]], dtype=float)
    cost = np.linalg.norm(km.cluster_centers_[:, None, :] - refs[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    names = {int(r): ("home" if c == 0 else "away") for r, c in zip(rows, cols)}

    d_all = np.linalg.norm(per_track[["L", "a", "b"]].to_numpy()[:, None, :] - km.cluster_centers_[None], axis=2)
    nearest, dist = d_all.argmin(1), d_all.min(1)
    d_fit = np.linalg.norm(X - km.cluster_centers_[fit_labels], axis=1)
    medians = {k: float(np.median(d_fit[fit_labels == k])) if (fit_labels == k).any() else 1.0 for k in (0, 1)}
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


KIT_ROLES = ("", "player")  # roles whose shirt colour says which team a person plays for


def majority_role(tracks: pd.DataFrame) -> pd.Series:
    """Per person track, the detector role seen most often ("" when the detector gave none)."""
    people = tracks[(tracks["track_id"] >= 0) & (tracks["object_class"] == "person")]
    return people.groupby("track_id")["role"].agg(
        lambda r: r[r != ""].mode().iloc[0] if (r != "").any() else ""
    )


def apply_roles(tracks: pd.DataFrame, table: pd.DataFrame, roles: pd.Series) -> pd.DataFrame:
    """Add the track-level ``role`` and a ``team_id`` that respects it.

    Referees belong to no team ("other"). Goalkeepers wear a third kit, so their colour cannot name a team;
    they stay "unknown" until a position-based rule can (needs the calibrated pitch).
    """
    out = tracks.merge(table[["track_id", "team_id"]], on="track_id", how="left")
    out["role"] = out["track_id"].map(roles).fillna("")
    out.loc[out["role"] == "referee", "team_id"] = "other"
    out.loc[out["role"] == "goalkeeper", "team_id"] = "unknown"
    return out


def team_switch_path(evidence: np.ndarray, switch_cost: float) -> np.ndarray:
    """Most likely kit per sample (0 = home, 1 = away) for one track, by Viterbi over two states.

    ``evidence`` is the per-sample log-likelihood ratio for home over away (0 where no crop was read); changing
    state costs ``switch_cost``. One contrary crop cannot flip the path; a run of them whose total evidence beats the
    switch cost does, at the sample where the shirt changes.
    """
    emit = np.c_[evidence, -evidence]
    score = emit[0].copy()
    back = np.zeros((len(evidence), 2), dtype=int)
    for t in range(1, len(evidence)):
        stay, switch = score, score[::-1] - switch_cost
        back[t] = np.where(stay >= switch, [0, 1], [1, 0])
        score = np.maximum(stay, switch) + emit[t]
    path = np.empty(len(evidence), dtype=int)
    path[-1] = int(score.argmax())
    for t in range(len(evidence) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth_teams(tracks: pd.DataFrame, colours: pd.DataFrame, centres: dict, params: dict) -> pd.DataFrame:
    """Give every row of a home/away track its own team, from that track's shirt colour over time.

    A tracker that swaps two opponents during an occlusion leaves one track id on both players; a single team per
    track then mislabels everything after the swap. Here each crop votes by how much nearer it is to one kit centre
    than the other (``llr_scale`` Lab units per unit of evidence, capped at ``max_llr``), and ``team_switch_path``
    decides where the shirt really changed. Rows of referees, goalkeepers, "other"/"unknown" tracks, and tracks whose
    total evidence is under ``min_evidence``, keep their label.
    """
    out = tracks.copy()
    home, away = np.asarray(centres["home"], float), np.asarray(centres["away"], float)
    lab = colours[["L", "a", "b"]].to_numpy(float)
    llr = (np.linalg.norm(lab - away, axis=1) - np.linalg.norm(lab - home, axis=1)) / float(params["llr_scale"])
    llr = pd.Series(np.clip(llr, -params["max_llr"], params["max_llr"]),
                    index=pd.MultiIndex.from_frame(colours[["track_id", "sample_id"]]))
    llr = llr.groupby(level=[0, 1]).mean()
    eligible = (out["object_class"] == "person") & out["team_id"].isin(["home", "away"]) & out["role"].isin(KIT_ROLES)
    for tid, grp in out[eligible].sort_values("sample_id").groupby("track_id"):
        keys = pd.MultiIndex.from_arrays([np.full(len(grp), tid), grp["sample_id"].to_numpy()])
        ev = llr.reindex(keys).fillna(0.0).to_numpy()
        if np.abs(ev).sum() < float(params["min_evidence"]):
            continue
        path = team_switch_path(ev, float(params["switch_cost"]))
        out.loc[grp.index, "team_id"] = np.where(path == 0, "home", "away")
    return out


def _runs(flags: np.ndarray, min_len: int):
    """(start, end) index pairs, end inclusive, of the stretches of ``flags`` that are True for at least ``min_len``."""
    padded = np.r_[False, flags, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(a, b - 1) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def mark_outlier_runs(tracks: pd.DataFrame, colours: pd.DataFrame, centres: dict, params: dict) -> pd.DataFrame:
    """Relabel the rows of a home/away track whose crops stay far from both kits: another person is on that id.

    After an identity switch the referee (or a keeper) rides inside a player's track, and a single team per track, or
    the two-state smoothing above, paints him as a home or away player. A run of at least ``min_run`` consecutive
    crops each farther than ``min_gap`` kit-centre separations from both kit centres is not a team-mate, so its rows
    (first to last crop of the run, crop-less samples included) become ``other``/``referee``; when the detector called
    most of them goalkeeper (``detector_role``) they become ``unknown``/``goalkeeper``. Shorter runs, and tracks that
    are already other, unknown, referee or goalkeeper, are left alone.
    """
    out = tracks.copy()
    home, away = np.asarray(centres["home"], float), np.asarray(centres["away"], float)
    limit = float(params["min_gap"]) * float(np.linalg.norm(home - away))
    lab = colours[["L", "a", "b"]].to_numpy(float)
    far = pd.Series(np.minimum(np.linalg.norm(lab - home, axis=1), np.linalg.norm(lab - away, axis=1)) > limit,
                    index=pd.MultiIndex.from_frame(colours[["track_id", "sample_id"]]))
    far = far.groupby(level=[0, 1]).max()
    eligible = (out["object_class"] == "person") & out["team_id"].isin(["home", "away"]) & out["role"].isin(KIT_ROLES)
    for tid, grp in out[eligible].sort_values("sample_id").groupby("track_id"):
        sid = grp["sample_id"].to_numpy()
        flags = far.reindex(pd.MultiIndex.from_arrays([np.full(len(grp), tid), sid]))
        has_crop = flags.notna().to_numpy()
        crop_pos = np.flatnonzero(has_crop)
        for start, end in _runs(flags.to_numpy()[crop_pos].astype(bool), int(params["min_run"])):
            rows = grp.index[(sid >= sid[crop_pos[start]]) & (sid <= sid[crop_pos[end]])]
            keeper = "detector_role" in out and (out.loc[rows, "detector_role"] == "goalkeeper").mean() > 0.5
            out.loc[rows, ["team_id", "role"]] = ["unknown", "goalkeeper"] if keeper else ["other", "referee"]
    return out


def drop_implausible_goalkeepers(tracks: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Drop short goalkeeper tracks far smaller than the players: pitch marks and ad boards the detector calls a keeper.
    Real keepers are full-size (0.8-1.1 x the median player track) and last many samples; the false ones are 0.3-0.7 x
    and a handful of rows. Needs player tracks as the size reference; without any, nothing is dropped."""
    box_h = tracks["bbox_y2"] - tracks["bbox_y1"]
    per_track = box_h.groupby(tracks["track_id"]).median()
    role = tracks.groupby("track_id")["role"].agg(lambda s: s.mode().iloc[0])
    rows = tracks.groupby("track_id").size()
    players = per_track[role == "player"]
    if players.empty:
        return tracks
    small = per_track < float(params["min_rel_height"]) * players.median()
    drop = role.index[(role == "goalkeeper") & small & (rows <= int(params["max_rows"]))]
    return tracks[~tracks["track_id"].isin(drop)]


def split_team_switches(tracks: pd.DataFrame) -> pd.DataFrame:
    """Cut a person track wherever its per-row team changes; each later piece gets a new, unused track id."""
    out = tracks.copy()
    people = out[(out["object_class"] == "person") & (out["track_id"] >= 0)].sort_values(["track_id", "sample_id"])
    piece = (people["team_id"] != people.groupby("track_id")["team_id"].shift()).groupby(people["track_id"]).cumsum() - 1
    later = piece[piece > 0]
    if later.empty:
        return out
    keys = pd.MultiIndex.from_arrays([people.loc[later.index, "track_id"], later])
    codes = pd.factorize(keys)[0]
    out.loc[later.index, "track_id"] = int(out["track_id"].max()) + 1 + codes
    n = len(pd.unique(keys))
    log.info("Team switches: %d track pieces cut from %d tracks", n, people.loc[later.index, "track_id"].nunique())
    return out
