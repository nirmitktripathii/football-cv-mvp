"""Conservative tracklet linking: decide which track fragments are the same person.

A tracker breaks a person's trajectory at occlusions and when the camera leaves them. This module
joins fragments (tracklets) only when the evidence is unambiguous, because a wrong join silently
fuses two people into one trajectory while a missed join merely leaves a gap.

A pair (a -> b) is a candidate when all of these hold:
  * same shot (never across a camera cut: position says nothing after a cut),
  * same outfield team, both labelled confidently,
  * b starts after a ends, within ``max_gap_s``,
  * b's first position is consistent with where a was heading (and a's last position with where b
    came from), within a chi-square gate whose spread grows with the gap and includes the
    calibration error of the pitch mapping,
  * the appearance descriptors (kit regions, if supplied) do not differ by more than their spread.

A candidate becomes a link only if it is the *unique* mutual best match with a margin: when a has
another plausible successor, or b another plausible predecessor, that is not clearly worse, the
pair is left unlinked and reported as ambiguous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football_cv.reconstruction.motion_models import MAX_SPEED, SIGMA_ACCEL, SIGMA_MEAS

CHI2_99_2D = 9.21
LINKABLE_TEAMS = ("home", "away")
VELOCITY_FIT_S = 1.0         # window used to estimate a tracklet's entry / exit velocity

DEFAULTS = {
    "max_gap_s": 6.0,
    "gate_chi2": CHI2_99_2D,
    "max_false_matches": 0.5,        # expected unseen team-mates fitting at least as well; above this, never link
    "teammate_density_per_m2": 10 / (105 * 68),   # 10 outfield players over the pitch
    "sigma_velocity_mps": 1.5,       # uncertainty of the entry / exit velocity from a ~1 s fit
    "calibration_sigma_m": 3.0,      # pitch-mapping error between two views (measured, docs/evaluation.md §10)
    "ambiguity_margin": 6.0,         # second-best cost must exceed best by this; 0 blocks any second candidate
    "exclude_edge_breaks": True,     # never link across a break where either end touches the edge of the view
    "require_confident_team": True,
    "min_samples": 3,
    "appearance_gate": 4.0,          # veto when descriptors differ by more than this many spreads
    "appearance_floor": 3.0,         # Lab units; spread never taken below this
}

TRACKLET_COLUMNS = ["track_id", "team_id", "team_status", "shot_id", "t0", "t1", "x0", "y0", "x1", "y1",
                    "vx0", "vy0", "vx1", "vy1", "edge0", "edge1", "n"]
LINK_COLUMNS = ["pred_track_id", "succ_track_id", "gap_s", "cost", "false_matches", "margin", "status"]


def settings(cfg: dict | None) -> dict:
    return {**DEFAULTS, **((cfg or {}).get("identity") or {})}


def _end_velocity(t: np.ndarray, xy: np.ndarray, at_end: bool) -> np.ndarray:
    """Least-squares velocity over the last / first ``VELOCITY_FIT_S`` seconds, speed-capped."""
    keep = t >= t[-1] - VELOCITY_FIT_S if at_end else t <= t[0] + VELOCITY_FIT_S
    tt, pp = t[keep], xy[keep]
    if len(tt) < 3:
        return np.zeros(2)
    v = np.polyfit(tt - tt.mean(), pp, 1)[0]
    speed = float(np.linalg.norm(v))
    return v if speed <= MAX_SPEED else v * (MAX_SPEED / speed)


def tracklet_summary(traj: pd.DataFrame, min_samples: int = DEFAULTS["min_samples"]) -> pd.DataFrame:
    """One row per (track, shot): endpoints, entry / exit velocity, team, from observed positions.

    Rows flagged ``jump`` or without a pitch position are ignored. Ids are unique across shots in
    this pipeline, so grouping by shot too is a guard rather than the usual case.
    """
    good = traj[traj["x_s"].notna() & ~traj["jump"].astype(bool)]
    rows = []
    for (tid, shot), g in good.groupby(["track_id", "shot_id"]):
        g = g.sort_values("timestamp_s")
        if len(g) < min_samples:
            continue
        t, xy = g["timestamp_s"].to_numpy(float), g[["x_s", "y_s"]].to_numpy(float)
        v0, v1 = _end_velocity(t, xy, False), _end_velocity(t, xy, True)
        rows.append({
            "track_id": tid, "team_id": g["team_id"].iloc[0],
            "team_status": g["team_status"].iloc[0] if "team_status" in g else "confident",
            "shot_id": shot, "t0": t[0], "t1": t[-1], "x0": xy[0, 0], "y0": xy[0, 1], "x1": xy[-1, 0],
            "y1": xy[-1, 1], "vx0": v0[0], "vy0": v0[1], "vx1": v1[0], "vy1": v1[1],
            "edge0": bool(g["edge"].iloc[0]) if "edge" in g else False,
            "edge1": bool(g["edge"].iloc[-1]) if "edge" in g else False, "n": len(g),
        })
    return pd.DataFrame(rows, columns=TRACKLET_COLUMNS)


def _prediction_var(gap_s: np.ndarray, s: dict) -> np.ndarray:
    """Variance (m^2, per axis) of an extrapolation over ``gap_s`` under a white-acceleration model."""
    return (s["calibration_sigma_m"] ** 2 + SIGMA_MEAS**2 + (s["sigma_velocity_mps"] * gap_s) ** 2
            + SIGMA_ACCEL**2 * gap_s**3 / 3.0)


def _appearance_z(pred: pd.Series, succ: pd.Series, app: pd.DataFrame, s: dict) -> np.ndarray:
    """Descriptor distance in units of the two tracks' own spread; NaN when either lacks a descriptor."""
    cols = [c for c in app.columns if c.startswith("d_")]
    idx = app.set_index("track_id")
    a, b = idx.reindex(pred.to_numpy()), idx.reindex(succ.to_numpy())
    dist = np.linalg.norm(a[cols].to_numpy(float) - b[cols].to_numpy(float), axis=1)
    spread = np.sqrt(a["spread"].to_numpy(float) ** 2 + b["spread"].to_numpy(float) ** 2 + s["appearance_floor"] ** 2)
    return dist / spread


def candidate_pairs(trk: pd.DataFrame, s: dict, appearance: pd.DataFrame | None = None) -> pd.DataFrame:
    """All plausible (pred, succ) pairs with their cost; the cost is the mean of the two chi-square terms."""
    a_idx, b_idx = np.meshgrid(np.arange(len(trk)), np.arange(len(trk)), indexing="ij")
    A, B = trk.iloc[a_idx.ravel()].reset_index(drop=True), trk.iloc[b_idx.ravel()].reset_index(drop=True)
    gap = (B["t0"] - A["t1"]).to_numpy(float)
    ok = (A["shot_id"] == B["shot_id"]) & (A["track_id"] != B["track_id"]) & (gap > 0) & (gap <= s["max_gap_s"])
    ok &= (A["team_id"] == B["team_id"]) & A["team_id"].isin(LINKABLE_TEAMS)
    if s["require_confident_team"]:
        ok &= (A["team_status"] == "confident") & (B["team_status"] == "confident")
    if s["exclude_edge_breaks"]:
        ok &= ~(A["edge1"].astype(bool) | B["edge0"].astype(bool))
    ok = ok.to_numpy()
    A, B, gap = A[ok].reset_index(drop=True), B[ok].reset_index(drop=True), gap[ok]
    if not len(A):
        return pd.DataFrame(columns=["pred", "succ", "gap_s", "cost", "false_matches"])
    var = _prediction_var(gap, s)
    fwd = np.c_[B["x0"] - (A["x1"] + A["vx1"] * gap), B["y0"] - (A["y1"] + A["vy1"] * gap)]
    bwd = np.c_[A["x1"] - (B["x0"] - B["vx0"] * gap), A["y1"] - (B["y0"] - B["vy0"] * gap)]
    z_f, z_b = (fwd**2).sum(1) / var, (bwd**2).sum(1) / var
    keep = np.maximum(z_f, z_b) <= s["gate_chi2"]
    # a person cannot cover the straight-line distance faster than the speed cap (plus mapping error)
    dist = np.hypot(B["x0"] - A["x1"], B["y0"] - A["y1"]).to_numpy()
    keep &= dist <= MAX_SPEED * gap + 2 * s["calibration_sigma_m"]
    out = pd.DataFrame({"pred": A["track_id"], "succ": B["track_id"], "gap_s": gap, "cost": 0.5 * (z_f + z_b)})
    # Expected number of *unobserved* team-mates inside the region at least as good as this pair
    # (density x area). Rivals that are observed are handled by decide_links; this bounds the unseen.
    out["false_matches"] = s["teammate_density_per_m2"] * np.pi * out["cost"] * var
    if appearance is not None:
        app_z = pd.Series(_appearance_z(out["pred"], out["succ"], appearance, s)).fillna(0.0)
        keep &= (app_z <= s["appearance_gate"]).to_numpy()
        out["cost"] = out["cost"] + 0.5 * app_z**2
    return out[keep].reset_index(drop=True)


def decide_links(cands: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Keep a pair only if it is the mutual best match and clearly better than every rival.

    One row per candidate. ``margin`` is the cost of the best rival (another successor of the same
    predecessor, or another predecessor of the same successor) minus this pair's cost; ``status`` is
    ``linked`` (margin above ``ambiguity_margin``), ``ambiguous`` (a rival is not clearly worse),
    ``not_best`` (a rival is better) or ``weak`` (unambiguous among observed tracklets but an unseen
    team-mate would plausibly fit as well: ``false_matches`` above ``max_false_matches``).
    """
    if cands.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)
    c = cands.reset_index(drop=True)

    def rival_gap(key: str) -> np.ndarray:
        out = np.full(len(c), np.inf)
        for _, idx in c.groupby(key).groups.items():
            if len(idx) < 2:
                continue
            costs = c.loc[idx, "cost"].to_numpy()
            for k, i in enumerate(idx):
                out[i] = np.delete(costs, k).min() - costs[k]
        return out

    gap_to_rival = np.minimum(rival_gap("pred"), rival_gap("succ"))
    margin = float(s["ambiguity_margin"])
    clear = gap_to_rival > margin if margin > 0 else np.isinf(gap_to_rival)
    c["margin"] = gap_to_rival
    c["status"] = np.where(clear, "linked", np.where(gap_to_rival < 0, "not_best", "ambiguous"))
    c.loc[(c["status"] == "linked") & (c["false_matches"] > s["max_false_matches"]), "status"] = "weak"
    return c.rename(columns={"pred": "pred_track_id", "succ": "succ_track_id"})[LINK_COLUMNS]


def assign_identities(track_ids: pd.Series | np.ndarray, links: pd.DataFrame) -> pd.DataFrame:
    """Map each track to an identity id: the smallest track id of its chain of linked fragments."""
    parent = {int(t): int(t) for t in track_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in links[links["status"] == "linked"].itertuples():
        ra, rb = find(int(r.pred_track_id)), find(int(r.succ_track_id))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return pd.DataFrame({"track_id": list(parent), "identity_id": [find(t) for t in parent]})


def link_tracklets(traj: pd.DataFrame, cfg: dict | None = None, appearance: pd.DataFrame | None = None):
    """Run the linker. Returns (identity map, decisions table, tracklet summary)."""
    s = settings(cfg)
    trk = tracklet_summary(traj, s["min_samples"])
    decisions = decide_links(candidate_pairs(trk, s, appearance), s)
    ids = pd.concat([traj["track_id"], trk["track_id"]]).drop_duplicates()
    return assign_identities(ids, decisions), decisions, trk
