"""Probable regions for players who are not observed: an *inferred* patch, never a position.

For every unobserved sample two Gaussian beliefs about where the player is are fused:

* **motion**  the bracketing / last observations through the motion model (``x_est_m``, ``y_est_m``,
  ``sigma_m`` of ``player_states``): where the player can have gone since they were last seen
  (and, if they were seen again, where they must be heading);
* **shape**   the player's habitual offset from the centroid of the team-mates that *are* observed at
  that moment (a crude formation prior learned from this identity's own observed samples).

The fused Gaussian is then widened by the pitch-mapping error (``calibration_sigma_m``), which both
sources inherit from the same homography. The result is a centre, a covariance and the area of the
``region_mass`` (default 90 %) ellipse. Output rows are tagged ``source = INFERRED`` and are written to
their own table: they are never merged into ``player_states`` and never carry a dot-like coordinate.

Scope: this is a calibrated belief region, not an identification. It says nothing about *which*
off-screen player is which beyond the identity it was computed for, and the shape prior is only as
good as the number of observed team-mates (``min_teammates``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from football_cv.reconstruction import motion_models as mm

PITCH = (105.0, 68.0)
SOURCE_LABEL = "INFERRED"

DEFAULTS = {
    "calibration_sigma_m": 5.0,   # pitch-mapping error per axis (about 4 m perpendicular on propagated frames, docs/evaluation.md section 10), added after fusion
    "shape_sigma_floor_m": 6.0,   # a formation role never pins a player tighter than this
    "min_teammates": 3,           # observed team-mates needed for a centroid
    "region_mass": 0.9,
    "use_motion": True,
    "use_shape": True,
    "time_key": "timestamp_s",
    "horizon_s": 8.0,             # how long before / after its observed span an identity keeps a region
    "team_size": 11,              # players per side: bounds how many unseen players there can be
    "exclude_observed_mass": 0.5,  # a region this likely to hold an observed team-mate is a stale fragment, not a new player
    "sigma_growth_per_m": 0.15,   # motion sigma -> sigma * (1 + g * sigma); tuned on period 1 (docs/evaluation.md section 11)
}
REGION_COLUMNS = ["time_key", "identity_id", "team_id", "cx", "cy", "sxx", "syy", "area_m2", "components", "source"]


def settings(cfg: dict | None) -> dict:
    return DEFAULTS | ((cfg or {}).get("regions") or {})


def team_centroid_excluding_self(df: pd.DataFrame, observed: pd.Series, key: str, min_teammates: int) -> pd.DataFrame:
    """Mean position of the *other* observed team-mates at each row's time (NaN with too few)."""
    obs = df[observed]
    grp = obs.groupby([key, "team_id"])
    total = grp[["x_m", "y_m"]].sum().rename(columns={"x_m": "sx", "y_m": "sy"}).join(grp.size().rename("n"))
    j = df[[key, "team_id"]].join(total, on=[key, "team_id"]).fillna({"sx": 0.0, "sy": 0.0, "n": 0})
    own = observed.to_numpy()
    n = j["n"].to_numpy() - own
    sx = j["sx"].to_numpy() - np.where(own, df["x_m"].fillna(0.0), 0.0)
    sy = j["sy"].to_numpy() - np.where(own, df["y_m"].fillna(0.0), 0.0)
    ok = n >= min_teammates
    return pd.DataFrame({"gx": np.where(ok, sx / np.maximum(n, 1), np.nan),
                         "gy": np.where(ok, sy / np.maximum(n, 1), np.nan)}, index=df.index)


def shape_offsets(df: pd.DataFrame, observed: pd.Series, centroid: pd.DataFrame, floor_m: float) -> pd.DataFrame:
    """Per identity: median offset from the team centroid and its robust spread, from observed samples only."""
    d = pd.DataFrame({"identity_id": df["identity_id"], "ox": df["x_m"] - centroid["gx"], "oy": df["y_m"] - centroid["gy"]})
    d = d[observed & centroid["gx"].notna()]
    med = d.groupby("identity_id")[["ox", "oy"]].median()
    spread = d.join(med, on="identity_id", rsuffix="_m")
    mad_x = (spread["ox"] - spread["ox_m"]).abs().groupby(spread["identity_id"]).median() * 1.4826
    mad_y = (spread["oy"] - spread["oy_m"]).abs().groupby(spread["identity_id"]).median() * 1.4826
    return med.assign(sx=np.maximum(mad_x, floor_m), sy=np.maximum(mad_y, floor_m))


def fuse(mu_a, var_a, mu_b, var_b):
    """Product of two axis-aligned Gaussians (per axis, precision weighted)."""
    prec = 1.0 / var_a + 1.0 / var_b
    return (mu_a / var_a + mu_b / var_b) / prec, 1.0 / prec


def infer_regions(states: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Region rows for every unobserved sample of ``states`` (``player_states`` columns, see module doc)."""
    s = settings(cfg)
    key = s["time_key"]
    df = states.reset_index(drop=True)
    observed = df["position_source"].eq("OBSERVED")
    centroid = team_centroid_excluding_self(df, observed, key, s["min_teammates"])
    offsets = shape_offsets(df, observed, centroid, s["shape_sigma_floor_m"])
    o = offsets.reindex(df["identity_id"])
    o.index = df.index

    mx, my = df["x_est_m"].to_numpy(float), df["y_est_m"].to_numpy(float)
    sd = np.maximum(df["sigma_m"].to_numpy(float), 0.5)
    mv = (sd * (1.0 + s["sigma_growth_per_m"] * sd)) ** 2
    have_motion = s["use_motion"] & np.isfinite(mx) & np.isfinite(my)
    shx, shy = (centroid["gx"] + o["ox"]).to_numpy(float), (centroid["gy"] + o["oy"]).to_numpy(float)
    have_shape = s["use_shape"] & np.isfinite(shx) & np.isfinite(shy) & o["sx"].notna().to_numpy()

    cx = np.full(len(df), np.nan)
    cy = np.full(len(df), np.nan)
    vx = np.full(len(df), np.nan)
    vy = np.full(len(df), np.nan)
    both = have_motion & have_shape
    cx[both], vx[both] = fuse(mx[both], mv[both], shx[both], o["sx"].to_numpy(float)[both] ** 2)
    cy[both], vy[both] = fuse(my[both], mv[both], shy[both], o["sy"].to_numpy(float)[both] ** 2)
    only_m = have_motion & ~have_shape
    cx[only_m], cy[only_m], vx[only_m], vy[only_m] = mx[only_m], my[only_m], mv[only_m], mv[only_m]
    only_s = have_shape & ~have_motion
    cx[only_s], cy[only_s] = shx[only_s], shy[only_s]
    vx[only_s], vy[only_s] = o["sx"].to_numpy(float)[only_s] ** 2, o["sy"].to_numpy(float)[only_s] ** 2
    vx, vy = vx + s["calibration_sigma_m"] ** 2, vy + s["calibration_sigma_m"] ** 2

    r2 = chi2.ppf(s["region_mass"], 2)
    comp = np.select([both, only_m, only_s], ["motion+shape", "motion", "shape"], "")
    out = pd.DataFrame({
        "time_key": df[key], "identity_id": df["identity_id"], "team_id": df["team_id"], "cx": cx, "cy": cy,
        "sxx": vx, "syy": vy, "area_m2": np.pi * r2 * np.sqrt(vx * vy), "components": comp, "source": SOURCE_LABEL,
    })
    return out[~observed & np.isfinite(cx)].reset_index(drop=True)[REGION_COLUMNS]


def mark_plausible_unseen(regions: pd.DataFrame, states: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add ``plausible_unseen``: which regions may stand for a player who is really off-screen.

    Without re-identification every track fragment gets a region, including fragments of players who are
    visible right now under another id. Two rules thin them out, per team and time: a region whose
    ``exclude_observed_mass`` ellipse holds an observed team-mate is dropped (that person is probably
    the one on screen), and at most ``team_size`` minus the observed count of the team survive, smallest
    area first. Only ``home`` and ``away`` regions can qualify; goalkeepers and referees labelled ``other`` are not counted, which allows one region too many.
    The result says "an unseen player of this team is probably here", not which one. Not evaluated against
    ground truth (docs/evaluation.md section 11).
    """
    s = settings(cfg)
    key = s["time_key"]
    out = regions.reset_index(drop=True).copy()
    out["plausible_unseen"] = False
    if out.empty:
        return out
    obs = states[states["position_source"].eq("OBSERVED") & states["team_id"].isin(("home", "away"))]
    obs = obs.rename(columns={key: "time_key"})[["time_key", "team_id", "x_m", "y_m"]]
    n_obs = obs.groupby(["time_key", "team_id"]).size().rename("n_obs")
    pairs = out[["time_key", "team_id", "cx", "cy", "sxx", "syy"]].reset_index().merge(obs, on=["time_key", "team_id"])
    m2 = (pairs["x_m"] - pairs["cx"]) ** 2 / pairs["sxx"] + (pairs["y_m"] - pairs["cy"]) ** 2 / pairs["syy"]
    stale = set(pairs.loc[m2 <= chi2.ppf(s["exclude_observed_mass"], 2), "index"])
    out = out.join(n_obs, on=["time_key", "team_id"])
    slots = (s["team_size"] - out["n_obs"].fillna(0)).clip(lower=0)
    fresh = out[~out.index.isin(stale) & out["team_id"].isin(("home", "away"))]
    rank = fresh.groupby(["time_key", "team_id"])["area_m2"].rank(method="first")
    out.loc[rank.index, "plausible_unseen"] = (rank <= slots.loc[rank.index]).to_numpy()
    return out.drop(columns="n_obs")


def covers(regions: pd.DataFrame, truth_x: np.ndarray, truth_y: np.ndarray, mass: float) -> np.ndarray:
    """Whether the true position lies inside each region's ``mass`` ellipse."""
    m2 = (truth_x - regions["cx"].to_numpy()) ** 2 / regions["sxx"].to_numpy() \
        + (truth_y - regions["cy"].to_numpy()) ** 2 / regions["syy"].to_numpy()
    return m2 <= chi2.ppf(mass, 2)


def extend_states(states: pd.DataFrame, timeline: pd.DataFrame, horizon_s: float) -> pd.DataFrame:
    """Add unobserved rows up to ``horizon_s`` before the first and after the last observation of each (identity, shot).

    ``player_states`` stops at a short horizon (nothing is published beyond what the motion model can
    support); a probable region can honestly go on longer because it widens as certainty fades. The
    motion belief is a Kalman filter run forward (after the last observation) or on reversed time
    (before the first). Added rows have ``position_source`` ``PREDICTED`` and no published position.
    """
    obs = states[states["position_source"] == "OBSERVED"]
    new_rows = []
    for (ident, shot), g in obs.groupby(["identity_id", "shot_id"]):
        tl = timeline[timeline["shot_id"] == shot].sort_values("timestamp_s")
        t = tl["timestamp_s"].to_numpy(float)
        z = np.full((len(t), 2), np.nan)
        pos = g.set_index("timestamp_s")[["x_m", "y_m"]]
        idx = np.searchsorted(t, pos.index.to_numpy())
        ok = (idx < len(t)) & (np.abs(t[np.minimum(idx, len(t) - 1)] - pos.index.to_numpy()) < 1e-6)
        z[idx[ok]] = pos.to_numpy()[ok]
        seen = np.isfinite(z[:, 0])
        if seen.sum() < 2:
            continue
        have = set(states.loc[(states["identity_id"] == ident) & (states["shot_id"] == shot), "timestamp_s"].round(3))
        for sign in (1.0, -1.0):
            order = np.argsort(sign * t)
            mean_f, sd_f, _, _ = mm.kalman_rts(sign * t[order], z[order], seen[order])
            edge_t = t[seen].max() if sign > 0 else t[seen].min()
            for k, i in enumerate(order):
                dt = sign * (t[i] - edge_t)
                if dt <= 0 or dt > horizon_s or round(t[i], 3) in have or not np.isfinite(mean_f[k, 0]):
                    continue
                new_rows.append({"sample_id": int(tl["sample_id"].iloc[i]), "timestamp_s": float(t[i]), "identity_id": ident,
                                 "track_id": ident, "team_id": g["team_id"].iloc[0], "shot_id": shot, "x_m": np.nan,
                                 "y_m": np.nan, "x_est_m": mean_f[k, 0], "y_est_m": mean_f[k, 1], "sigma_m": sd_f[k],
                                 "position_source": "PREDICTED"})
    return pd.concat([states, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else states
