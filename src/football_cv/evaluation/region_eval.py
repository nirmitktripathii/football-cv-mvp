"""Evaluate inferred regions against SkillCorner detected positions by hiding stretches of them.

A trial takes a player and a stretch of ``gap_s`` seconds that lies inside one detected run, hides the
player's positions in the stretch, builds the same ``player_states``-like table the pipeline would have
(observed team-mates, motion estimate for the hidden player) and asks ``regions.infer_regions`` for a
region at every hidden sample. The truth is the hidden detected position. Reported per gap length:
coverage of the ``region_mass`` ellipse (calibrated if it equals the mass), median area, median error of
the centre.

Degradation mimics the video pipeline: per-frame noise and a slowly drifting offset of the whole
pitch mapping (an Ornstein-Uhlenbeck process, common to all players, with ``CORRELATION_S`` memory).

Limits: truth only exists where SkillCorner detected the player, and those stretches are biased towards
play the camera follows; real off-screen motion may differ. Pre-registered criterion (fixed before the
first run): coverage within [0.85, 0.95] at every gap length, and for gaps <= 4 s a median area less
than half that of the shape-only baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football_cv.evaluation import skillcorner as sc
from football_cv.reconstruction import motion_models as mm
from football_cv.reconstruction import regions

SAMPLE_FPS = 5.0
CONTEXT_S = 20.0            # observed context kept either side of a hidden stretch
CORRELATION_S = 15.0        # memory of the drifting pitch-mapping offset
COVERAGE_BAND = (0.85, 0.95)
GAP_LENGTHS_S = (1.0, 2.0, 4.0, 8.0, 15.0)


def drifting_offset(t: np.ndarray, sigma_m: float, rng: np.random.Generator) -> np.ndarray:
    """(n, 2) offset of the pitch mapping over time: stationary OU with std ``sigma_m``."""
    out = np.zeros((len(t), 2))
    if sigma_m <= 0 or len(t) == 0:
        return out
    out[0] = rng.normal(0.0, sigma_m, 2)
    for i in range(1, len(t)):
        rho = np.exp(-(t[i] - t[i - 1]) / CORRELATION_S)
        out[i] = rho * out[i - 1] + np.sqrt(1 - rho**2) * rng.normal(0.0, sigma_m, 2)
    return out


def hide_trial(table: pd.DataFrame, run: pd.Series, gap_s: float, cfg: dict, rng: np.random.Generator,
               offset_sigma_m: float, noise_sigma_m: float = 1.0) -> pd.DataFrame | None:
    """One hidden stretch; one row per hidden sample with the region, the truth and which component set was used."""
    step = int(sc.FPS / SAMPLE_FPS)
    lead = int(CONTEXT_S * sc.FPS)
    start = int(rng.integers(run.f0 + int(1.0 * sc.FPS), run.f1 - int((gap_s + 1.0) * sc.FPS) + 1))
    hide0, hide1 = start, start + int(gap_s * sc.FPS)
    win = table[(table.frame >= hide0 - lead) & (table.frame <= hide1 + lead) & (table.frame % step == 0)].copy()
    mates = win[(win.team == run.team) & win.detected]
    me = win[win.player_id == run.player_id]
    hidden = me[(me.frame >= hide0) & (me.frame <= hide1) & me.detected]
    if len(hidden) < 2:
        return None

    t_all = np.sort(win.t_s.unique())
    off = pd.DataFrame(drifting_offset(t_all, offset_sigma_m, rng), index=t_all, columns=["ox", "oy"])

    def degrade(df: pd.DataFrame) -> pd.DataFrame:
        o = off.loc[df.t_s.to_numpy()].to_numpy()
        noise = rng.normal(0.0, noise_sigma_m, (len(df), 2))
        return df.assign(x_m=df.x.to_numpy() + o[:, 0] + noise[:, 0], y_m=df.y.to_numpy() + o[:, 1] + noise[:, 1])

    mates = degrade(mates[mates.player_id != run.player_id])
    mine = degrade(me[me.detected & ~me.frame.between(hide0, hide1)])
    t_me = np.sort(me.t_s.unique())
    z = np.full((len(t_me), 2), np.nan)
    idx = np.searchsorted(t_me, mine.t_s.to_numpy())
    z[idx] = mine[["x_m", "y_m"]].to_numpy()
    observed = np.isfinite(z[:, 0])
    if observed.sum() < 4:
        return None
    _, _, mean_s, sd_s = mm.kalman_rts(t_me, z, observed)

    mine_rows = pd.DataFrame({
        "t_s": t_me, "identity_id": run.player_id, "team_id": run.team,
        "x_m": z[:, 0], "y_m": z[:, 1], "x_est_m": mean_s[:, 0], "y_est_m": mean_s[:, 1], "sigma_m": sd_s,
        "position_source": np.where(observed, "OBSERVED", "INTERPOLATED"),
    })
    mate_rows = pd.DataFrame({
        "t_s": mates.t_s.to_numpy(), "identity_id": mates.player_id.to_numpy(), "team_id": mates.team.to_numpy(),
        "x_m": mates.x_m.to_numpy(), "y_m": mates.y_m.to_numpy(), "x_est_m": np.nan, "y_est_m": np.nan,
        "sigma_m": np.nan, "position_source": "OBSERVED",
    })
    states = pd.concat([mine_rows, mate_rows], ignore_index=True)
    reg = regions.infer_regions(states, {"regions": {**(cfg.get("regions") or {}), "time_key": "t_s"}})
    reg = reg[reg.identity_id == run.player_id].merge(hidden[["t_s", "x", "y"]], left_on="time_key", right_on="t_s")
    return reg if len(reg) else None


def run_gap_trials(table: pd.DataFrame, cfg: dict, gap_s: float, n_trials: int, seed: int, offset_sigma_m: float,
                   period: int | None = None, min_run_s: float | None = None) -> dict:
    """Aggregate over ``n_trials`` random hidden stretches of length ``gap_s``."""
    rng = np.random.default_rng(seed)
    if period is not None:
        table = table[table.period == period]
    runs = sc.detected_runs(table, min_run_s or gap_s + 2.0)
    if runs.empty:
        return {"trials": 0}
    mass = regions.settings(cfg)["region_mass"]
    rows = []
    for _, run in runs.sample(n_trials, replace=True, random_state=int(rng.integers(1 << 31))).iterrows():
        res = hide_trial(table, run, gap_s, cfg, rng, offset_sigma_m)
        if res is None:
            continue
        res["covered"] = regions.covers(res, res["x"].to_numpy(), res["y"].to_numpy(), mass)
        res["centre_err_m"] = np.hypot(res.cx - res.x, res.cy - res.y)
        rows.append(res)
    if not rows:
        return {"trials": 0}
    r = pd.concat(rows, ignore_index=True)
    return {
        "trials": len(rows), "samples": len(r), "coverage": float(r.covered.mean()),
        "median_area_m2": float(r.area_m2.median()), "median_centre_err_m": float(r.centre_err_m.median()),
        "components": {k: int(v) for k, v in r.components.value_counts().items()},
    }
