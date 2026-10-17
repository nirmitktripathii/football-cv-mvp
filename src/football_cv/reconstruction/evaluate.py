"""Masking evaluation of the gap-filling methods against the system's own observed trajectories.

Method: take a fully observed, consecutive stretch of one identity within one shot, delete a
window of ``L`` samples (plus one guard sample on each side, because ``x_s`` is a rolling median
that would otherwise leak the hidden values), run every method on what remains and score the
inner ``L`` samples against the hidden ``x_s`` values.

What this does and does not show: the "truth" is the pipeline's own calibrated projection, so the
numbers measure how well a motion model recovers a *tracked, projected* path, including its
calibration drift and jitter. They are not errors against real pitch positions. History-only
methods (``hold``, ``cv``, ``kalman``) use samples before the window only; ``linear`` and ``rts``
also use the samples after it, so they answer "fill a bracketed gap", the others "predict ahead".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football_cv.reconstruction import motion_models as mm

GUARD = 1               # samples removed from the input on each side of the scored window
MIN_CONTEXT = 5         # observed samples required on each side of a window (1 s at 5 fps)
CHI2_95_2D = 2.448      # radius in sigmas holding 95 % of an isotropic 2-D Gaussian
METHODS = ("hold", "linear", "cv", "kalman", "rts")


def _runs(good: pd.DataFrame):
    """Consecutive-sample observed runs per identity and shot: yields (identity, t, z)."""
    for (ident, _), g in good.groupby(["track_id", "shot_id"]):
        g = g.sort_values("sample_id")
        brk = (g["sample_id"].diff() != 1).cumsum()
        for _, r in g.groupby(brk):
            yield ident, r["timestamp_s"].to_numpy(), r[["x_s", "y_s"]].to_numpy()


def _windows(n: int, length: int):
    """Window starts with MIN_CONTEXT clean samples before and after (guards included), non-overlapping."""
    span = length + 2 * GUARD
    start = MIN_CONTEXT
    while start + span + MIN_CONTEXT <= n:
        yield start
        start += span + MIN_CONTEXT


def masking_eval(good: pd.DataFrame, gap_lengths_s=(0.6, 1.0, 2.0, 4.0, 8.0), sample_dt_s: float = 0.2) -> pd.DataFrame:
    """One row per (window, method, scored sample): error in metres and the method's sigma."""
    rows = []
    for gap_s in gap_lengths_s:
        length = int(round(gap_s / sample_dt_s))
        for ident, t, z in _runs(good):
            for w, start in enumerate(_windows(len(t), length)):
                lo, hi = start, start + length + 2 * GUARD
                obs = np.ones(len(t), bool)
                obs[lo:hi] = False
                z_in = z.copy()
                z_in[~obs] = np.nan
                inner = slice(lo + GUARD, hi - GUARD)
                for name, (mean, std) in mm.reconstruct_methods(t, z_in, obs).items():
                    err = np.linalg.norm(mean[inner] - z[inner], axis=1)
                    for k, (e, s) in enumerate(zip(err, std[inner])):
                        rows.append((gap_s, ident, start, w, name, k * sample_dt_s, e, s))
    return pd.DataFrame(rows, columns=["gap_s", "identity", "start", "window", "method", "offset_s", "err_m", "sigma_m"])


def _ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 300) -> tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    means = rng.choice(values, (n_boot, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(res: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per gap length and method: windows, MAE (+95 % bootstrap CI over windows), RMSE, median, p90, sigma coverage."""
    rng = np.random.default_rng(seed)
    out = []
    for (gap_s, name), g in res.groupby(["gap_s", "method"]):
        win = g.groupby(["identity", "start"])["err_m"].mean().to_numpy()
        ok = g["err_m"].notna() & g["sigma_m"].notna()
        lo, hi = _ci(win, rng)
        out.append({
            "gap_s": gap_s, "method": name, "windows": len(win), "mae_m": g["err_m"].mean(),
            "mae_ci_lo": lo, "mae_ci_hi": hi, "rmse_m": float(np.sqrt((g["err_m"] ** 2).mean())),
            "median_m": g["err_m"].median(), "p90_m": g["err_m"].quantile(0.9),
            "sigma_mean_m": g.loc[ok, "sigma_m"].mean(),
            "coverage_95": float((g.loc[ok, "err_m"] <= CHI2_95_2D * g.loc[ok, "sigma_m"]).mean()),
        })
    return pd.DataFrame(out).sort_values(["gap_s", "mae_m"]).reset_index(drop=True)
