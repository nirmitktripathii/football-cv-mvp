"""Motion models that fill gaps in one identity's pitch trajectory (metres, real timestamps).

Every function takes the identity's *whole* sample timeline ``t`` (seconds, strictly increasing,
not assumed evenly spaced), the positions ``z`` (n, 2) with NaN where the player was not
observed, and the boolean mask ``observed``. Each returns a position mean (n, 2) and a per-axis
standard deviation (n,) in metres. Nothing here knows about camera views or player states;
that is ``reconstruct.py``.

Methods, simplest first:

* ``hold``      last observation carried forward (std grows with the speed cap).
* ``linear``    linear interpolation between the bounding observations (needs both sides).
* ``cv``        constant-velocity extrapolation from the last observations (history only).
* ``kalman``    forward constant-velocity Kalman filter, history only (std from the filter).
* ``rts``       Kalman filter + Rauch-Tung-Striebel smoother, uses both sides of a gap.

A constant-velocity model is the honest baseline: nothing in this module knows football
context (ball, teammates, role). Context-conditioned prediction is future work (see
``docs/future_pipeline_roadmap.md``).
"""
from __future__ import annotations

import numpy as np

SIGMA_ACCEL = 2.5     # m/s^2, process noise: players change velocity within seconds
SIGMA_MEAS = 1.0      # m, measurement noise: projection jitter of a foot point
SIGMA_VEL0 = 4.0      # m/s, prior velocity spread at the first observation
MAX_SPEED = 9.0       # m/s, speed cap for hold/cv std growth and velocity estimates
CV_FIT_POINTS = 3     # observations used to estimate velocity for ``cv``


def _filter_axis(t, z, observed, sigma_a, sigma_m):
    """Forward Kalman filter for one axis; returns (x_pred, P_pred, x_filt, P_filt, F list)."""
    n = len(t)
    xp, Pp = np.zeros((n, 2)), np.zeros((n, 2, 2))
    xf, Pf = np.zeros((n, 2)), np.zeros((n, 2, 2))
    Fs = [np.eye(2)] * n
    started = False
    for i in range(n):
        if not started:
            if not observed[i]:
                xp[i] = xf[i] = np.nan
                Pp[i] = Pf[i] = np.nan
                continue
            xf[i] = [z[i], 0.0]
            Pf[i] = np.diag([sigma_m**2, SIGMA_VEL0**2])
            xp[i], Pp[i] = xf[i], Pf[i]
            started = True
            continue
        dt = t[i] - t[i - 1]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = sigma_a**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        Fs[i] = F
        xp[i] = F @ xf[i - 1]
        Pp[i] = F @ Pf[i - 1] @ F.T + Q
        if observed[i]:
            S = Pp[i][0, 0] + sigma_m**2
            K = Pp[i][:, 0] / S
            xf[i] = xp[i] + K * (z[i] - xp[i][0])
            Pf[i] = (np.eye(2) - np.outer(K, [1.0, 0.0])) @ Pp[i]
        else:
            xf[i], Pf[i] = xp[i], Pp[i]
    return xp, Pp, xf, Pf, Fs


def _smooth_axis(xp, Pp, xf, Pf, Fs):
    n = len(xf)
    xs, Ps = xf.copy(), Pf.copy()
    for i in range(n - 2, -1, -1):
        if np.isnan(xf[i, 0]) or np.isnan(xs[i + 1, 0]):
            continue
        C = Pf[i] @ Fs[i + 1].T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    return xs, Ps


def kalman_rts(t, z, observed, sigma_a=SIGMA_ACCEL, sigma_m=SIGMA_MEAS):
    """(filtered mean, filtered std, smoothed mean, smoothed std); NaN before the first observation."""
    t = np.asarray(t, float)
    means_f, means_s, var_f, var_s = [], [], [], []
    for ax in (0, 1):
        xp, Pp, xf, Pf, Fs = _filter_axis(t, z[:, ax], observed, sigma_a, sigma_m)
        xs, Ps = _smooth_axis(xp, Pp, xf, Pf, Fs)
        means_f.append(xf[:, 0]); means_s.append(xs[:, 0])
        var_f.append(Pf[:, 0, 0]); var_s.append(Ps[:, 0, 0])
    sd = lambda v: np.sqrt(np.maximum(np.mean(v, axis=0), 0.0))  # noqa: E731
    return np.stack(means_f, 1), sd(var_f), np.stack(means_s, 1), sd(var_s)


def hold(t, z, observed):
    t = np.asarray(t, float)
    mean = np.full_like(z, np.nan)
    std = np.full(len(t), np.nan)
    last = None
    for i in range(len(t)):
        if observed[i]:
            last = i
        if last is not None:
            mean[i] = z[last]
            std[i] = SIGMA_MEAS + MAX_SPEED * (t[i] - t[last]) / 2
    return mean, std


def linear(t, z, observed):
    """Linear interpolation; NaN outside the observed span (no extrapolation)."""
    t = np.asarray(t, float)
    idx = np.flatnonzero(observed)
    mean = np.full_like(z, np.nan)
    std = np.full(len(t), np.nan)
    if len(idx) < 2:
        return mean, std
    inside = (t >= t[idx[0]]) & (t <= t[idx[-1]])
    for ax in (0, 1):
        mean[inside, ax] = np.interp(t[inside], t[idx], z[idx, ax])
    # uncertainty: grows with the distance to the nearest observation on either side
    left = np.searchsorted(t[idx], t, side="right") - 1
    right = np.minimum(left + 1, len(idx) - 1)
    d = np.minimum(np.abs(t - t[idx][np.clip(left, 0, None)]), np.abs(t[idx][right] - t))
    std[inside] = SIGMA_MEAS + MAX_SPEED * d[inside] / 2
    return mean, std


def cv(t, z, observed, fit_points=CV_FIT_POINTS):
    """Constant velocity from the last ``fit_points`` observations (history only)."""
    t = np.asarray(t, float)
    mean = np.full_like(z, np.nan)
    std = np.full(len(t), np.nan)
    hist: list[int] = []
    for i in range(len(t)):
        if observed[i]:
            hist.append(i)
            mean[i], std[i] = z[i], SIGMA_MEAS
            continue
        if not hist:
            continue
        use = hist[-fit_points:]
        if len(use) >= 2:
            v = np.array([np.polyfit(t[use], z[use, ax], 1)[0] for ax in (0, 1)])
            speed = float(np.hypot(*v))
            if speed > MAX_SPEED:
                v *= MAX_SPEED / speed
        else:
            v = np.zeros(2)
        dt = t[i] - t[hist[-1]]
        mean[i] = z[hist[-1]] + v * dt
        std[i] = SIGMA_MEAS + SIGMA_ACCEL * dt**2 / 2
    return mean, std


def reconstruct_methods(t, z, observed):
    """All methods on one timeline: ``{name: (mean (n,2), std (n,))}``."""
    t = np.asarray(t, float)
    z = np.asarray(z, float)
    observed = np.asarray(observed, bool)
    mf, sf, ms, ss = kalman_rts(t, z, observed)
    return {
        "hold": hold(t, z, observed), "linear": linear(t, z, observed), "cv": cv(t, z, observed),
        "kalman": (mf, sf), "rts": (ms, ss),
    }
