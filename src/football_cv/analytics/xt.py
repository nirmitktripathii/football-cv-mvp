"""Expected Threat (xT) lookups for observed ball positions: EXPLORATORY, not action xT.

Genuine xT values *successful ball-moving actions of the team in possession*
(Singh 2019; ``socceraction.xthreat``). The CV pipeline sees ball positions in
about half of the wide-view frames, but not who possesses the ball, nor whether
a movement was a pass, a dribble or a loose ball. So this module reports:

* ``xt_home_attacking`` / ``xt_away_attacking``: the xT grid value of the ball's
  zone from each team's attacking perspective;
* ``dxt_home`` / ``dxt_away``: the change of that value between consecutive
  observations of one ball run (no gaps longer than ``MAX_GAP_S``).

These are labelled "exploratory ball-zone progression". They become genuine
xT only once a possession/action stage exists (see the roadmap).

Coordinate convention matches socceraction: the grid row for zone ``yj`` is
``w - 1 - yj`` and a team attacking towards x = 0 is mirrored with
``(L - x, W - y)``, exactly as SPADL's ``play_left_to_right`` does. In the
pitch frame used here, y = 0 is the near (camera-side) touchline, which is the
attacker's right-hand side for the team attacking towards +x, the same as
SPADL's y = 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from football_cv.trajectories.clean import usable

LENGTH_M, WIDTH_M = 105.0, 68.0
LABEL = "exploratory ball-zone progression (no possession or action detection)"


def load_grid(path: Path) -> np.ndarray:
    """xT grid as saved by ``ExpectedThreat.save_model``: ``w`` rows by ``l`` columns."""
    grid = np.asarray(json.loads(Path(path).read_text(encoding="utf-8")), dtype=float)
    if grid.ndim != 2 or not np.isfinite(grid).all():
        raise ValueError(f"{path}: not a finite 2-D xT grid")
    return grid


def cell_value(grid: np.ndarray, x, y, length: float = LENGTH_M, width: float = WIDTH_M) -> np.ndarray:
    """Grid value at pitch coordinates already expressed with the team attacking towards +x."""
    w, l = grid.shape
    xi = np.clip((np.asarray(x, dtype=float) / length * l).astype(int), 0, l - 1)
    yj = np.clip((np.asarray(y, dtype=float) / width * w).astype(int), 0, w - 1)
    return grid[w - 1 - yj, xi]


def attacking_frame(x, y, attacks_right: bool, length: float = LENGTH_M, width: float = WIDTH_M):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (x, y) if attacks_right else (length - x, width - y)


def ball_threat(traj: pd.DataFrame, grid: np.ndarray, home_attacks: str) -> pd.DataFrame:
    """Per observed, calibrated ball position: zone xT for both teams and its change within a run.

    ``home_attacks`` is ``"right"`` or ``"left"`` (towards x = 105 or x = 0) for the clip window.
    """
    if home_attacks not in ("right", "left"):
        raise ValueError("home_attacks must be 'right' or 'left'")
    ball = usable(traj)
    ball = ball[ball["track_id"] < 0].sort_values("timestamp_s")
    home_right = home_attacks == "right"
    out = ball[["sample_id", "timestamp_s", "segment_id", "run_id", "x_s", "y_s"]].copy()
    out["xt_home_attacking"] = cell_value(grid, *attacking_frame(out["x_s"], out["y_s"], home_right))
    out["xt_away_attacking"] = cell_value(grid, *attacking_frame(out["x_s"], out["y_s"], not home_right))
    grp = out.groupby(["segment_id", "run_id"])
    out["dxt_home"] = grp["xt_home_attacking"].diff()
    out["dxt_away"] = grp["xt_away_attacking"].diff()
    out["label"] = LABEL
    return out.reset_index(drop=True)


def window_summary(ball_xt: pd.DataFrame, start_s: float, end_s: float) -> dict:
    """Peak zone xT and summed positive progression per team inside one time window."""
    w = ball_xt[(ball_xt["timestamp_s"] >= start_s) & (ball_xt["timestamp_s"] <= end_s)]
    if w.empty:
        return {"ball_observations": 0, "label": LABEL}
    return {
        "ball_observations": int(len(w)),
        "peak_xt_home_attacking": round(float(w["xt_home_attacking"].max()), 4),
        "peak_xt_away_attacking": round(float(w["xt_away_attacking"].max()), 4),
        "positive_dxt_home": round(float(w["dxt_home"].clip(lower=0).sum()), 4),
        "positive_dxt_away": round(float(w["dxt_away"].clip(lower=0).sum()), 4),
        "label": LABEL,
    }
