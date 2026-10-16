"""Analytics on synthetic positions: xT lookups match socceraction, shape geometry, candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_cv.analytics import spatial, xt
from football_cv.sequences import candidates


def _traj(rows: list[tuple]) -> pd.DataFrame:
    """rows: (sample_id, timestamp_s, track_id, team_id, x, y), one segment, one run per track."""
    df = pd.DataFrame(rows, columns=["sample_id", "timestamp_s", "track_id", "team_id", "x_s", "y_s"])
    return df.assign(segment_id=0, run_id=0, jump=False, short_track=False)


@pytest.fixture
def grid() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((12, 16))


def test_cell_value_matches_socceraction_rate(grid):
    from socceraction.spadl import config as spadlcfg
    from socceraction.xthreat import ExpectedThreat

    model = ExpectedThreat(l=16, w=12)
    model.xT = grid
    rng = np.random.default_rng(1)
    n = 50
    actions = pd.DataFrame({
        "start_x": rng.uniform(0, 105, n), "start_y": rng.uniform(0, 68, n),
        "end_x": rng.uniform(0, 105, n), "end_y": rng.uniform(0, 68, n),
        "type_id": spadlcfg.actiontypes.index("pass"), "result_id": spadlcfg.results.index("success"),
    })
    expected = model.rate(actions)
    ours = xt.cell_value(grid, actions["end_x"], actions["end_y"]) - xt.cell_value(grid, actions["start_x"], actions["start_y"])
    np.testing.assert_allclose(ours, expected)


def test_attacking_frame_mirrors_both_axes():
    x, y = xt.attacking_frame([10.0, 100.0], [5.0, 60.0], attacks_right=False)
    np.testing.assert_allclose(x, [95.0, 5.0])
    np.testing.assert_allclose(y, [63.0, 8.0])


def test_cell_value_clips_out_of_pitch_points(grid):
    assert xt.cell_value(grid, 110.0, -2.0) == grid[11, 15]  # last column, row of y = 0


def test_ball_threat_is_symmetric_between_teams(grid):
    traj = _traj([(i, i * 0.2, -1, "ball", 20.0 + 10 * i, 30.0) for i in range(5)])
    out = xt.ball_threat(traj, grid, home_attacks="right")
    mirrored = xt.cell_value(grid, *xt.attacking_frame(out["x_s"], out["y_s"], attacks_right=False))
    np.testing.assert_allclose(out["xt_away_attacking"], mirrored)
    assert out["dxt_home"].isna().sum() == 1  # first observation of the run has no previous value
    with pytest.raises(ValueError):
        xt.ball_threat(traj, grid, home_attacks="up")


def test_team_shape_square_formation():
    pts = [(0, 0.0, i, "home", x, y) for i, (x, y) in enumerate([(10, 10), (30, 10), (10, 30), (30, 30)])]
    shape = spatial.team_shape(_traj(pts))
    row = shape.iloc[0]
    assert (row["width_m"], row["depth_m"], row["hull_area_m2"]) == (20.0, 20.0, 400.0)
    assert row["mean_nn_dist_m"] == 20.0
    assert (row["centroid_x_m"], row["centroid_y_m"]) == (20.0, 20.0)


def test_team_shape_skips_teams_with_too_few_players():
    pts = [(0, 0.0, i, "away", 10.0 * i, 5.0) for i in range(3)]
    assert spatial.team_shape(_traj(pts)).empty


def test_ball_proximity_names_nearest_player():
    traj = _traj([(0, 0.0, 1, "home", 50.0, 30.0), (0, 0.0, 2, "away", 60.0, 30.0), (0, 0.0, -1, "ball", 58.0, 31.0)])
    prox = spatial.ball_proximity(traj)
    assert prox.iloc[0]["nearest_track_id"] == 2
    assert prox.iloc[0]["nearest_team"] == "away"


def test_candidates_find_final_third_window(grid):
    ball = [(i, i * 0.2, -1, "ball", 60.0 + i, 34.0) for i in range(30)]  # x 60 -> 89 over 5.8 s
    near = [(i, i * 0.2, 7, "home", 60.0 + i, 35.0) for i in range(30)]
    traj = _traj(ball + near)
    ball_xt = xt.ball_threat(traj, grid, home_attacks="right")
    prox = spatial.ball_proximity(traj)
    calib = pd.DataFrame({"sample_id": range(30), "timestamp_s": [i * 0.2 for i in range(30)], "keyframe_dt_s": 1.0})
    cand = candidates.find_candidates(ball_xt, prox, calib, home_attacks="right")
    assert len(cand) == 1
    c = cand.iloc[0]
    assert (c["attacking_direction"], c["probable_attacking_team"]) == ("right", "home")
    assert c["nearest_player_share_attacking_team"] == 1.0
    assert c["confidence"].startswith("medium")


def test_candidates_ignore_midfield_windows(grid):
    traj = _traj([(i, i * 0.2, -1, "ball", 50.0, 34.0) for i in range(30)])
    ball_xt = xt.ball_threat(traj, grid, home_attacks="left")
    calib = pd.DataFrame({"sample_id": range(30), "timestamp_s": [i * 0.2 for i in range(30)], "keyframe_dt_s": 0.0})
    assert candidates.find_candidates(ball_xt, spatial.ball_proximity(traj), calib, "left").empty
