"""drop_implausible_goalkeepers: a short goalkeeper track far smaller than the players is a pitch mark or board
mis-detected as a person; real keepers are full-size and long-lived."""
import pandas as pd

from football_cv.tracking.teams import drop_implausible_goalkeepers

PARAMS = {"min_rel_height": 0.7, "max_rows": 40}


def _track(tid, rows, height, role):
    return [{"sample_id": i, "track_id": tid, "object_class": "person", "role": role,
             "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 10.0, "bbox_y2": float(height)} for i in range(rows)]


def _frame(*tracks):
    return pd.DataFrame([r for t in tracks for r in t])


def test_tiny_short_goalkeeper_track_is_dropped():
    df = _frame(_track(1, 100, 34, "player"), _track(2, 10, 14, "goalkeeper"))
    assert set(drop_implausible_goalkeepers(df, PARAMS)["track_id"]) == {1}


def test_full_size_goalkeeper_is_kept():
    df = _frame(_track(1, 100, 34, "player"), _track(2, 10, 30, "goalkeeper"))
    assert set(drop_implausible_goalkeepers(df, PARAMS)["track_id"]) == {1, 2}


def test_small_but_long_goalkeeper_track_is_kept():
    df = _frame(_track(1, 100, 34, "player"), _track(2, 80, 14, "goalkeeper"))
    assert set(drop_implausible_goalkeepers(df, PARAMS)["track_id"]) == {1, 2}


def test_small_players_are_not_touched():
    df = _frame(_track(1, 100, 34, "player"), _track(2, 5, 12, "player"), _track(3, 5, 12, "referee"))
    assert set(drop_implausible_goalkeepers(df, PARAMS)["track_id"]) == {1, 2, 3}


def test_ball_rows_and_input_are_untouched():
    df = _frame(_track(1, 100, 34, "player"), _track(2, 10, 14, "goalkeeper"))
    before = df.copy()
    drop_implausible_goalkeepers(df, PARAMS)
    pd.testing.assert_frame_equal(df, before)


def test_no_players_means_no_reference_and_nothing_is_dropped():
    df = _frame(_track(2, 10, 14, "goalkeeper"))
    assert len(drop_implausible_goalkeepers(df, PARAMS)) == 10
