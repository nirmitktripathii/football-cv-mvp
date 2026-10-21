import pandas as pd
import pytest

from football_cv.calibration.offpitch import drop_off_pitch_people, reject_unplayable_balls

L, W = 105.0, 68.0


def _rows(track_id, xs, ys, role="player", cls="person"):
    return pd.DataFrame({
        "sample_id": range(len(xs)), "timestamp_s": [i * 0.2 for i in range(len(xs))], "track_id": track_id,
        "object_class": cls, "role": role, "x_m": xs, "y_m": ys,
    })


def test_a_person_standing_outside_the_touchline_is_dropped():
    proj = pd.concat([_rows(1, [50.0] * 6, [10.0] * 6), _rows(2, [50.0] * 6, [-4.0] * 6)], ignore_index=True)
    out, dropped = drop_off_pitch_people(proj, L, W, outside_m=2.0, min_rows=3)
    assert dropped == [2]
    assert set(out["track_id"]) == {1}


def test_a_player_who_steps_out_briefly_is_kept():
    ys = [10.0, 10.0, -3.0, -3.5, 10.0, 10.0, 10.0]
    out, dropped = drop_off_pitch_people(_rows(1, [50.0] * 7, ys), L, W, outside_m=2.0, min_rows=3)
    assert dropped == [] and len(out) == 7


def test_referees_and_goalkeepers_are_exempt_and_short_tracks_are_kept():
    proj = pd.concat([
        _rows(1, [50.0] * 5, [-4.0] * 5, role="referee"),
        _rows(2, [108.0] * 5, [30.0] * 5, role="goalkeeper"),
        _rows(3, [50.0] * 2, [-4.0] * 2),
    ], ignore_index=True)
    out, dropped = drop_off_pitch_people(proj, L, W, outside_m=2.0, min_rows=3)
    assert dropped == [] and len(out) == 12


def test_rows_without_a_pitch_position_never_count():
    proj = _rows(1, [None] * 6, [None] * 6)
    out, dropped = drop_off_pitch_people(proj, L, W, outside_m=2.0, min_rows=3)
    assert dropped == [] and len(out) == 6


def test_a_spare_ball_far_outside_the_pitch_is_dropped():
    play = _rows(-1, [50.0, 50.5, 51.0], [30.0, 30.0, 30.0], cls="ball")
    spare = _rows(-1, [60.0] * 4, [-5.0] * 4, cls="ball").assign(sample_id=range(10, 14), timestamp_s=[10.0 + i * 0.2 for i in range(4)])
    out = reject_unplayable_balls(pd.concat([play, spare], ignore_index=True), L, W)
    assert len(out) == 3 and out["y_m"].min() > 0


def test_a_ball_kicked_out_of_play_stays_linked_to_the_play():
    rows = _rows(-1, [100.0, 104.0, 108.0, 112.0], [30.0] * 4, cls="ball")  # crosses the goal line in 0.6 s
    assert len(reject_unplayable_balls(rows, L, W)) == 4


def test_a_static_detection_on_a_penalty_spot_is_dropped_but_a_ball_passing_over_it_is_kept():
    static = _rows(-1, [11.0] * 5, [34.0] * 5, cls="ball")
    passing = _rows(-1, [5.0, 8.0, 11.0, 14.0, 17.0], [34.0] * 5, cls="ball")
    assert reject_unplayable_balls(static, L, W).empty
    assert len(reject_unplayable_balls(passing, L, W)) == 5


def test_non_ball_rows_pass_through_the_ball_filter():
    mix = pd.concat([_rows(1, [50.0] * 3, [-6.0] * 3), _rows(-1, [60.0] * 3, [-6.0] * 3, cls="ball")], ignore_index=True)
    out = reject_unplayable_balls(mix, L, W)
    assert (out["object_class"] == "person").all() and len(out) == 3


# --- corner flags (docs/corner_flags.md) ---------------------------------------------------------------------------

def test_a_still_track_on_a_corner_is_a_flag_and_is_dropped():
    from football_cv.calibration.offpitch import drop_corner_flags
    flag = _rows(7, [0.3, 0.35, 0.3, 0.32], [0.2, 0.25, 0.2, 0.22])
    far_corner = _rows(8, [L - 0.2] * 3, [W - 0.1] * 3)
    mid = _rows(9, [50.0] * 4, [10.0] * 4)
    out, dropped = drop_corner_flags(pd.concat([flag, far_corner, mid], ignore_index=True), L, W)
    assert dropped == [7, 8]
    assert set(out["track_id"]) == {9}


def test_a_mover_near_a_corner_and_a_still_track_away_from_it_are_kept():
    from football_cv.calibration.offpitch import drop_corner_flags
    taker = _rows(1, [3.0, 2.0, 1.0, 1.5], [4.0, 3.0, 1.0, 2.5])   # runs up: spread above still_m
    keeper_area = _rows(2, [8.0] * 4, [30.0] * 4)                  # still, nowhere near a corner
    out, dropped = drop_corner_flags(pd.concat([taker, keeper_area], ignore_index=True), L, W)
    assert dropped == [] and len(out) == 8


def test_referees_goalkeepers_short_tracks_and_non_person_rows_are_kept_by_the_flag_rule():
    from football_cv.calibration.offpitch import drop_corner_flags
    proj = pd.concat([
        _rows(1, [0.3] * 4, [0.3] * 4, role="referee"), _rows(2, [0.3] * 4, [0.3] * 4, role="goalkeeper"),
        _rows(3, [0.3], [0.3]), _rows(4, [0.3] * 4, [0.3] * 4, role="player", cls="ball")], ignore_index=True)
    out, dropped = drop_corner_flags(proj, L, W)
    assert dropped == [] and len(out) == len(proj)


def test_rows_without_a_pitch_position_never_make_a_flag():
    from football_cv.calibration.offpitch import drop_corner_flags
    rows = _rows(1, [None] * 4, [None] * 4)
    out, dropped = drop_corner_flags(rows, L, W)
    assert dropped == [] and len(out) == 4
