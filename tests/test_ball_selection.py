"""Ball selection: persistent ball detections (spare balls, penalty spot, line marks) are dropped before the per-frame pick."""
import numpy as np
import pandas as pd

from football_cv.tracking.tracker import select_balls, static_ball_mask

PARAMS = {"static_radius_px": 6.0, "static_window_s": 2.0, "static_min_frames": 3, "min_confidence": 0.0}


def _balls(rows):
    """rows: (sample_id, t, segment, conf, cx, cy) -> detections with an 8 px box around the centre, and the centres."""
    df = pd.DataFrame(rows, columns=["sample_id", "timestamp_s", "segment_id", "confidence", "cx", "cy"])
    xy = df[["cx", "cy"]].to_numpy(dtype=float)
    for k, sign in (("bbox_x1", -4), ("bbox_x2", 4)):
        df[k] = df["cx"] + sign
    for k, sign in (("bbox_y1", -4), ("bbox_y2", 4)):
        df[k] = df["cy"] + sign
    return df.drop(columns=["cx", "cy"]), xy


def test_detection_repeated_in_place_is_static():
    # Arrange: a spare ball seen in five consecutive 0.2 s samples at the same stabilised spot
    df, xy = _balls([(i, i * 0.2, 0, 0.8, 500.0, 300.0) for i in range(5)])
    # Act
    mask = static_ball_mask(df, xy, PARAMS)
    # Assert
    assert mask.all()


def test_moving_ball_is_not_static():
    df, xy = _balls([(i, i * 0.2, 0, 0.8, 500.0 + 40 * i, 300.0) for i in range(5)])
    assert not static_ball_mask(df, xy, PARAMS).any()


def test_short_repeat_below_min_frames_is_kept():
    # each detection has only two neighbours at its spot, one short of static_min_frames=3
    df, xy = _balls([(i, i * 0.2, 0, 0.8, 500.0, 300.0) for i in range(3)])
    assert not static_ball_mask(df, xy, PARAMS).any()


def test_repeats_outside_the_time_window_do_not_count():
    # the same spot seen again only 5 s later is not persistence within 2 s
    rows = [(0, 0.0, 0, 0.8, 500.0, 300.0)] + [(i, 5.0 + i * 0.2, 0, 0.8, 500.0, 300.0) for i in range(1, 3)]
    df, xy = _balls(rows)
    assert not static_ball_mask(df, xy, PARAMS).any()


def test_repeats_in_another_segment_do_not_count():
    # stabilised coordinates of different camera segments are not comparable
    rows = [(i, i * 0.2, i % 2, 0.8, 500.0, 300.0) for i in range(6)]
    df, xy = _balls(rows)
    assert not static_ball_mask(df, xy, PARAMS).any()


def test_two_detections_in_one_frame_count_once():
    # one frame with two stacked detections is not persistence
    rows = [(0, 0.0, 0, 0.8, 500.0, 300.0), (0, 0.0, 0, 0.7, 501.0, 300.0), (1, 0.2, 0, 0.8, 500.0, 300.0)]
    df, xy = _balls(rows)
    assert not static_ball_mask(df, xy, PARAMS).any()


def test_select_balls_prefers_the_moving_ball_over_a_stronger_static_one():
    # a confident spare ball (0.9) sits still while the real ball (0.4) moves; the static one must not win each frame
    rows = []
    for i in range(6):
        rows.append((i, i * 0.2, 0, 0.9, 900.0, 600.0))
        rows.append((i, i * 0.2, 0, 0.4, 100.0 + 50 * i, 200.0))
    df, xy = _balls(rows)
    picked = select_balls(df, xy, PARAMS)
    assert len(picked) == 6
    assert (picked["confidence"] == 0.4).all()


def test_select_balls_leaves_frames_without_a_ball_empty():
    df, xy = _balls([(0, 0.0, 0, 0.8, 100.0, 200.0), (4, 0.8, 0, 0.8, 300.0, 200.0)])
    assert sorted(select_balls(df, xy, PARAMS)["sample_id"]) == [0, 4]


def test_min_confidence_floor_is_applied():
    df, xy = _balls([(0, 0.0, 0, 0.15, 100.0, 200.0), (1, 0.2, 0, 0.6, 140.0, 200.0)])
    picked = select_balls(df, xy, {**PARAMS, "min_confidence": 0.3})
    assert list(picked["sample_id"]) == [1]
