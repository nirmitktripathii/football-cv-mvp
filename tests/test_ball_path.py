"""Ball path selection (Viterbi) and bounded gap filling."""
import numpy as np
import pandas as pd
import pytest

from football_cv.tracking.ball_path import DETECTED, INTERPOLATED, PathParams, ball_track, fill_gaps, select_path

DT = 0.2


def _samples(n, segments=None):
    seg = segments if segments is not None else [0] * n
    return pd.DataFrame({"sample_id": range(n), "timestamp_s": np.arange(n) * DT, "segment_id": seg})


def _cands(rows):
    """rows: (sample_id, confidence, x, y); the segment id is 0."""
    df = pd.DataFrame(rows, columns=["sample_id", "confidence", "x", "y"])
    df["timestamp_s"] = df["sample_id"] * DT
    df["segment_id"] = 0
    return df[["sample_id", "timestamp_s", "segment_id", "confidence", "x", "y"]]


def _line(ids, conf=0.7, x0=100.0, vx=40.0):
    return [(i, conf, x0 + vx * i, 50.0) for i in ids]


def test_trajectory_consistent_candidate_beats_distant_confident_one():
    # Arrange: a ball moving 40 px per sample, plus a confident false alarm 300 px away at sample 3
    rows = _line([0, 1, 2, 4, 5], conf=0.5) + [(3, 0.9, 400.0, 400.0), (3, 0.3, 220.0, 50.0)]
    # Act
    path = select_path(_cands(rows))
    # Assert: the weaker candidate on the trajectory is chosen at sample 3
    chosen = path[path["sample_id"] == 3]
    assert len(chosen) == 1 and chosen.iloc[0]["x"] == pytest.approx(220.0)


def test_at_most_one_candidate_per_sample():
    path = select_path(_cands(_line(range(5)) + [(2, 0.6, 140.0, 55.0)]))
    assert path["sample_id"].is_unique


def test_isolated_detections_are_kept_not_discarded():
    # Arrange: three confident detections 2 s apart, too far in time to be linked to one another
    rows = [(0, 0.9, 100.0, 50.0), (10, 0.9, 900.0, 300.0), (20, 0.9, 300.0, 600.0)]
    # Act
    path = select_path(_cands(rows))
    # Assert: a break costs less than the evidence a detection carries, so none is dropped
    assert list(path["sample_id"]) == [0, 10, 20]


def test_original_settings_reproduce_the_discarding_behaviour():
    rows = [(0, 0.9, 100.0, 50.0), (10, 0.9, 900.0, 300.0), (20, 0.9, 300.0, 600.0)]
    assert len(select_path(_cands(rows), PathParams(detection_bonus=0.0))) == 1


def test_empty_candidates_give_empty_track():
    out = ball_track(_cands([]), _samples(5))
    assert out.empty


def test_link_beyond_max_speed_is_not_made():
    # 450 px/s * 0.2 s = 90 px is the limit; a 200 px jump in one sample cannot be one ball
    path = select_path(_cands([(0, 0.8, 100.0, 50.0), (1, 0.8, 300.0, 50.0)]))
    assert not path["link_prev"].any()


def test_straight_motion_gap_is_filled_accurately():
    # Arrange: samples 2 and 3 missing from a constant-velocity track
    path = select_path(_cands(_line([0, 1, 4, 5])))
    # Act
    out = fill_gaps(path, _samples(6))
    # Assert
    filled = out[out["status"] == INTERPOLATED].set_index("sample_id")
    assert list(filled.index) == [2, 3]
    assert filled["x"].to_numpy() == pytest.approx([180.0, 220.0], abs=2.0)
    assert (out[out["status"] == DETECTED]["sigma_px"] == 0.0).all()


@pytest.mark.parametrize("method", ["linear", "hold"])
def test_comparison_methods_fill_the_same_samples(method):
    path = select_path(_cands(_line([0, 1, 4, 5])))
    out = fill_gaps(path, _samples(6), PathParams(method=method))
    assert sorted(out.loc[out["status"] == INTERPOLATED, "sample_id"]) == [2, 3]


def test_gap_longer_than_limit_is_left_empty():
    path = select_path(_cands(_line([0, 1, 5, 6])), PathParams(max_link_s=1.0))
    out = fill_gaps(path, _samples(7), PathParams(max_fill_samples=2))
    assert (out["status"] == INTERPOLATED).sum() == 0


def test_no_fill_across_a_shot_cut():
    # Arrange: sample 2 is missing and the motion segment changes between samples 1 and 3
    cands = _cands(_line([0, 1, 3, 4]))
    cands["segment_id"] = [0, 0, 1, 1]
    # Act
    out = fill_gaps(select_path(cands), _samples(5, segments=[0, 0, 0, 1, 1]))
    # Assert
    assert (out["status"] == INTERPOLATED).sum() == 0


def test_no_extrapolation_before_first_or_after_last_detection():
    out = ball_track(_cands(_line([2, 3, 4])), _samples(8))
    assert set(out["sample_id"]) == {2, 3, 4}
