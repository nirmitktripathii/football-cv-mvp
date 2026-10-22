"""mark_outlier_runs: a run of crops far from both kits inside a home/away track is a different person on that id
(the referee after an identity switch), labelled for those rows only."""
import numpy as np
import pandas as pd

from football_cv.tracking.teams import mark_outlier_runs

DARK, RED, YELLOW = (89.0, 133.0, 122.0), (115.0, 157.0, 146.0), (220.0, 128.0, 180.0)
CENTRES = {"home": DARK, "away": RED}
PARAMS = {"min_gap": 1.4, "min_run": 3}


def _track(tid, kits, team="home", role="player", detector=None):
    rows = [{"sample_id": i, "track_id": tid, "object_class": "person", "role": role, "team_id": team,
             "detector_role": (detector or {}).get(i, role)} for i in range(len(kits))]
    crops = [{"sample_id": i, "track_id": tid, "L": k[0], "a": k[1], "b": k[2]} for i, k in enumerate(kits) if k]
    return pd.DataFrame(rows), pd.DataFrame(crops)


def _run(kits, **kw):
    tracks, crops = _track(1, kits, **kw)
    return mark_outlier_runs(tracks, crops, CENTRES, PARAMS)


def test_referee_rows_inside_a_player_track_are_relabelled():
    out = _run([DARK] * 6 + [YELLOW] * 5 + [DARK] * 6)
    assert out["team_id"].tolist() == ["home"] * 6 + ["other"] * 5 + ["home"] * 6
    assert out["role"].tolist() == ["player"] * 6 + ["referee"] * 5 + ["player"] * 6


def test_a_run_shorter_than_min_run_changes_nothing():
    out = _run([DARK] * 6 + [YELLOW] * 2 + [DARK] * 6)
    assert (out["team_id"] == "home").all() and (out["role"] == "player").all()


def test_interrupted_far_crops_are_not_a_run():
    out = _run([DARK, YELLOW, YELLOW, DARK, YELLOW, YELLOW, DARK, DARK])
    assert (out["team_id"] == "home").all()


def test_samples_without_a_crop_inside_a_run_follow_the_run():
    out = _run([DARK] * 4 + [YELLOW, YELLOW, None, None, YELLOW] + [DARK] * 4)
    assert out["team_id"].tolist()[4:9] == ["other"] * 5
    assert out["team_id"].tolist()[:4] == ["home"] * 4 and out["team_id"].tolist()[9:] == ["home"] * 4


def test_threshold_is_in_units_of_the_kit_gap():
    near = (89.0, 133.0, 160.0)    # 38 from the home kit, under 1.4 x the 41 between the kits
    assert (_run([DARK] * 3 + [near] * 5)["team_id"] == "home").all()


def test_run_the_detector_called_goalkeeper_stays_a_goalkeeper():
    out = _run([DARK] * 3 + [YELLOW] * 5, detector={i: "goalkeeper" for i in range(3, 8)})
    assert out["team_id"].tolist()[3:] == ["unknown"] * 5 and out["role"].tolist()[3:] == ["goalkeeper"] * 5


def test_other_unknown_and_referee_tracks_are_left_alone():
    for team, role in (("other", "player"), ("unknown", "player"), ("other", "referee"), ("unknown", "goalkeeper")):
        out = _run([YELLOW] * 8, team=team, role=role)
        assert (out["team_id"] == team).all() and (out["role"] == role).all()


def test_input_frame_is_not_modified():
    tracks, crops = _track(1, [DARK] * 3 + [YELLOW] * 5)
    mark_outlier_runs(tracks, crops, CENTRES, PARAMS)
    assert (tracks["team_id"] == "home").all()
