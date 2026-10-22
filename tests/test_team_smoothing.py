"""Per-sample team labels: shirt colour over time with a cost for switching, so one bad crop cannot flip a player but
an identity switch between opponents does, and the track is then cut where the shirt changes."""
import numpy as np
import pandas as pd

from football_cv.tracking.teams import smooth_teams, split_team_switches, team_switch_path

DARK, RED = (89.0, 133.0, 122.0), (115.0, 155.0, 146.0)
CENTRES = {"home": DARK, "away": RED}
PARAMS = {"llr_scale": 20.0, "max_llr": 3.0, "switch_cost": 8.0, "min_evidence": 1.5}


def _track(tid, kits, team, role="player"):
    rows = [{"sample_id": i, "track_id": tid, "object_class": "person", "role": role, "team_id": team}
            for i in range(len(kits))]
    crops = [{"sample_id": i, "track_id": tid, "L": k[0], "a": k[1], "b": k[2]}
             for i, k in enumerate(kits) if k is not None]
    return rows, crops


def _clip(*tracks):
    rows, crops = [], []
    for r, c in tracks:
        rows += r
        crops += c
    return pd.DataFrame(rows), pd.DataFrame(crops)


def test_path_ignores_one_contrary_sample():
    assert (team_switch_path(np.array([2.0, 2.0, -2.0, 2.0, 2.0]), switch_cost=8.0) == 0).all()


def test_path_follows_a_sustained_change():
    path = team_switch_path(np.array([2.0] * 6 + [-2.0] * 6), switch_cost=8.0)
    assert path.tolist() == [0] * 6 + [1] * 6


def test_rows_follow_the_shirt_after_an_identity_switch():
    tracks, crops = _clip(_track(1, [DARK] * 10 + [RED] * 10, "home"))
    out = smooth_teams(tracks, crops, CENTRES, PARAMS)
    assert out["team_id"].tolist() == ["home"] * 10 + ["away"] * 10


def test_a_single_bad_crop_does_not_flip_a_player():
    tracks, crops = _clip(_track(1, [DARK] * 6 + [RED] + [DARK] * 6, "home"))
    assert (smooth_teams(tracks, crops, CENTRES, PARAMS)["team_id"] == "home").all()


def test_samples_without_a_crop_take_the_neighbouring_label():
    tracks, crops = _clip(_track(1, [DARK] * 5 + [None] * 3 + [DARK] * 5, "home"))
    assert (smooth_teams(tracks, crops, CENTRES, PARAMS)["team_id"] == "home").all()


def test_referees_goalkeepers_and_weak_evidence_keep_their_label():
    tracks, crops = _clip(
        _track(1, [RED] * 8, "other", role="referee"),
        _track(2, [DARK] * 8, "unknown", role="goalkeeper"),
        _track(3, [None] * 7 + [(103.0, 145.0, 135.0)], "home"),  # one in-between crop is too little evidence
    )
    out = smooth_teams(tracks, crops, CENTRES, PARAMS)
    assert out.groupby("track_id")["team_id"].first().to_dict() == {1: "other", 2: "unknown", 3: "home"}


def test_split_gives_the_second_kit_its_own_track_id():
    tracks, crops = _clip(_track(1, [DARK] * 10 + [RED] * 10, "home"), _track(2, [RED] * 5, "away"))
    out = split_team_switches(smooth_teams(tracks, crops, CENTRES, PARAMS))
    first, second = out[out["track_id"] == 1], out[(out["sample_id"] >= 10) & (out["team_id"] == "away")]
    assert (first["team_id"] == "home").all() and len(first) == 10
    assert second["track_id"].nunique() == 1 and second["track_id"].iloc[0] not in (1, 2)
    assert out[out["sample_id"] < 5].groupby("track_id").size().to_dict() == {1: 5, 2: 5}


def test_split_leaves_ball_rows_alone():
    tracks, crops = _clip(_track(1, [DARK] * 10 + [RED] * 10, "home"))
    ball = pd.DataFrame([{"sample_id": 3, "track_id": -1, "object_class": "ball", "role": "", "team_id": "ball"}])
    out = split_team_switches(smooth_teams(pd.concat([tracks, ball], ignore_index=True), crops, CENTRES, PARAMS))
    assert out[out["object_class"] == "ball"]["track_id"].tolist() == [-1]
