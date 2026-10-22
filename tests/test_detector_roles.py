"""Class-set recognition and role-aware team logic, on tiny synthetic inputs."""
import pandas as pd
import pytest

from football_cv.detection.detector import class_roles, detection_summary
from football_cv.tracking import teams


def test_four_class_model_maps_people_to_person_with_a_role():
    roles = class_roles({0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"})
    assert roles == {0: ("ball", ""), 1: ("person", "goalkeeper"), 2: ("person", "player"), 3: ("person", "referee")}


def test_coco_model_keeps_person_and_ball_without_roles():
    assert class_roles({0: "person", 32: "sports ball", 5: "bus"}) == {0: ("person", ""), 32: ("ball", "")}


def test_unknown_class_set_is_rejected():
    with pytest.raises(ValueError, match="Unrecognised"):
        class_roles({0: "cat", 1: "dog"})


def _tracks():
    rows = []
    for tid, roles in {1: ["player"] * 3, 2: ["goalkeeper", "player", "goalkeeper"], 3: ["referee"] * 2, 4: ["player"]}.items():
        rows += [{"track_id": tid, "object_class": "person", "role": r} for r in roles]
    rows.append({"track_id": -1, "object_class": "ball", "role": ""})
    return pd.DataFrame(rows)


def test_majority_role_per_track():
    assert teams.majority_role(_tracks()).to_dict() == {1: "player", 2: "goalkeeper", 3: "referee", 4: "player"}


def test_referee_gets_no_team_and_goalkeeper_is_not_named_by_kit_colour():
    tr = _tracks()
    table = pd.DataFrame({"track_id": [1, 2, 3, 4], "team_id": ["home", "away", "away", "home"]})
    out = teams.apply_roles(tr, table, teams.majority_role(tr))
    by_track = out.groupby("track_id")[["role", "team_id"]].first()
    assert by_track.loc[1].tolist() == ["player", "home"]
    assert by_track.loc[2].tolist() == ["goalkeeper", "unknown"]
    assert by_track.loc[3].tolist() == ["referee", "other"]
    assert by_track.loc[-1, "role"] == ""


def test_summary_reports_role_counts_only_for_role_aware_detections():
    dets = pd.DataFrame({"sample_id": [0, 0, 0], "object_class": ["person", "person", "ball"],
                         "role": ["player", "referee", ""], "confidence": [0.9, 0.8, 0.7]})
    assert detection_summary(dets, 1)["person"]["by_role"] == {"player": 1, "referee": 1}
    assert "by_role" not in detection_summary(dets.assign(role=""), 1)["person"]
