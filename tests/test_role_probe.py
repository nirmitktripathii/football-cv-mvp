"""Role probe and keeper prior (docs/role_probe.md stage 2): model maths, prior rule, relabel guards, stage no-op."""
import numpy as np
import pandas as pd
import pytest

from football_cv import pipeline
from football_cv.calibration.projection import H_COLS
from football_cv.tracking import role_probe as rp

PITCH = (105.0, 68.0)
SETTINGS = {"min_side_px": 8, "feature": "fc", "penalty_depth_m": 16.5, "penalty_half_width_m": 20.16, "tie_m": 2.0,
            "keeper_rule": "prior_or_probe"}


def _prior(xy, eligible=None, p_keeper=None, frame=None):
    xy = np.array(xy, float)
    n = len(xy)
    return rp.keeper_prior(np.zeros(n, int) if frame is None else np.array(frame),
                           xy, np.ones(n, bool) if eligible is None else np.array(eligible),
                           np.zeros(n) if p_keeper is None else np.array(p_keeper), PITCH, 16.5, 20.16, 2.0)


def test_role_model_matches_sklearn_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat(rp.CLASSES, 40)
    x = rng.normal(size=(120, 6)) + np.repeat(np.eye(3, 6) * 4, 40, axis=0)
    model = rp.RoleModel.fit(x, y)
    proba = model.proba(x)
    assert proba.shape == (120, 3) and np.allclose(proba.sum(axis=1), 1)
    assert (np.array(rp.CLASSES)[proba.argmax(axis=1)] == y).mean() > 0.95
    model.save(tmp_path / "m.npz")
    again = rp.RoleModel.load(tmp_path / "m.npz")
    assert np.allclose(again.proba(x), proba)


def test_role_model_refuses_other_class_order(tmp_path):
    np.savez(tmp_path / "bad.npz", mean=np.zeros(2), scale=np.ones(2), coef=np.zeros((3, 2)), intercept=np.zeros(3),
             classes=np.array(["referee", "player", "goalkeeper"]))
    with pytest.raises(ValueError):
        rp.RoleModel.load(tmp_path / "bad.npz")


def test_prior_picks_box_nearest_goal_centre_at_each_end():
    fires = _prior([[2.0, 34.0], [10.0, 30.0], [103.0, 33.0], [60.0, 34.0]])
    assert fires.tolist() == [True, False, True, False]


def test_prior_ignores_boxes_outside_the_penalty_area_or_pitch():
    fires = _prior([[20.0, 34.0], [-1.0, 34.0], [3.0, 60.0]])
    assert not fires.any()


def test_prior_excludes_ineligible_boxes():
    fires = _prior([[1.0, 34.0], [4.0, 34.0]], eligible=[False, True])
    assert fires.tolist() == [False, True]


def test_prior_breaks_ties_within_tolerance_by_probe_probability():
    fires = _prior([[2.0, 34.0], [3.5, 34.0], [8.0, 34.0]], p_keeper=[0.1, 0.9, 0.99])
    assert fires.tolist() == [False, True, False]  # 8 m is beyond the 2 m tie band, so its high probability is ignored


def test_prior_is_per_frame():
    fires = _prior([[2.0, 34.0], [2.0, 34.0]], frame=[0, 1])
    assert fires.tolist() == [True, True]


def test_assign_roles_rules():
    p = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
    fires = np.array([False, False, True, False])
    assert rp.assign_roles(p, fires, "prior_or_probe").tolist() == ["player", "goalkeeper", "referee", "referee"]
    assert rp.assign_roles(p, np.array([True, False, False, False]), "prior_only").tolist() == [
        "goalkeeper", "player", "referee", "referee"]
    with pytest.raises(ValueError):
        rp.assign_roles(p, fires, "nope")


class _Embedder:
    """Fake embedder: the crop's mean grey level decides the class (dark keeper, bright referee, mid player)."""

    def embed(self, crops):
        return {"fc": np.array([[c.mean()] for c in crops], float)}


def _model():
    # one feature: <60 goalkeeper, >200 referee, otherwise player
    coef = np.array([[0.0], [-1.0], [1.0]])
    return rp.RoleModel(np.zeros(1), np.ones(1), coef * 0.2, np.array([0.0, 12.0, -40.0]))


def _identity_calibration(sample_ids, status="keyframe"):
    row = dict(zip(H_COLS, np.eye(3).ravel()))
    return pd.DataFrame([{"sample_id": s, "calibration_status": status, **row} for s in sample_ids])


def _tracks(rows):
    cols = ["sample_id", "track_id", "object_class", "role", "team_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "foot_x", "foot_y"]
    return pd.DataFrame(rows, columns=cols)


def _frame_on_disk(tmp_path, grey=128):
    import cv2

    img = np.full((200, 200, 3), grey, np.uint8)
    cv2.imwrite(str(tmp_path / "f0.jpg"), img)
    return pd.DataFrame({"sample_id": [0, 1], "path": ["f0.jpg", "f0.jpg"]})


def test_relabel_keeps_rows_without_crop_or_calibration_and_sets_team(tmp_path):
    frames = _frame_on_disk(tmp_path, grey=30)  # dark crops: the fake probe says goalkeeper
    tracks = _tracks([
        [0, 1, "person", "player", "home", 10, 10, 40, 60, 5.0, 34.0],     # crop + calibration, near the left goal
        [0, 2, "person", "referee", "other", 50, 10, 53, 12, 50.0, 30.0],  # crop too small -> keeps detector role
        [1, 3, "person", "player", "home", 10, 10, 40, 60, 5.0, 30.0],     # sample without calibration -> keeps role
        [0, -1, "ball", "ball", "unknown", 100, 100, 104, 104, 80.0, 30.0],
    ])
    calib = _identity_calibration([0])
    out, report = rp.relabel(tracks, calib, frames, tmp_path, _model(), _Embedder(), {1: "home"}, SETTINGS, PITCH)
    assert out["role"].tolist() == ["goalkeeper", "referee", "player", "ball"]
    assert out["team_id"].tolist() == ["unknown", "other", "home", "unknown"]
    assert out["role_pipeline"].tolist() == ["player", "referee", "player", "ball"]
    assert out["role_source"].tolist()[0] in ("prior", "probe") and out["role_source"].tolist()[1:] == ["detector"] * 3
    assert report["person_rows"] == 3 and report["scored_rows"] == 1 and report["no_crop_rows"] == 1
    assert report["no_calibration_rows"] == 1 and report["changed_rows"] == 1


def test_relabel_demoted_keeper_takes_kit_team_and_referee_becomes_other(tmp_path):
    frames = _frame_on_disk(tmp_path, grey=128)  # mid grey: the fake probe says player
    tracks = _tracks([
        [0, 5, "person", "goalkeeper", "unknown", 10, 10, 40, 60, 50.0, 34.0],
        [0, 6, "person", "referee", "other", 60, 10, 90, 60, 60.0, 34.0],
    ])
    out, _ = rp.relabel(tracks, _identity_calibration([0]), frames, tmp_path, _model(), _Embedder(),
                        {5: "away", 6: "home"}, SETTINGS, PITCH)
    assert out["role"].tolist() == ["player", "player"]
    assert out["team_id"].tolist() == ["away", "home"]


def test_roles_stage_is_a_no_op_when_disabled(tmp_path):
    cfg = {"role_probe": {"enabled": False}, "paths": {"processed_dir": str(tmp_path)}, "clip": {"id": "x"}}
    assert pipeline.stage_roles(cfg) is None
    assert not list(tmp_path.iterdir())


def test_roles_stage_is_registered_between_calibrate_and_identity():
    stages = list(pipeline.STAGES)
    assert stages.index("calibrate") + 1 == stages.index("roles") == stages.index("identity") - 1
