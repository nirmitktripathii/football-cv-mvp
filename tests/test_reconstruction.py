"""Reconstruction, player states, masking evaluation, team confidence and the state renderer.

Everything here is synthetic: a constant-velocity player seen through a known camera homography.
The real-footage numbers live in ``docs/evaluation.md`` and ``artifacts/<clip>/reconstruction``.
"""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from football_cv.calibration.motion import apply_h
from football_cv.calibration.projection import H_COLS
from football_cv.reconstruction import evaluate as ev
from football_cv.reconstruction import motion_models as mm
from football_cv.reconstruction import reconstruct as rc
from football_cv.reconstruction.states import ANNOTATION_ONLY_STATES, PlayerState, PositionSource
from football_cv.tracking.teams import add_team_confidence

DT = 0.2
VX, VY = 3.0, 0.5            # m/s of the synthetic player
X0, Y0 = 40.0, 30.0


def _camera_h():
    """Image -> pitch homography of a broadcast-like camera (pitch x 20..85 m visible at the near edge)."""
    pitch = np.array([[20, 0], [85, 0], [80, 68], [25, 68]], dtype=float)
    image = np.array([[100, 700], [1180, 700], [1000, 180], [280, 180]], dtype=float)
    H_p2i, _ = cv2.findHomography(pitch, image, 0)
    H_i2p = np.linalg.inv(H_p2i)
    return H_i2p / H_i2p[2, 2]


H = _camera_h()


def _world(n=60, missing=(), shots=None, status=None, jumps=(), ident=7):
    """Timeline, calibration and trajectory of one identity moving at constant velocity.

    ``missing`` sample ids are not detected; ``shots[i]`` is the shot of sample i; ``status`` maps a
    sample id to its calibration status (default ``propagated``); ``jumps`` are detected-but-rejected.
    """
    sid = np.arange(n)
    t = sid * DT
    shots = np.zeros(n, int) if shots is None else np.asarray(shots)
    timeline = pd.DataFrame({"sample_id": sid, "timestamp_s": t, "shot_id": shots})
    status = status or {}
    calib = pd.DataFrame({"sample_id": sid, "calibration_status": [status.get(i, "propagated") for i in sid],
                          "keyframe_dt_s": 1.0})
    for k, c in enumerate(H_COLS):
        calib[c] = H.ravel()[k]
    keep = [i for i in sid if i not in set(missing) and status.get(i, "propagated") != "not_wide_view"]
    traj = pd.DataFrame({
        "track_id": ident, "sample_id": keep, "timestamp_s": t[keep], "shot_id": shots[keep], "team_id": "home",
        "x_s": X0 + VX * t[keep], "y_s": Y0 + VY * t[keep], "confidence": 0.9,
        "jump": [i in set(jumps) for i in keep], "short_track": False,
    })
    return traj, timeline, calib


def _run(traj, timeline, calib, cfg=None, annotations=None):
    states, gaps, warnings = rc.reconstruct(traj, timeline, calib, cfg, annotations)
    return states.set_index("sample_id"), gaps, warnings


# ---- motion models ---------------------------------------------------------------------------

def _line(n=40):
    t = np.arange(n) * DT
    return t, np.column_stack([X0 + VX * t, Y0 + VY * t])


def test_models_recover_constant_velocity_inside_a_gap():
    t, truth = _line()
    observed = np.ones(len(t), bool)
    observed[15:20] = False
    z = np.where(observed[:, None], truth, np.nan)
    out = mm.reconstruct_methods(t, z, observed)
    for name in ("linear", "rts"):
        assert np.abs(out[name][0][15:20] - truth[15:20]).max() < 0.3, name


def test_linear_has_no_estimate_outside_the_observed_span():
    t, truth = _line()
    observed = np.zeros(len(t), bool)
    observed[10:20] = True
    z = np.where(observed[:, None], truth, np.nan)
    mean, _ = mm.reconstruct_methods(t, z, observed)["linear"]
    assert np.isnan(mean[:10]).all() and np.isnan(mean[20:]).all()


def test_prediction_uncertainty_grows_with_time_since_last_observation():
    t, truth = _line()
    observed = np.zeros(len(t), bool)
    observed[:20] = True
    z = np.where(observed[:, None], truth, np.nan)
    out = mm.reconstruct_methods(t, z, observed)
    for name in ("hold", "cv", "kalman"):
        std = out[name][1][20:]
        assert np.all(np.diff(std) > 0), name


def test_rts_uncertainty_peaks_in_the_middle_of_a_gap():
    t, truth = _line()
    observed = np.ones(len(t), bool)
    observed[10:30] = False
    z = np.where(observed[:, None], truth, np.nan)
    std = mm.reconstruct_methods(t, z, observed)["rts"][1]
    assert std[20] > std[10] and std[20] > std[29]
    assert std[9] < std[20]


def test_cv_extrapolation_continues_the_last_velocity():
    t, truth = _line()
    observed = np.zeros(len(t), bool)
    observed[:20] = True
    z = np.where(observed[:, None], truth, np.nan)
    mean, _ = mm.reconstruct_methods(t, z, observed)["cv"]
    assert np.abs(mean[20:25] - truth[20:25]).max() < 0.2


# ---- view geometry ---------------------------------------------------------------------------

def test_homography_round_trip():
    pts = np.array([[640.0, 400.0], [100.0, 700.0], [1000.0, 200.0]])
    pitch = apply_h(H, pts)
    assert apply_h(np.linalg.inv(H), pitch) == pytest.approx(pts, abs=1e-6)


def test_view_state_inside_outside_and_straddling():
    size, margin = (1280, 720), 20.0
    inside = apply_h(H, np.array([[640.0, 400.0]]))[0]
    assert rc.view_state(H, inside, 0.5, size, margin) is PlayerState.OCCLUDED
    assert rc.view_state(H, np.array([-40.0, 34.0]), 0.5, size, margin) is PlayerState.OUTSIDE_CAMERA_VIEW
    edge = apply_h(H, np.array([[1280.0, 400.0]]))[0]
    assert rc.view_state(H, edge, 3.0, size, margin) is PlayerState.UNKNOWN_MISSING


# ---- reconstruction --------------------------------------------------------------------------

def test_observed_samples_are_published_unchanged():
    traj, tl, calib = _world(missing=range(20, 25))
    st, _, _ = _run(traj, tl, calib)
    obs = st[st["position_source"] == PositionSource.OBSERVED.value]
    assert len(obs) == len(traj)
    assert obs["x_m"].to_numpy() == pytest.approx(traj.set_index("sample_id").loc[obs.index, "x_s"].to_numpy())
    assert (obs["state"] == PlayerState.VISIBLE_ON_PITCH.value).all()
    assert (obs["sigma_m"] == mm.SIGMA_MEAS).all()


def test_short_interior_gap_is_interpolated_and_marked_occluded():
    traj, tl, calib = _world(missing=range(20, 25))
    st, gaps, _ = _run(traj, tl, calib)
    gap = st.loc[20:24]
    assert (gap["position_source"] == PositionSource.INTERPOLATED.value).all()
    assert (gap["state"] == PlayerState.OCCLUDED.value).all()
    assert gap["x_m"].to_numpy() == pytest.approx(X0 + VX * gap["timestamp_s"].to_numpy(), abs=0.3)
    assert gap["gap_s"].iloc[0] == pytest.approx(6 * DT)
    assert gaps[gaps["kind"] == "interior"]["filled"].all()


def test_trailing_prediction_stops_at_the_horizon():
    traj, tl, calib = _world(n=60, missing=range(30, 60))
    st, _, _ = _run(traj, tl, calib)
    tail = st.loc[30:]
    assert tail["timestamp_s"].max() <= st.loc[29, "timestamp_s"] + rc.DEFAULTS["predict_horizon_s"] + 1e-9
    assert tail["sigma_m"].is_monotonic_increasing
    published = tail[tail["position_source"] != ""]
    assert len(published) and (published["position_source"] == PositionSource.PREDICTED.value).all()
    withheld = tail[tail["position_source"] == ""]
    assert withheld["x_m"].isna().all() and withheld["reason"].str.contains("max_sigma_m").all()
    assert (published["sigma_m"] <= rc.DEFAULTS["max_sigma_m"]).all()


def test_uncertainty_grows_with_gap_length():
    short, tl, calib = _world(missing=range(20, 23))
    longer, _, _ = _world(missing=range(20, 36))
    s_short, _, _ = _run(short, tl, calib)
    s_long, _, _ = _run(longer, tl, calib)
    def worst(st):
        return st.loc[st["position_source"] != PositionSource.OBSERVED.value, "sigma_m"].max()

    assert worst(s_long) > worst(s_short)


def test_gap_longer_than_max_gap_is_not_filled_but_keeps_its_estimate():
    traj, tl, calib = _world(n=120, missing=range(20, 80))     # 12 s
    st, gaps, _ = _run(traj, tl, calib)
    mid = st.loc[50]
    assert np.isnan(mid["x_m"]) and mid["position_source"] == ""
    assert np.isfinite(mid["x_est_m"]) and "max_gap_s" in mid["reason"]
    assert not gaps[gaps["kind"] == "interior"]["filled"].any()


def test_uncalibrated_frames_get_no_coordinates():
    traj, tl, calib = _world(missing=range(20, 25), status={22: "rejected:drift", 23: "rejected:drift"})
    st, _, _ = _run(traj, tl, calib)
    bad = st.loc[[22, 23]]
    assert bad["x_m"].isna().all() and bad["x_est_m"].isna().all()
    assert (bad["state"] == PlayerState.UNKNOWN_MISSING.value).all()
    assert bad["reason"].str.contains("uncalibrated").all()
    assert st.loc[20, "position_source"] == PositionSource.INTERPOLATED.value


def test_gaps_are_never_bridged_across_a_camera_cut():
    shots = [0] * 20 + [1] * 20
    traj, tl, calib = _world(n=40, shots=shots, missing=range(20, 40))
    traj = traj.iloc[:20].copy()
    late = traj.iloc[:5].copy()
    late["sample_id"] = np.arange(35, 40)
    late["timestamp_s"] = late["sample_id"] * DT
    late["shot_id"] = 1
    late["x_s"] = X0 + VX * late["timestamp_s"]
    late["y_s"] = Y0 + VY * late["timestamp_s"]
    st, gaps, _ = _run(pd.concat([traj, late], ignore_index=True), tl, calib)
    between = st.loc[(st.index >= 20) & (st.index < 35)]
    assert not (between["position_source"] == PositionSource.INTERPOLATED.value).any()
    cut = gaps[gaps["kind"] == "across_shots"]
    assert len(cut) == 1 and not cut["filled"].iloc[0]


def test_jump_rows_are_reported_as_rejected_not_filled():
    traj, tl, calib = _world(jumps=[30])
    st, _, _ = _run(traj, tl, calib)
    row = st.loc[30]
    assert row["state"] == PlayerState.UNKNOWN_MISSING.value and np.isnan(row["x_m"])
    assert "jump" in row["reason"]


def test_no_gap_ever_produces_an_annotation_only_state():
    traj, tl, calib = _world(n=120, missing=list(range(20, 25)) + list(range(40, 100)))
    st, _, _ = _run(traj, tl, calib)
    assert not set(st["state"]) & {s.value for s in ANNOTATION_ONLY_STATES}
    assert not (st["position_source"] == PositionSource.MANUALLY_ANNOTATED.value).any()


def test_far_gap_never_claims_a_state_that_needs_annotation():
    traj, tl, calib = _world(n=80, missing=range(30, 80))
    traj["x_s"] = (traj["x_s"] + 40.0).clip(upper=88.0)
    st, _, _ = _run(traj, tl, calib)
    states = set(st.loc[30:, "state"])
    assert states <= {PlayerState.OUTSIDE_CAMERA_VIEW.value, PlayerState.UNKNOWN_MISSING.value, PlayerState.OCCLUDED.value}


# ---- annotations -----------------------------------------------------------------------------

def test_annotations_must_use_annotation_only_states(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(json.dumps([{"identity_id": 7, "state": "OCCLUDED", "start_s": 1, "end_s": 2}]))
    with pytest.raises(ValueError):
        rc.load_annotations(path)
    path.write_text(json.dumps([{"identity_id": 7, "state": "OFF_PITCH_INJURY_TREATMENT", "start_s": 3, "end_s": 2}]))
    with pytest.raises(ValueError):
        rc.load_annotations(path)
    assert rc.load_annotations(None) == [] and rc.load_annotations(tmp_path / "missing.json") == []


def test_annotation_marks_a_missing_stretch_and_never_overwrites_observations():
    traj, tl, calib = _world(missing=range(20, 30))
    ann = [{"identity_id": 7, "state": "OFF_PITCH_INJURY_TREATMENT", "start_s": 3.5, "end_s": 4.5,
            "x_m": 52.5, "y_m": -3.0, "note": "treatment by physio"}]
    st, _, warnings = _run(traj, tl, calib, annotations=ann)
    inside_gap = st.loc[20:29]
    inside_gap = inside_gap[inside_gap["timestamp_s"].between(3.5, 4.5)]
    assert len(inside_gap) and (inside_gap["state"] == "OFF_PITCH_INJURY_TREATMENT").all()
    assert (inside_gap["position_source"] == PositionSource.MANUALLY_ANNOTATED.value).all()
    observed_in_window = st[(st["timestamp_s"] >= 3.5) & (st["timestamp_s"] < 4.0)]
    assert (observed_in_window["position_source"] == PositionSource.OBSERVED.value).all()
    assert warnings and "observed" in warnings[0]


def test_annotation_without_coordinates_keeps_position_unknown():
    traj, tl, calib = _world(missing=range(20, 30))
    ann = [{"identity_id": 7, "state": "SUBSTITUTED_CONFIRMED", "start_s": 4.0, "end_s": 5.0}]
    st, _, _ = _run(traj, tl, calib, annotations=ann)
    row = st.loc[22]
    assert row["state"] == "SUBSTITUTED_CONFIRMED" and np.isnan(row["x_m"]) and row["position_source"] == ""


# ---- team confidence -------------------------------------------------------------------------

def test_team_confidence_flags_tracks_whose_crops_disagree():
    rng = np.random.default_rng(0)
    centres = np.array([[70.0, 140.0, 85.0], [235.0, 128.0, 128.0]])
    names = {0: "home", 1: "away"}
    rows = []
    for track, kit, n in [(1, 0, 20), (2, 1, 20), (3, 0, 20), (4, 0, 3)]:
        for k in range(n):
            c = centres[kit] + rng.normal(0, 2, 3)
            if track == 3 and k % 2:
                c = centres[1] + rng.normal(0, 2, 3)          # half the crops look like the other kit
            rows.append({"track_id": track, "L": c[0], "a": c[1], "b": c[2]})
    colours = pd.DataFrame(rows)
    table = pd.DataFrame({"track_id": [1, 2, 3, 4, 5], "team_id": ["home", "away", "home", "home", "other"],
                          "colour_distance": 0.0, "n_colour_samples": [20, 20, 20, 3, 8]})
    out = add_team_confidence(table, colours, centres, names).set_index("track_id")
    assert out["team_id"].tolist() == ["home", "away", "home", "home", "other"]       # labels untouched
    assert out.loc[[1, 2], "team_status"].tolist() == ["confident", "confident"]
    assert out.loc[3, "team_status"] == "uncertain" and out.loc[3, "agreement"] == pytest.approx(0.5)
    assert out.loc[4, "team_status"] == "uncertain"                                   # too few crops
    assert out.loc[5, "team_status"] == "other"


# ---- masking evaluation ----------------------------------------------------------------------

def _good(n=80):
    t = np.arange(n) * DT
    return pd.DataFrame({"track_id": 1, "shot_id": 0, "sample_id": np.arange(n), "timestamp_s": t,
                         "x_s": X0 + VX * t, "y_s": Y0 + VY * t})


def test_masking_eval_is_near_exact_on_straight_runs_and_covers_every_method():
    res = ev.masking_eval(_good(), gap_lengths_s=(1.0, 2.0))
    assert set(res["method"]) == set(ev.METHODS)
    summ = ev.summarise(res).set_index(["gap_s", "method"])
    assert summ.loc[(2.0, "linear"), "mae_m"] < 1e-6
    assert summ.loc[(2.0, "rts"), "mae_m"] < 0.3
    assert summ.loc[(2.0, "hold"), "mae_m"] > summ.loc[(2.0, "cv"), "mae_m"]


def test_masking_eval_does_not_leak_the_hidden_window():
    good = _good()
    biased = good.assign(x_s=good["x_s"] + np.where(good["sample_id"].between(5, 12), 25.0, 0.0))
    res = ev.masking_eval(biased, gap_lengths_s=(1.0,))
    first = res[(res["start"] == res["start"].min()) & (res["method"] == "linear")]
    assert first["err_m"].max() > 10       # a bias inside the hidden window cannot be recovered from outside it


def test_masking_eval_skips_runs_too_short_for_context():
    assert ev.masking_eval(_good(n=8), gap_lengths_s=(2.0,)).empty


# ---- renderer --------------------------------------------------------------------------------

def test_state_renderer_writes_one_frame_per_sample_in_the_window(tmp_path):
    from PIL import Image
    from football_cv.visualization.pitch_states import plot_state_snapshot, render_state_animation

    traj, tl, calib = _world(n=30, missing=range(10, 15), shots=[0] * 25 + [1] * 5, status={20: "not_wide_view"})
    st, _, _ = rc.reconstruct(traj, tl, calib)
    frames = tl.drop(columns="shot_id").assign(frame_id=tl["sample_id"] * 6, match_clock="00:00", replay_s=0.0, path="x.jpg")
    shots = tl[["sample_id", "shot_id"]].assign(timestamp_s=tl["timestamp_s"])
    ball = pd.DataFrame({"track_id": [-1], "jump": [False], "x_s": [50.0], "y_s": [30.0], "sample_id": [3]})
    out = render_state_animation(st, ball, frames, shots, calib, tmp_path / "s.gif", start_s=3.5, end_s=5.5, fps=5)
    expected = int(((tl["timestamp_s"] >= 3.5) & (tl["timestamp_s"] <= 5.5)).sum())
    assert Image.open(out).n_frames == expected

    flat = st.copy()
    _, _, counts = plot_state_snapshot(flat, 12)
    assert counts["interpolated"] == 1 and counts["observed"] == 0
    _, _, counts2 = plot_state_snapshot(flat, 5)
    assert counts2["observed"] == 1
