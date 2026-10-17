"""Inferred regions: a calibrated patch for unseen players, kept apart from observed positions.

Synthetic team of four on fixed spots; player 4 is hidden for part of the clip. Measured coverage on real
tracking (SkillCorner) is in ``docs/evaluation.md`` section 11, not here.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from football_cv.reconstruction import regions as rg

T = np.round(np.arange(0.0, 10.0 + 1e-9, 0.2), 3)
HIDDEN = (T >= 4.0) & (T <= 6.0)


def _team(n_mates=3, motion=(27.0, 25.0), sigma=1.0):
    spots = [(10, 10), (20, 20), (30, 30), (20, 20)][:n_mates]
    rows = []
    for k, (x, y) in enumerate(spots, start=1):
        rows.append(pd.DataFrame({"timestamp_s": T, "identity_id": k, "team_id": "home", "x_m": x, "y_m": y,
                                  "x_est_m": x, "y_est_m": y, "sigma_m": 0.5, "position_source": "OBSERVED"}))
    hidden = pd.DataFrame({
        "timestamp_s": T, "identity_id": 4, "team_id": "home", "x_m": np.where(HIDDEN, np.nan, 25.0),
        "y_m": np.where(HIDDEN, np.nan, 25.0), "x_est_m": np.where(HIDDEN, motion[0], 25.0),
        "y_est_m": np.where(HIDDEN, motion[1], 25.0), "sigma_m": np.where(HIDDEN, sigma, 0.5),
        "position_source": np.where(HIDDEN, "INTERPOLATED", "OBSERVED")})
    return pd.concat(rows + [hidden], ignore_index=True)


def test_fuse_is_precision_weighted():
    mu, var = rg.fuse(np.array([0.0]), np.array([4.0]), np.array([10.0]), np.array([4.0]))
    assert mu[0] == pytest.approx(5.0) and var[0] == pytest.approx(2.0)
    mu, var = rg.fuse(np.array([0.0]), np.array([0.01]), np.array([10.0]), np.array([100.0]))
    assert mu[0] == pytest.approx(0.0, abs=0.01) and var[0] < 0.01


def test_only_unobserved_samples_get_regions_and_they_are_labelled_inferred():
    reg = rg.infer_regions(_team())
    assert list(reg.columns) == rg.REGION_COLUMNS
    assert (reg.identity_id == 4).all() and len(reg) == int(HIDDEN.sum())
    assert set(reg.source) == {"INFERRED"}
    assert reg[["cx", "cy", "sxx", "syy", "area_m2"]].notna().all().all()
    assert reg.time_key.between(4.0, 6.0).all()


def test_region_is_wider_than_the_pitch_mapping_error_and_area_follows_the_ellipse_formula():
    reg = rg.infer_regions(_team())
    cal2 = rg.DEFAULTS["calibration_sigma_m"] ** 2
    assert (reg.sxx > cal2).all() and (reg.syy > cal2).all()
    expected = np.pi * chi2.ppf(0.9, 2) * np.sqrt(reg.sxx * reg.syy)
    assert np.allclose(reg.area_m2, expected)


def test_shape_prior_is_used_with_enough_teammates_and_pulls_the_centre_towards_the_formation_spot():
    reg = rg.infer_regions(_team(n_mates=3))
    assert set(reg.components) == {"motion+shape"}
    # shape says (25, 25), motion says (27, 25); the tighter motion belief dominates but not completely
    assert reg.cx.between(26.0, 27.0).all() and reg.cy.between(24.9, 25.1).all()


def test_too_few_teammates_falls_back_to_motion_only():
    reg = rg.infer_regions(_team(n_mates=2))
    assert set(reg.components) == {"motion"}
    assert np.allclose(reg.cx, 27.0)


def test_components_can_be_switched_off():
    shape_only = rg.infer_regions(_team(), {"regions": {"use_motion": False}})
    assert set(shape_only.components) == {"shape"} and np.allclose(shape_only.cx, 25.0)
    motion_only = rg.infer_regions(_team(), {"regions": {"use_shape": False}})
    assert set(motion_only.components) == {"motion"}
    assert (shape_only.area_m2 > motion_only.area_m2).all()      # the formation prior is the loose one


def test_covers_matches_the_chi_square_ellipse():
    reg = pd.DataFrame({"cx": [0.0, 0.0], "cy": [0.0, 0.0], "sxx": [4.0, 4.0], "syy": [4.0, 4.0]})
    inside = 2.0 * np.sqrt(chi2.ppf(0.9, 2)) * 0.99
    out = covers = rg.covers(reg, np.array([inside, inside * 1.05]), np.array([0.0, 0.0]), 0.9)
    assert list(out) == [True, False]
    assert covers.dtype == bool


def _extend_inputs():
    # one identity observed 2.0-4.0 s in a timeline running 0-10 s; states hold only its observed rows
    t = np.round(np.arange(0.0, 10.0 + 1e-9, 0.2), 3)
    timeline = pd.DataFrame({"sample_id": np.arange(len(t)), "timestamp_s": t, "shot_id": 1})
    seen = t[(t >= 2.0) & (t <= 4.0)]
    states = pd.DataFrame({"timestamp_s": seen, "identity_id": 7, "track_id": 7, "team_id": "home", "shot_id": 1,
                           "sample_id": np.searchsorted(t, seen), "x_m": 30.0 + 2.0 * (seen - 2.0), "y_m": 20.0,
                           "x_est_m": 30.0 + 2.0 * (seen - 2.0), "y_est_m": 20.0, "sigma_m": 0.5,
                           "position_source": "OBSERVED"})
    return states, timeline


def test_extend_states_adds_unobserved_rows_within_the_horizon_only():
    states, timeline = _extend_inputs()
    out = rg.extend_states(states, timeline, horizon_s=3.0)
    added = out[out.position_source == "PREDICTED"]
    assert len(out) == len(states) + len(added) and len(added) > 0
    assert added.timestamp_s.between(-1e-9, 7.0 + 1e-9).all()
    assert not added.timestamp_s.between(2.0 - 1e-9, 4.0 + 1e-9).any()
    assert added.timestamp_s.min() >= 2.0 - 3.0 - 1e-9 or added.timestamp_s.min() == pytest.approx(0.0)
    assert added[["x_m", "y_m"]].isna().all().all()                  # nothing is published as a position
    assert added.x_est_m.notna().all() and (added.sigma_m > 0).all()


def test_extended_uncertainty_grows_with_distance_from_the_last_observation_and_the_belief_runs_on():
    states, timeline = _extend_inputs()
    after = rg.extend_states(states, timeline, horizon_s=5.0).query("position_source == 'PREDICTED' and timestamp_s > 4")
    after = after.sort_values("timestamp_s")
    assert after.sigma_m.is_monotonic_increasing and after.sigma_m.iloc[-1] > after.sigma_m.iloc[0]
    assert after.x_est_m.iloc[0] > 30.0 + 2.0 * 2.0 - 1.0            # still moving the way it was heading


def test_extension_of_a_barely_seen_identity_is_skipped():
    states, timeline = _extend_inputs()
    one = states.iloc[:1]
    assert rg.extend_states(one, timeline, 3.0) is one


def _unseen_case(n_observed, observed_at_centre=False):
    rows = [{"timestamp_s": 1.0, "team_id": "home", "x_m": 5.0 + 8 * k, "y_m": 5.0 + 4 * k, "position_source": "OBSERVED"}
            for k in range(n_observed)]
    if observed_at_centre:
        rows.append({"timestamp_s": 1.0, "team_id": "home", "x_m": 60.0, "y_m": 30.0, "position_source": "OBSERVED"})
    states = pd.DataFrame(rows)
    reg = pd.DataFrame({"time_key": 1.0, "identity_id": [101, 102, 103], "team_id": "home", "cx": [60.0, 90.0, 95.0],
                        "cy": [30.0, 10.0, 60.0], "sxx": 20.0, "syy": 20.0, "area_m2": [300.0, 200.0, 100.0],
                        "components": "motion", "source": "INFERRED"})
    return reg, states


def test_plausible_unseen_keeps_the_smallest_regions_up_to_the_number_of_missing_players():
    reg, states = _unseen_case(n_observed=9)                # 11 - 9 = 2 players can be unseen
    out = rg.mark_plausible_unseen(reg, states)
    assert out.set_index("identity_id").plausible_unseen.to_dict() == {101: False, 102: True, 103: True}
    assert list(out.columns[:-1]) == rg.REGION_COLUMNS and out.source.eq("INFERRED").all()


def test_plausible_unseen_drops_a_region_that_probably_holds_an_observed_teammate():
    reg, states = _unseen_case(n_observed=3, observed_at_centre=True)
    out = rg.mark_plausible_unseen(reg, states).set_index("identity_id")
    assert not out.loc[101, "plausible_unseen"] and out.loc[102, "plausible_unseen"]


def test_plausible_unseen_is_empty_when_the_whole_team_is_on_screen():
    reg, states = _unseen_case(n_observed=11)
    assert not rg.mark_plausible_unseen(reg, states).plausible_unseen.any()
    assert rg.mark_plausible_unseen(reg.iloc[0:0], states).empty


def test_regions_of_unassigned_teams_are_never_plausible_unseen():
    reg, states = _unseen_case(n_observed=3)
    assert not rg.mark_plausible_unseen(reg.assign(team_id="other"), states).plausible_unseen.any()
