"""Conservative tracklet linking: what it links, and (more important) what it refuses to link.

Synthetic constant-velocity people on a pitch. Real-data numbers (SkillCorner precision / recall)
live in ``docs/evaluation.md`` section 11, not here.
"""
import numpy as np
import pandas as pd
import pytest

from football_cv.tracking import identity as idn
from football_cv.tracking.appearance import DESCRIPTOR_COLUMNS, describe_tracks

DT = 0.2


def _track(tid, t0, t1, x0, y0, vx=3.0, vy=0.0, team="home", shot=1, status="confident", edge0=False, edge1=False, jump=False):
    t = np.arange(t0, t1 + 1e-9, DT)
    edge = np.zeros(len(t), bool)
    edge[0], edge[-1] = edge0, edge1
    return pd.DataFrame({
        "track_id": tid, "shot_id": shot, "timestamp_s": t, "x_s": x0 + vx * (t - t0), "y_s": y0 + vy * (t - t0),
        "jump": jump, "team_id": team, "team_status": status, "edge": edge,
    })


def _link(*tracks, cfg=None, appearance=None):
    return idn.link_tracklets(pd.concat(tracks, ignore_index=True), cfg, appearance)


def _status(decisions, a, b):
    row = decisions[(decisions.pred_track_id == a) & (decisions.succ_track_id == b)]
    return None if row.empty else row.status.iloc[0]


def test_unambiguous_continuation_is_linked():
    a = _track(1, 0, 3, 30, 30)
    b = _track(2, 4, 8, 42, 30)            # exactly where a was heading, 1 s later
    ident, dec, _ = _link(a, b)
    assert _status(dec, 1, 2) == "linked"
    assert ident.set_index("track_id").identity_id.to_dict() == {1: 1, 2: 1}


def test_two_plausible_successors_are_left_unlinked():
    a = _track(1, 0, 3, 30, 30)
    b1 = _track(2, 4, 8, 42, 30)
    b2 = _track(3, 4, 8, 43.5, 31)         # a team-mate 1.8 m away: not clearly worse
    ident, dec, _ = _link(a, b1, b2)
    assert {_status(dec, 1, 2), _status(dec, 1, 3)} <= {"ambiguous", "not_best"}
    assert "linked" not in set(dec.status)
    assert ident.identity_id.nunique() == 3


def test_ambiguity_among_predecessors_is_also_refused():
    a1 = _track(1, 0, 3, 30, 30)
    a2 = _track(2, 0, 3, 31, 31)
    b = _track(3, 4, 8, 42, 30)
    _, dec, _ = _link(a1, a2, b)
    assert "linked" not in set(dec.status)


@pytest.mark.parametrize("kwargs", [
    {"team": "away"},                                   # different team
    {"shot": 2},                                        # different camera shot
    {"status": "uncertain"},                            # team label not confident
])
def test_gates_block_linking(kwargs):
    a = _track(1, 0, 3, 30, 30)
    b = _track(2, 4, 8, 42, 30, **kwargs)
    _, dec, _ = _link(a, b)
    assert dec.empty or "linked" not in set(dec.status)


def test_other_team_labels_are_never_linked():
    a = _track(1, 0, 3, 30, 30, team="other")
    b = _track(2, 4, 8, 42, 30, team="other")
    _, dec, _ = _link(a, b)
    assert dec.empty


def test_break_at_the_view_edge_is_not_linked_unless_enabled():
    a = _track(1, 0, 3, 30, 30, edge1=True)
    b = _track(2, 4, 8, 42, 30)
    assert _link(a, b)[1].empty
    _, dec, _ = _link(a, b, cfg={"identity": {"exclude_edge_breaks": False}})
    assert _status(dec, 1, 2) == "linked"


def test_pair_faster_than_a_person_can_run_is_rejected():
    a = _track(1, 0, 3, 30, 30)
    b = _track(2, 4, 8, 90, 30)
    assert _link(a, b)[1].empty


def test_lone_candidate_with_a_long_gap_is_weak_not_linked():
    a = _track(1, 0, 3, 30, 30, vx=0.0)
    b = _track(2, 9, 12, 42, 33, vx=0.0)   # 6 s later, 12 m away: fits only because the gate is wide by then
    _, dec, _ = _link(a, b, cfg={"identity": {"max_gap_s": 8.0}})
    assert _status(dec, 1, 2) == "weak"
    assert dec.false_matches.iloc[0] > idn.DEFAULTS["max_false_matches"]


def test_overlapping_in_time_is_not_a_continuation():
    a = _track(1, 0, 6, 30, 30)
    b = _track(2, 4, 8, 42, 30)
    assert _link(a, b)[1].empty


def test_jump_rows_are_ignored_in_tracklet_summary():
    good = _track(1, 0, 3, 30, 30)
    bad = _track(1, 3.2, 4, 90, 90, jump=True)
    trk = idn.tracklet_summary(pd.concat([good, bad]))
    assert trk.t1.iloc[0] == pytest.approx(3.0) and trk.x1.iloc[0] == pytest.approx(39.0)


def test_chain_takes_smallest_track_id_and_ambiguous_pairs_do_not_merge():
    links = pd.DataFrame({"pred_track_id": [5, 7, 9], "succ_track_id": [7, 9, 11], "gap_s": 1.0, "cost": 0.1,
                          "false_matches": 0.0, "margin": [np.inf, np.inf, 1.0],
                          "status": ["linked", "linked", "ambiguous"]})
    m = idn.assign_identities([5, 7, 9, 11], links).set_index("track_id").identity_id.to_dict()
    assert m == {5: 5, 7: 5, 9: 5, 11: 11}


def _descriptor(tid, shorts_l):
    row = {c: 100.0 for c in DESCRIPTOR_COLUMNS} | {"d_shorts_L": shorts_l, "track_id": tid, "spread": 1.0, "n": 10}
    return row


def test_appearance_veto_blocks_a_different_kit_but_not_a_matching_one():
    a, b = _track(1, 0, 3, 30, 30), _track(2, 4, 8, 42, 30)
    same = pd.DataFrame([_descriptor(1, 100.0), _descriptor(2, 101.0)])
    different = pd.DataFrame([_descriptor(1, 100.0), _descriptor(2, 180.0)])
    assert _status(_link(a, b, appearance=same)[1], 1, 2) == "linked"
    assert _link(a, b, appearance=different)[1].empty


def test_missing_descriptor_never_vetoes():
    a, b = _track(1, 0, 3, 30, 30), _track(2, 4, 8, 42, 30)
    only_one = pd.DataFrame([_descriptor(1, 100.0)])
    assert _status(_link(a, b, appearance=only_one)[1], 1, 2) == "linked"


def test_describe_tracks_median_and_minimum_crops():
    rng = np.random.default_rng(0)
    rows = [{"track_id": 1, **{c: 50.0 + rng.normal() for c in DESCRIPTOR_COLUMNS}} for _ in range(6)]
    rows += [{"track_id": 2, **{c: 10.0 for c in DESCRIPTOR_COLUMNS}} for _ in range(2)]
    desc = describe_tracks(pd.DataFrame(rows))
    assert list(desc.track_id) == [1] and desc.spread.iloc[0] > 0 and desc.n.iloc[0] == 6


def test_undecided_team_links_only_when_allowed_and_never_across_confident_teams():
    a = _track(1, 0, 3, 30, 30)
    b = _track(2, 4, 8, 42, 30, team="unknown", status="unknown")
    assert "linked" not in set(_link(a, b)[1].status)
    cfg = {"identity": {"allow_unknown_team": True}}
    assert _status(_link(a, b, cfg=cfg)[1], 1, 2) == "linked"
    other_team = _track(3, 4, 8, 42, 30, team="away")
    assert _link(a, other_team, cfg=cfg)[1].empty


def test_referee_ends_never_link_even_with_unknown_teams_allowed():
    a = _track(1, 0, 3, 30, 30, team="other", status="other")
    b = _track(2, 4, 8, 42, 30, team="unknown", status="unknown")
    assert _link(a, b, cfg={"identity": {"allow_unknown_team": True}})[1].empty


def test_kalman_ends_link_a_clean_continuation_and_reject_a_far_one():
    cfg = {"identity": {"kalman_ends": True}}
    a = _track(1, 0, 3, 30, 30)
    assert _status(_link(a, _track(2, 4, 8, 42, 30), cfg=cfg)[1], 1, 2) == "linked"
    assert _link(a, _track(3, 4, 8, 42, 55), cfg=cfg)[1].empty


def test_chain_reports_its_weakest_link_margin():
    links = pd.DataFrame({"pred_track_id": [5, 7], "succ_track_id": [7, 9], "gap_s": 1.0, "cost": 0.1,
                          "false_matches": 0.0, "margin": [np.inf, 0.8], "status": "linked"})
    m = idn.assign_identities([5, 7, 9, 11], links).set_index("track_id").chain_min_margin
    assert m[5] == m[9] == pytest.approx(0.8) and np.isnan(m[11])
