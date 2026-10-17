"""SkillCorner loader and the scoring helpers of the linker evaluation, on tiny synthetic inputs.

The numbers measured on the real open match (data/external/skillcorner, git-ignored) are reported in
``docs/evaluation.md`` section 11; these tests only pin the behaviour those numbers depend on.
"""
import json

import numpy as np
import pandas as pd
import pytest

from football_cv.evaluation import linker_eval as le
from football_cv.evaluation import skillcorner as sc

MATCH = {"home_team": {"id": 1}, "away_team": {"id": 2},
         "players": [{"id": 10, "team_id": 1}, {"id": 20, "team_id": 2}, {"id": 30, "team_id": 99}]}


def _frame(frame, players, timestamp="00:00:00.00", period=1, corners=None):
    d = {"frame": frame, "timestamp": timestamp, "period": period,
         "player_data": [{"player_id": p, "x": x, "y": y, "is_detected": det} for p, x, y, det in players]}
    if corners:
        d["image_corners_projection"] = corners
    return json.dumps(d)


def test_teams_are_keyed_on_player_id_and_unknown_sides_are_other():
    assert sc._team_map(MATCH) == {10: "home", 20: "away", 30: "other"}


def test_parse_tracking_skips_frames_without_timestamp_and_keeps_the_detected_flag():
    lines = [_frame(0, [(10, 1.0, 2.0, True), (20, 3.0, 4.0, False)]),
             _frame(1, [(10, 9.0, 9.0, True)], timestamp=None),
             _frame(5, [(77, 5.0, 6.0, True)])]
    table = sc.parse_tracking(lines, sc._team_map(MATCH))
    assert list(table.columns) == sc.COLUMNS and len(table) == 3
    assert table.frame.tolist() == [0, 0, 5]
    assert table.t_s.tolist() == [0.0, 0.0, 0.5]
    assert table.team.tolist() == ["home", "away", "other"]          # unknown player id -> other
    assert table.detected.tolist() == [True, False, True]


def _table(frames, player=10, team="home", detected=True):
    return pd.DataFrame({"frame": frames, "t_s": np.asarray(frames) / sc.FPS, "period": 1, "player_id": player,
                         "team": team, "x": 0.0, "y": 0.0, "detected": detected})


def test_detected_runs_split_on_missing_frames_unless_the_allowed_step_is_widened():
    frames = list(range(0, 20)) + [21] + list(range(22, 30)) + list(range(40, 45))
    runs = sc.detected_runs(_table(frames), min_len_s=0.4)
    assert runs[["f0", "f1"]].values.tolist() == [[0, 19], [21, 29], [40, 44]]    # default: consecutive frames only
    wide = sc.detected_runs(_table(frames), min_len_s=0.4, max_hole_frames=2)
    assert wide[["f0", "f1"]].values.tolist() == [[0, 29], [40, 44]]               # one missing frame bridged
    assert sc.detected_runs(_table(frames), min_len_s=2.0, max_hole_frames=2)[["f0", "f1"]].values.tolist() == [[0, 29]]


def test_detected_runs_ignore_extrapolated_positions():
    t = pd.concat([_table(range(0, 20)), _table(range(20, 60), detected=False)])
    assert sc.detected_runs(t, min_len_s=1.0)[["f0", "f1"]].values.tolist() == [[0, 19]]


def test_load_match_parses_once_then_reads_the_parquet_cache(tmp_path):
    (tmp_path / "7_match.json").write_text(json.dumps(MATCH))
    raw = tmp_path / "7_tracking_extrapolated.jsonl"
    raw.write_text("\n".join([_frame(0, [(10, 1.0, 2.0, True)]), _frame(1, [(20, 3.0, 4.0, True)])]), encoding="utf-8")
    first = sc.load_match(tmp_path, 7)
    assert (tmp_path / "7_tracking.parquet").exists() and first.team.tolist() == ["home", "away"]
    raw.unlink()                                                      # the cache alone must now suffice
    assert len(sc.load_match(tmp_path, 7)) == 2


def _view(frame, tl, bl, br, tr):
    return {"frame": frame, **dict(zip(sc.VIEW_COLUMNS[1:], [*tl, *bl, *br, *tr]))}


def test_inside_margin_is_signed_distance_to_the_view_polygon_whatever_the_orientation():
    square = _view(0, (0, 10), (0, 0), (10, 0), (10, 10))
    flipped = _view(1, (0, 0), (0, 10), (10, 10), (10, 0))
    views = pd.DataFrame([square, flipped])
    for frame in (0, 1):
        m = sc.inside_margin_m(views, np.full(3, frame), np.array([5.0, 1.0, 12.0]), np.array([5.0, 5.0, 5.0]))
        assert m == pytest.approx([5.0, 1.0, -2.0])


def test_inside_margin_is_nan_without_a_view_for_the_frame():
    views = pd.DataFrame([_view(0, (0, 10), (0, 0), (10, 0), (10, 10))])
    assert np.isnan(sc.inside_margin_m(views, np.array([5]), np.array([5.0]), np.array([5.0]))).all()


def test_load_views_reads_corner_polygons_and_skips_frames_without_one(tmp_path):
    corners = {f"{a}_{b}": v for b, (x, y) in zip(("top_left", "bottom_left", "bottom_right", "top_right"),
                                                  ((0, 10), (0, 0), (10, 0), (10, 10))) for a, v in (("x", x), ("y", y))}
    (tmp_path / "7_tracking_extrapolated.jsonl").write_text(
        "\n".join([_frame(0, [], corners=corners), _frame(1, [], corners={"x_top_left": None}), _frame(2, [])]),
        encoding="utf-8")
    views = sc.load_views(tmp_path, 7)
    assert views.frame.tolist() == [0] and list(views.columns) == sc.VIEW_COLUMNS
    assert views.iloc[0][["x_tl", "y_tl", "x_br", "y_br"]].tolist() == [0, 10, 10, 0]


def _tracklets():
    # player A is seen in tracklets 1, 2 and 3; player B in tracklet 4
    return pd.DataFrame({"track_id": [1, 2, 3, 4], "t0": [0.0, 3.0, 20.0, 3.5], "t1": [2.0, 5.0, 22.0, 6.0]})


TRUTH = {1: "A", 2: "A", 3: "A", 4: "B"}


def test_true_adjacent_pairs_need_the_same_player_and_a_gap_within_the_limit():
    assert le.true_adjacent_pairs(_tracklets(), TRUTH, max_gap_s=6.0) == {(1, 2)}        # 2->3 gap is 15 s
    assert le.true_adjacent_pairs(_tracklets(), TRUTH, max_gap_s=20.0) == {(1, 2), (2, 3)}


def test_score_window_counts_right_and_wrong_links_and_impure_identities():
    decisions = pd.DataFrame({"pred_track_id": [1, 2, 3], "succ_track_id": [2, 4, 4], "status": ["linked", "linked", "ambiguous"]})
    identities = pd.DataFrame({"track_id": [1, 2, 3, 4], "identity_id": [1, 1, 3, 1]})   # 4 wrongly joined to A
    out = le.score_window(decisions, identities, _tracklets(), TRUTH, max_gap_s=6.0)
    assert out == {"links": 2, "links_right": 1, "gold_pairs": 1, "gold_linked": 1, "ambiguous": 1,
                   "candidates": 3, "identities": 2, "impure_identities": 1}


def test_greedy_baseline_links_every_predecessor_to_its_cheapest_successor_regardless_of_ambiguity():
    dec = pd.DataFrame({"pred_track_id": [1, 1, 2], "succ_track_id": [2, 3, 3], "cost": [0.5, 0.4, 1.0],
                        "status": ["ambiguous"] * 3})
    out = le.greedy_decisions(dec)
    assert out.loc[out.status == "linked", ["pred_track_id", "succ_track_id"]].values.tolist() == [[1, 3], [2, 3]]
    assert le.greedy_decisions(dec.iloc[0:0]).empty
