"""Tracking metrics on tiny synthetic windows whose scores can be worked out by hand."""
import math

import numpy as np
import pandas as pd
import pytest

from football_cv.evaluation import mot
from football_cv.evaluation.cvat import GT_COLUMNS


def _gt(frame, gid, x, label="person", role="player", team="home", w=10.0):
    return {"frame": frame, "gt_id": gid, "label": label, "x1": x, "y1": 0.0, "x2": x + w, "y2": 20.0,
            "occluded": False, "role": role, "team": team}


def _pred(sid, tid, x, cls="person", team="home", w=10.0):
    return {"sample_id": sid, "track_id": tid, "object_class": cls, "team_id": team,
            "bbox_x1": x, "bbox_y1": 0.0, "bbox_x2": x + w, "bbox_y2": 20.0}


def _frames(rows, columns=None):
    return pd.DataFrame(rows, columns=columns)


def test_perfect_tracking_scores_one_whatever_the_ids():
    gt = _frames([_gt(t, g, 30.0 * g) for t in range(5) for g in (0, 1)], GT_COLUMNS)
    pred = _frames([_pred(t, 100 + g, 30.0 * g) for t in range(5) for g in (0, 1)])
    s = mot.mot_summary(mot.evaluate_window(gt, pred, num_frames=5).raw)
    assert s["HOTA"] == 1.0 and s["MOTA"] == 1.0 and s["IDF1"] == 1.0 and s["IDSW"] == 0


def test_one_identity_switch_matches_the_hand_computed_scores():
    # One player over 4 frames; the tracker uses id 1 for frames 0-1 and id 2 for frames 2-3.
    gt = _frames([_gt(t, 0, 0.0) for t in range(4)], GT_COLUMNS)
    pred = _frames([_pred(t, 1 if t < 2 else 2, 0.0) for t in range(4)])
    s = mot.mot_summary(mot.evaluate_window(gt, pred, num_frames=4).raw)
    assert s["IDSW"] == 1
    assert s["MOTA"] == pytest.approx(1 - 1 / 4)                    # 1 - (FN + FP + IDSW) / GT
    assert s["IDF1"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 2))      # best mapping keeps 2 of 4 boxes
    assert s["DetA"] == 1.0 and s["AssA"] == pytest.approx(0.5)      # each match: 2 / (2 + 2 + 0)
    assert s["HOTA"] == pytest.approx(math.sqrt(0.5), abs=1e-4)


def test_predictions_on_ignore_regions_are_dropped_unless_they_match_a_person():
    gt = _frames([_gt(0, 0, 0.0), _gt(0, 1, 50.0, label="ignore", w=40.0)], GT_COLUMNS)
    pred = _frames([_pred(0, 1, 0.0), _pred(0, 2, 60.0)])     # second box is staff inside the ignore box
    r = mot.evaluate_window(gt, pred, num_frames=1)
    assert r.counts["pred_boxes_dropped_on_ignore"] == 1 and mot.mot_summary(r.raw)["CLR_FP"] == 0

    gt2 = _frames([_gt(0, 0, 0.0), _gt(0, 1, 0.0, label="ignore", w=40.0)], GT_COLUMNS)
    r2 = mot.evaluate_window(gt2, _frames([_pred(0, 1, 0.0)]), num_frames=1)
    assert r2.counts["pred_boxes_dropped_on_ignore"] == 0 and mot.mot_summary(r2.raw)["CLR_TP"] == 1


def test_team_goalkeeper_and_referee_labels_are_scored_on_matched_boxes():
    gt = _frames([_gt(0, 0, 0.0, team="home"), _gt(0, 1, 30.0, team="away"),
                  _gt(0, 2, 60.0, role="goalkeeper", team="away"), _gt(0, 3, 90.0, role="referee", team="none"),
                  _gt(0, 4, 120.0, team="home")], GT_COLUMNS)
    pred = _frames([_pred(0, 1, 0.0, team="home"), _pred(0, 2, 30.0, team="home"),
                    _pred(0, 3, 60.0, team="other"), _pred(0, 4, 90.0, team="away"),
                    _pred(0, 5, 300.0, team="home")])               # gt 4 is missed, pred 5 is a false positive
    lab = mot.label_metrics(mot.evaluate_window(gt, pred, num_frames=1).pairs)
    assert lab["matched_boxes"] == 4
    assert lab["team_outfield"]["coverage"]["rate"] == 1.0
    assert lab["team_outfield"]["accuracy_per_box"]["rate"] == 0.5
    assert lab["goalkeeper_affiliation"]["coverage"]["rate"] == 0.0     # "other" is no call
    assert lab["referee_given_team"]["rate"] == 1.0
    assert lab["role_accuracy"] is None                                 # the tracker has no role column yet


def test_ball_recall_precision_and_longest_miss():
    gt = _frames([_gt(t, 9, 100.0, label="ball", role="", team="", w=4.0) for t in range(6)], GT_COLUMNS)
    xs = [100.0, 103.0, 150.0, 150.0, 150.0, 100.0, 100.0]          # frames 2-4 far off, frame 6 has no ball
    pred = _frames([_pred(t, -1, x, cls="ball", team="ball", w=4.0) for t, x in enumerate(xs)])
    b = mot.ball_summary(mot.evaluate_window(gt, pred, num_frames=7).ball, fps=5.0)
    assert (b["hits"], b["recall"], b["precision"]) == (3, 0.5, round(3 / 7, 4))
    assert b["max_missed_run_frames"] == 3 and b["max_missed_run_s"] == 0.6
    assert b["centre_error_px_median"] == 0.0


def test_wilson_interval():
    assert mot.wilson(0, 0) is None
    assert mot.wilson(5, 10) == pytest.approx([0.2366, 0.7634], abs=1e-4)
    assert mot.wilson(0, 10)[0] == 0.0


def test_report_pools_windows_by_summing_counts():
    gt = _frames([_gt(t, 0, 0.0) for t in range(3)], GT_COLUMNS)
    pred = _frames([_pred(t, 1, 0.0) for t in range(3)])
    rep = mot.report({"a": mot.evaluate_window(gt, pred, 3), "b": mot.evaluate_window(gt, pred.iloc[:1], 3)}, fps=5.0)
    assert rep["pooled"]["mot"]["CLR_TP"] == 3 + 1 and rep["pooled"]["counts"]["gt_boxes"] == 6
    assert rep["windows"]["b"]["mot"]["CLR_FN"] == 2
    assert np.isfinite(rep["pooled"]["mot"]["HOTA"])
