"""CVAT pre-fill export and the parser for the corrected export, on tiny synthetic tracks."""
import pandas as pd
import pytest

from football_cv.evaluation import cvat


def _row(sid, tid, x, cls="person", team="home"):
    return {"sample_id": sid, "track_id": tid, "object_class": cls, "team_id": team,
            "bbox_x1": x, "bbox_y1": 10.0, "bbox_x2": x + 8.0, "bbox_y2": 30.0}


def test_round_trip_keeps_boxes_and_closes_gaps_with_outside():
    tracks = pd.DataFrame([_row(0, 5, 0.0), _row(1, 5, 1.0), _row(3, 5, 3.0),          # gap at frame 2
                           _row(1, 7, 50.0, team="other"),
                           _row(2, -1, 90.0, cls="ball", team="ball")])
    xml = cvat.to_cvat_video_xml(tracks, num_frames=5, width=100, height=100, name="t")
    assert xml.count('outside="1"') == 4      # track 5 at 2 and 4, track 7 at 2, ball at 3
    gt = cvat.parse_cvat_xml(xml)
    assert list(gt.columns) == cvat.GT_COLUMNS
    person5 = gt[gt.gt_id == 0]
    assert person5.frame.tolist() == [0, 1, 3] and person5.x1.tolist() == [0.0, 1.0, 3.0]
    assert set(person5.role) == {"player"} and set(person5.team) == {"home"}
    other = gt[gt.gt_id == 1]
    assert other.role.tolist() == ["unknown"] and other.team.tolist() == ["unknown"]   # not guessed from kit
    ball = gt[gt.label == "ball"]
    assert ball.frame.tolist() == [2] and ball.x1.iloc[0] == 90.0


def test_boxes_are_clipped_to_the_frame_and_out_of_window_rows_dropped():
    tracks = pd.DataFrame([_row(0, 1, -4.0), _row(9, 1, 0.0)])
    gt = cvat.parse_cvat_xml(cvat.to_cvat_video_xml(tracks, num_frames=3, width=100, height=100, name="t"))
    assert gt.frame.tolist() == [0] and gt.x1.iloc[0] == 0.0


def test_images_export_is_rejected():
    with pytest.raises(ValueError, match="CVAT for video"):
        cvat.parse_cvat_xml('<annotations><image id="0" name="000000.jpg"/></annotations>')


def test_duplicate_boxes_are_rejected():
    xml = ('<annotations><track id="0" label="person">'
           '<box frame="0" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/>'
           '<box frame="0" outside="0" occluded="0" xtl="0" ytl="0" xbr="1" ybr="1"/></track></annotations>')
    with pytest.raises(ValueError, match="duplicate"):
        cvat.parse_cvat_xml(xml)


def test_label_spec_lists_every_role_and_team():
    spec = {lab["name"]: lab for lab in cvat.label_spec()}
    assert set(spec) == {"person", "ball", "ignore"}
    attrs = {a["name"]: a["values"] for a in spec["person"]["attributes"]}
    assert attrs == {"role": list(cvat.ROLES), "team": list(cvat.TEAMS)}


def test_prefill_uses_detector_roles_when_present():
    rows = [dict(_row(0, 1, 0.0, team="other"), role="referee"), dict(_row(0, 2, 40.0, team="other"), role="goalkeeper"),
            dict(_row(0, 3, 80.0, team="home"), role="player")]
    gt = cvat.parse_cvat_xml(cvat.to_cvat_video_xml(pd.DataFrame(rows), num_frames=1, width=100, height=100, name="t"))
    got = {int(r.gt_id): (r.role, r.team) for r in gt.itertuples()}
    assert got == {0: ("referee", "none"), 1: ("goalkeeper", "unknown"), 2: ("player", "home")}
