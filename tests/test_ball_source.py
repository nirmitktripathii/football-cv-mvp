import pandas as pd

from football_cv.detection.detector import DETECTION_COLUMNS, merge_ball_source


def _row(sample, det, cls, conf, model):
    return {"sample_id": sample, "frame_id": sample * 5, "timestamp_s": sample / 5, "det_id": det,
            "object_class": cls, "role": "player" if cls == "person" else "", "confidence": conf,
            "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 8.0, "bbox_y2": 8.0, "model": model}


PEOPLE = pd.DataFrame([_row(0, 0, "person", 0.9, "main"), _row(0, 1, "ball", 0.3, "main"),
                       _row(1, 0, "person", 0.8, "main")], columns=DETECTION_COLUMNS)
BALLS = pd.DataFrame([_row(0, 0, "person", 0.7, "ball"), _row(0, 1, "ball", 0.6, "ball"),
                      _row(2, 0, "ball", 0.5, "ball")], columns=DETECTION_COLUMNS)


def test_people_come_from_the_main_detector_and_balls_from_the_ball_detector():
    out = merge_ball_source(PEOPLE, BALLS)
    assert set(out[out.object_class == "person"]["model"]) == {"main"}
    assert set(out[out.object_class == "ball"]["model"]) == {"ball"}
    assert len(out) == 4   # 2 people + 2 balls; the main ball and the ball detector's person are dropped


def test_det_ids_are_unique_within_each_sample():
    out = merge_ball_source(PEOPLE, BALLS)
    assert not out.duplicated(["sample_id", "det_id"]).any()
    assert out.groupby("sample_id")["det_id"].apply(lambda s: sorted(s) == list(range(len(s)))).all()


def test_a_sample_with_only_a_ball_is_kept_and_rows_are_ordered():
    out = merge_ball_source(PEOPLE, BALLS)
    assert 2 in set(out["sample_id"])
    assert list(out.columns) == DETECTION_COLUMNS
    assert out["sample_id"].is_monotonic_increasing
