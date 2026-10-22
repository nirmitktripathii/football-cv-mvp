"""YOLOX backend: coordinate handling and row building on synthetic arrays; the real model only if it is present."""
import numpy as np
import pytest

from football_cv.config import REPO_ROOT
from football_cv.detection import yolox_backend as yb
from football_cv.detection.detector import DETECTION_COLUMNS, class_roles, frame_rows

REC = {"sample_id": 7, "frame_id": 700, "timestamp_s": 28.0}
ROLES = class_roles(yb.CLASS_NAMES)


def test_letterbox_ratio_is_the_smaller_side_ratio():
    # 720x1280 frame into a 704x1280 input: height limits the scale
    assert yb.letterbox_ratio(720, 1280, (704, 1280)) == pytest.approx(704 / 720)


def test_boxes_return_to_original_pixels_and_score_is_objectness_times_class():
    # Arrange: one detection in network pixels, ratio 0.5
    out = np.array([[50.0, 100.0, 150.0, 300.0, 0.8, 0.5, 2.0]])
    # Act
    xyxy, conf, cls = yb.to_original_pixels(out, ratio=0.5, height=720, width=1280)
    # Assert
    assert xyxy.tolist() == [[100.0, 200.0, 300.0, 600.0]]
    assert conf == pytest.approx([0.4]) and cls.tolist() == [2]


def test_boxes_are_clipped_to_the_frame():
    out = np.array([[-10.0, -5.0, 700.0, 400.0, 1.0, 1.0, 0.0]])
    xyxy, _, _ = yb.to_original_pixels(out, ratio=0.5, height=720, width=1280)
    assert xyxy.tolist() == [[0.0, 0.0, 1280.0, 720.0]]


@pytest.mark.parametrize("empty", [None, np.zeros((0, 7))])
def test_no_detections_gives_empty_arrays(empty):
    xyxy, conf, cls = yb.to_original_pixels(empty, ratio=1.0, height=720, width=1280)
    assert xyxy.shape == (0, 4) and conf.shape == (0,) and cls.shape == (0,)


def test_class_names_match_the_trained_label_order():
    # the training dataset manifest lists the classes alphabetically; the head's output index follows it
    assert [yb.CLASS_NAMES[i] for i in range(4)] == ["ball", "goalkeeper", "player", "referee"]
    assert ROLES[3] == ("person", "referee") and ROLES[0] == ("ball", "")


def test_rows_apply_a_separate_confidence_floor_per_object_class():
    xyxy = np.array([[0, 0, 10, 20], [5, 5, 9, 9], [30, 30, 50, 90]], dtype=float)
    conf = np.array([0.30, 0.12, 0.20])    # person floor 0.25, ball floor 0.10
    cls = np.array([2, 0, 3])
    rows = frame_rows(REC, xyxy, conf, cls, ROLES, conf_person=0.25, conf_ball=0.10, provenance="m")
    assert [(r["object_class"], r["role"], r["det_id"]) for r in rows] == [("person", "player", 0), ("ball", "", 1)]
    assert set(rows[0]) == set(DETECTION_COLUMNS) and rows[0]["sample_id"] == 7


def test_non_finite_boxes_are_refused():
    with pytest.raises(RuntimeError, match="Non-finite"):
        frame_rows({**REC, "path": "x.png"}, np.array([[np.nan, 0, 1, 1]]), np.array([0.9]), np.array([2]),
                   ROLES, 0.25, 0.10, "m")


CHECKPOINT = REPO_ROOT / "data" / "models" / "yolox_m_football4" / "best_ckpt.pth"
YOLOX_DIR = REPO_ROOT / "data" / "external" / "YOLOX"


@pytest.mark.skipif(not (CHECKPOINT.exists() and YOLOX_DIR.exists()), reason="checkpoint or YOLOX checkout not present")
def test_real_model_loads_and_detects_on_a_sampled_frame():
    frames = sorted((REPO_ROOT / "data" / "processed").glob("inter_napoli_fb4_a1_open_t600/frames/*.*"))
    if not frames:
        pytest.skip("no sampled frames")
    predictor = yb.YoloxPredictor(CHECKPOINT, YOLOX_DIR, device="cpu", test_size=(704, 1280))
    (xyxy, conf, cls), = predictor.predict([str(frames[0])])
    assert xyxy.shape[1:] == (4,) and len(xyxy) == len(conf) == len(cls)
    assert np.isfinite(xyxy).all() and set(cls.tolist()) <= set(yb.CLASS_NAMES)
