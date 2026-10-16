import cv2
import numpy as np
import pytest

from football_cv.calibration import homography as hg
from football_cv.calibration.motion import apply_h, plausible
from football_cv.tracking.tracker import stabilize_boxes


def _camera_h():
    """A broadcast-like pitch -> image homography: near touchline at the bottom, perspective foreshortening."""
    pitch = np.array([[20, 0], [85, 0], [80, 68], [25, 68]], dtype=float)
    image = np.array([[100, 700], [1180, 700], [1000, 180], [280, 180]], dtype=float)
    H_p2i, _ = cv2.findHomography(pitch, image, 0)
    return H_p2i / H_p2i[2, 2]


def test_landmarks_are_inside_reference_pitch():
    marks = hg.landmarks()
    xs = np.array([p[0] for p in marks.values()])
    ys = np.array([p[1] for p in marks.values()])
    assert xs.min() == 0 and xs.max() == 105
    assert ys.min() == 0 and ys.max() == 68
    assert marks["pen_r_corner_top"] == pytest.approx((88.5, 54.16))


def test_fit_recovers_known_homography_exactly():
    H_p2i = _camera_h()
    names = ["corner_bl", "halfway_bottom", "circle_top", "pen_l_corner_top", "pen_l_corner_bottom", "centre_spot"]
    pitch = np.array([hg.landmarks()[n] for n in names])
    image = apply_h(H_p2i, pitch)
    H, rms = hg.fit_from_correspondences(image, pitch)
    assert rms < 1e-4
    np.testing.assert_allclose(apply_h(H, image), pitch, atol=1e-4)


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        hg.fit_from_correspondences(np.zeros((3, 2)), np.zeros((3, 2)))


def test_validate_accepts_broadcast_view_and_rejects_mirror():
    H_i2p = np.linalg.inv(_camera_h())
    assert hg.validate(H_i2p, (720, 1280))[0]
    mirror = np.diag([-1.0, 1.0, 1.0])
    mirror[0, 2] = 105.0
    ok, why = hg.validate(mirror @ H_i2p, (720, 1280))
    assert not ok and "mirrored" in why


def test_frame_homography_composes_camera_motion():
    H_k = np.linalg.inv(_camera_h())
    pan = np.array([[1.0, 0, -40], [0, 1, 5], [0, 0, 1]])  # frame t -> keyframe pixels
    H_t = hg.frame_homography(H_k, np.eye(3), pan)
    pt_t = np.array([[640.0, 500.0]])
    np.testing.assert_allclose(apply_h(H_t, pt_t), apply_h(H_k, apply_h(pan, pt_t)), atol=1e-9)


def test_plausible_rejects_large_zoom_and_flip_but_accepts_small_pan():
    assert plausible(np.array([[1.0, 0, 12], [0, 1, -3], [0, 0, 1]]))[0]
    assert not plausible(np.diag([2.0, 2.0, 1.0]))[0]
    assert not plausible(np.diag([-1.0, 1.0, 1.0]))[0]


def test_stabilize_boxes_translates_with_pan():
    boxes = np.array([[100.0, 200.0, 120.0, 260.0]])
    G = np.array([[1.0, 0, 30], [0, 1, -10], [0, 0, 1]])
    np.testing.assert_allclose(stabilize_boxes(boxes, G), [[130, 190, 150, 250]])
    assert stabilize_boxes(np.empty((0, 4)), G).shape == (0, 4)


def test_overlay_draws_pitch_lines_without_mutating_input():
    H_i2p = np.linalg.inv(_camera_h())
    img = np.zeros((720, 1280, 3), np.uint8)
    out = hg.draw_pitch_overlay(img, H_i2p)
    assert out[..., 2].max() == 255 and img.max() == 0
