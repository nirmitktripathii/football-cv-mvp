"""grass_mask must keep returning the pitch mask the original (slow) construction gave, and cover a line-split pitch."""
import cv2
import numpy as np
import pytest

from football_cv.calibration.shots import GRASS_HIGH, GRASS_LOW, MIN_COMPONENT_SHARE, grass_mask


def _reference_grass_mask(image: np.ndarray) -> np.ndarray:
    """The slow construction: convex hull over every pixel of the large green components."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    mask = np.zeros_like(green)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= MIN_COMPONENT_SHARE * areas.max())
    pts = np.column_stack(np.nonzero(np.isin(labels, keep))[::-1]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask


GRASS_BGR = (60, 150, 60)


def _pitch_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((180, 320, 3), (200, 200, 200), np.uint8)  # grey "stands"
    poly = np.array([[20, 60], [300, 40], [310, 170], [10, 175]], np.int32) + rng.integers(-8, 8, (4, 2))
    cv2.fillPoly(img, [poly], GRASS_BGR)
    for _ in range(12):  # players and lines punch holes of other colours
        x, y = rng.integers(30, 280), rng.integers(70, 160)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 6, int(y) + 14), (240, 240, 240), -1)
    cv2.line(img, (160, 50), (160, 170), (255, 255, 255), 2)
    cv2.circle(img, (30, 20), 9, GRASS_BGR, -1)  # a separate small green blob: must be ignored
    return img


@pytest.mark.parametrize("seed", range(6))
def test_grass_mask_matches_original_construction(seed):
    img = _pitch_image(seed)
    assert np.array_equal(grass_mask(img), _reference_grass_mask(img))


def test_grass_mask_is_empty_without_grass():
    img = np.full((90, 160, 3), 200, np.uint8)
    assert not grass_mask(img).any()


def test_grass_mask_covers_both_halves_when_a_line_splits_the_grass():
    # a thick halfway line cuts the grass into two components of similar size; both are pitch
    img = np.full((180, 320, 3), (200, 200, 200), np.uint8)
    cv2.fillPoly(img, [np.array([[10, 60], [310, 60], [310, 170], [10, 170]], np.int32)], GRASS_BGR)
    cv2.rectangle(img, (155, 55), (165, 175), (255, 255, 255), -1)
    mask = grass_mask(img)
    assert mask[150, 60] and mask[150, 260]   # left and right of the line
    assert not mask[20, 160]                   # stands stay out


def test_grass_mask_still_ignores_a_small_green_blob_beside_the_pitch():
    img = _pitch_image(0)
    assert not grass_mask(img)[20, 30]
