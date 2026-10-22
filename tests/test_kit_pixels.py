"""Kit pixels of a person crop: pitch grass is dropped, a lime referee shirt (hue inside the grass range) is not."""
import cv2
import numpy as np

from football_cv.tracking.appearance import band_colours
from football_cv.tracking.teams import torso_colour

GRASS_BGR = (70, 140, 60)        # hue ~ 55, value ~ 140: broadcast grass
LIME_BGR = (60, 255, 190)        # hue ~ 32, saturated, value 255: referee shirt
BLACK_BGR = (20, 20, 20)
BOX = (40, 20, 80, 120)          # x1, y1, x2, y2


def _person(shirt_bgr):
    img = np.full((160, 120, 3), GRASS_BGR, np.uint8)
    x1, y1, x2, y2 = BOX
    img[y1:y1 + 40, x1:x2] = shirt_bgr
    img[y1 + 40:y1 + 60, x1:x2] = BLACK_BGR
    img[y1 + 60:y2] = (230, 230, 230)
    return img


def _lab(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def test_lime_shirt_is_not_mistaken_for_grass():
    assert torso_colour(_person(LIME_BGR), BOX) is not None


def test_lime_shirt_colour_is_read_from_the_shirt_not_the_shorts():
    got = torso_colour(_person(LIME_BGR), BOX)
    assert abs(got[2] - _lab(LIME_BGR)[2]) < 25      # yellow b*, far from the black shorts' ~128


def test_grass_only_crop_still_has_no_kit_pixels():
    assert torso_colour(np.full((160, 120, 3), GRASS_BGR, np.uint8), BOX) is None


def test_band_descriptor_keeps_the_lime_shirt_band():
    v = band_colours(_person(LIME_BGR), BOX)
    assert v is not None and abs(v[2] - _lab(LIME_BGR)[2]) < 25
