import numpy as np
import pandas as pd
import pytest

from football_cv.detection.negatives import mine_negative_patches, patch_window


def test_patch_window_is_centred_and_clamped_to_the_frame():
    assert patch_window((640, 360), 320, (1280, 720)) == (480, 200, 800, 520)
    assert patch_window((10, 10), 320, (1280, 720)) == (0, 0, 320, 320)
    assert patch_window((1275, 715), 320, (1280, 720)) == (960, 400, 1280, 720)


def test_patch_window_rejects_a_patch_larger_than_the_frame():
    with pytest.raises(ValueError):
        patch_window((10, 10), 800, (640, 480))


def _tracks(rows):
    return pd.DataFrame(rows, columns=["sample_id", "object_class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])


def test_a_patch_holding_a_person_is_skipped_and_a_clean_one_kept():
    tracks = _tracks([
        (1, "ball", 630, 350, 650, 370), (1, "person", 700, 300, 740, 400),
        (2, "ball", 630, 350, 650, 370), (2, "person", 100, 100, 140, 200),
    ])
    patches, skipped = mine_negative_patches(tracks, [1, 2], 320, (1280, 720))
    assert [p["sample_id"] for p in patches] == [2]
    assert patches[0]["window"] == (480, 200, 800, 520)
    assert skipped == {1: "person_in_patch"}


def test_a_sample_without_a_ball_detection_is_reported_not_guessed():
    patches, skipped = mine_negative_patches(_tracks([(3, "person", 0, 0, 10, 10)]), [3], 320, (1280, 720))
    assert patches == [] and skipped == {3: "no_ball_detection"}


def test_several_ball_rows_in_one_sample_yield_one_patch_per_row():
    tracks = _tracks([(4, "ball", 100, 100, 120, 120), (4, "ball", 900, 500, 920, 520)])
    patches, _ = mine_negative_patches(tracks, [4], 320, (1280, 720))
    assert len(patches) == 2


def test_load_negatives_repeats_each_patch_under_distinct_names(tmp_path):
    import json

    from football_cv.detection.negatives import load_negatives

    d = tmp_path / "set"
    (d / "images").mkdir(parents=True)
    (d / "images" / "a.jpg").write_bytes(b"x")
    (d / "manifest.json").write_text(json.dumps({"label": "drone", "patches": [{"file": "a.jpg"}]}))
    items = load_negatives([d], repeat=3)
    assert [n for _, n in items] == ["neg_drone_s0_a_r0.jpg", "neg_drone_s0_a_r1.jpg", "neg_drone_s0_a_r2.jpg"]
    assert all(p == d / "images" / "a.jpg" for p, _ in items)


def test_load_negatives_rejects_a_manifest_naming_a_missing_image(tmp_path):
    import json

    from football_cv.detection.negatives import load_negatives

    (tmp_path / "images").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"label": "drone", "patches": [{"file": "gone.jpg"}]}))
    with pytest.raises(FileNotFoundError):
        load_negatives([tmp_path], repeat=1)


def _people(rows):
    return pd.DataFrame(rows, columns=["sample_id", "track_id", "object_class", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])


def test_a_rig_person_box_gets_a_patch_centred_on_it_and_other_rig_boxes_do_not_block_it():
    from football_cv.detection.negatives import mine_person_box_patches

    tracks = _people([(1, 83, "person", 630, 350, 650, 390), (1, 87, "person", 632, 345, 648, 372)])
    patches, skipped = mine_person_box_patches(tracks, [1], [83, 87], 320, (1280, 720))
    assert skipped == {}
    assert len(patches) == 1 and patches[0]["window"] == (480, 210, 800, 530)
    assert patches[0]["rig_box"] == (630, 350, 650, 390)          # the larger rig box is the anchor
    assert len(patches[0]["rig_boxes"]) == 2                        # both rig boxes are inside the patch


def test_a_real_person_in_the_patch_skips_it_even_when_a_rig_box_is_there_too():
    from football_cv.detection.negatives import mine_person_box_patches

    tracks = _people([(2, 83, "person", 630, 350, 650, 390), (2, 5, "person", 700, 300, 740, 400)])
    patches, skipped = mine_person_box_patches(tracks, [2], [83], 320, (1280, 720))
    assert patches == [] and skipped == {2: "person_in_patch"}


def test_a_sample_without_a_rig_track_is_reported_not_guessed():
    from football_cv.detection.negatives import mine_person_box_patches

    tracks = _people([(3, 5, "person", 0, 0, 10, 10)])
    patches, skipped = mine_person_box_patches(tracks, [3], [83], 320, (1280, 720))
    assert patches == [] and skipped == {3: "no_rig_box"}


def test_a_ball_row_in_the_patch_does_not_block_a_rig_patch():
    from football_cv.detection.negatives import mine_person_box_patches

    tracks = _people([(4, 83, "person", 630, 350, 650, 390), (4, -1, "ball", 640, 345, 650, 355)])
    patches, _ = mine_person_box_patches(tracks, [4], [83], 320, (1280, 720))
    assert len(patches) == 1
