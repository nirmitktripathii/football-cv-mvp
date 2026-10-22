"""recover_swallowed: ByteTrack never outputs a track's first detection or tracks seen once; this puts them back."""
import numpy as np

from football_cv.tracking.tracker import recover_swallowed


def box(x, y=0.0, w=10.0, h=20.0):
    return [x, y, x + w, y + h]


def run(boxes, conf, ids, next_id=100, iou_min=0.3, keep=0.5):
    out, nxt = recover_swallowed(
        [np.array(b, dtype=float).reshape(-1, 4) for b in boxes], [np.array(c, dtype=float) for c in conf],
        [np.array(i, dtype=int) for i in ids], next_id, iou_min, keep,
    )
    return [o.tolist() for o in out], nxt


def test_first_detection_of_a_track_is_backfilled():
    ids, _ = run([[box(0)], [box(2)], [box(4)]], [[0.9], [0.9], [0.9]], [[-1], [5], [5]])
    assert ids == [[5], [5], [5]]


def test_backfill_follows_the_track_back_over_several_samples():
    ids, _ = run([[box(0)], [box(2)], [box(4)], [box(6)]], [[0.8]] * 4, [[-1], [-1], [-1], [5]])
    assert ids == [[5], [5], [5], [5]]


def test_no_backfill_when_boxes_do_not_overlap_but_confident_detection_gets_own_id():
    ids, nxt = run([[box(100)], [box(0)], [box(2)]], [[0.9], [0.9], [0.9]], [[-1], [5], [5]])
    assert ids == [[100], [5], [5]]
    assert nxt == 101


def test_backfill_is_one_to_one_and_prefers_the_better_overlap():
    ids, _ = run([[box(4), box(0)], [box(0)]], [[0.9, 0.9], [0.9]], [[-1, -1], [5]], next_id=100)
    assert ids[0][1] == 5 and ids[0][0] == 100


def test_low_confidence_orphan_stays_dropped():
    ids, nxt = run([[box(0)]], [[0.3]], [[-1]])
    assert ids == [[-1]] and nxt == 100


def test_orphan_duplicate_of_an_emitted_box_is_not_recovered():
    ids, _ = run([[box(0), box(0.5)]], [[0.9, 0.95]], [[5, -1]])
    assert ids == [[5, -1]]


def test_orphans_seen_in_consecutive_samples_share_one_id():
    ids, nxt = run([[box(0)], [box(2)]], [[0.9], [0.9]], [[-1], [-1]])
    assert ids[0] == ids[1] and ids[0][0] >= 100 and nxt == 101


def test_existing_ids_and_input_are_not_modified():
    ids_in = [np.array([-1]), np.array([5])]
    recover_swallowed([np.array([box(0)]), np.array([box(1)])], [np.array([0.9]), np.array([0.9])], ids_in, 100, 0.3, 0.5)
    assert ids_in[0].tolist() == [-1]


def test_empty_samples_are_handled():
    ids, _ = run([np.empty((0, 4)), [box(0)]], [[], [0.9]], [[], [-1]])
    assert ids == [[], [100]]
