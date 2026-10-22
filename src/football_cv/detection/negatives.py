"""Hard-negative patches for the ball detector: frame regions around a known false ball (e.g. the drone).

A patch is a fixed-size window centred on the false detection and holding no tracked person, so it can be
added to the training set with no boxes at all. Whole frames cannot be used that way, because the people in
them would become unlabelled "background".
"""
from __future__ import annotations

import pandas as pd


def patch_window(centre: tuple[float, float], size: int, frame: tuple[int, int]) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) of a size x size window centred on ``centre``, shifted to stay inside ``frame`` (w, h)."""
    fw, fh = frame
    if size > fw or size > fh:
        raise ValueError(f"patch size {size} does not fit a {fw}x{fh} frame")
    x1 = int(min(max(round(centre[0] - size / 2), 0), fw - size))
    y1 = int(min(max(round(centre[1] - size / 2), 0), fh - size))
    return x1, y1, x1 + size, y1 + size


def _overlaps(window: tuple[int, int, int, int], boxes: pd.DataFrame) -> bool:
    x1, y1, x2, y2 = window
    return bool(((boxes.bbox_x1 < x2) & (boxes.bbox_x2 > x1) & (boxes.bbox_y1 < y2) & (boxes.bbox_y2 > y1)).any())


def mine_negative_patches(tracks: pd.DataFrame, sample_ids, size: int, frame: tuple[int, int]):
    """([{sample_id, window, ball_box}], {sample_id: reason}) for the ball rows of ``sample_ids``.

    A sample is skipped, with the reason recorded, when it has no ball row or when a person overlaps the patch.
    """
    patches, skipped = [], {}
    for sid in sample_ids:
        rows = tracks[tracks.sample_id == sid]
        balls = rows[rows.object_class == "ball"]
        if balls.empty:
            skipped[sid] = "no_ball_detection"
            continue
        people = rows[rows.object_class == "person"]
        kept = 0
        for b in balls.itertuples():
            window = patch_window(((b.bbox_x1 + b.bbox_x2) / 2, (b.bbox_y1 + b.bbox_y2) / 2), size, frame)
            if _overlaps(window, people):
                continue
            patches.append({"sample_id": int(sid), "window": window,
                            "ball_box": (b.bbox_x1, b.bbox_y1, b.bbox_x2, b.bbox_y2)})
            kept += 1
        if not kept:
            skipped[sid] = "person_in_patch"
    return patches, skipped


def mine_person_box_patches(tracks: pd.DataFrame, sample_ids, rig_track_ids, size: int, frame: tuple[int, int]):
    """([{sample_id, window, rig_box, rig_boxes}], {sample_id: reason}) for person rows that are known non-people.

    ``rig_track_ids`` are tracks a person has confirmed as the camera rig (detected as a person). The patch is centred on the
    largest rig box of the sample and kept as box-less background. It is skipped, with the reason recorded, when the
    sample has no rig box or when a person that is not a rig track overlaps it. Ball rows do not block a patch.
    """
    rig_ids = set(int(t) for t in rig_track_ids)
    patches, skipped = [], {}
    for sid in sample_ids:
        people = tracks[(tracks.sample_id == sid) & (tracks.object_class == "person")]
        rig = people[people.track_id.isin(rig_ids)]
        if rig.empty:
            skipped[sid] = "no_rig_box"
            continue
        area = (rig.bbox_x2 - rig.bbox_x1) * (rig.bbox_y2 - rig.bbox_y1)
        anchor = rig.loc[area.idxmax()]
        window = patch_window(((anchor.bbox_x1 + anchor.bbox_x2) / 2, (anchor.bbox_y1 + anchor.bbox_y2) / 2), size, frame)
        if _overlaps(window, people[~people.track_id.isin(rig_ids)]):
            skipped[sid] = "person_in_patch"
            continue
        inside = rig[(rig.bbox_x1 < window[2]) & (rig.bbox_x2 > window[0]) & (rig.bbox_y1 < window[3]) & (rig.bbox_y2 > window[1])]
        patches.append({"sample_id": int(sid), "window": window,
                        "rig_box": (anchor.bbox_x1, anchor.bbox_y1, anchor.bbox_x2, anchor.bbox_y2),
                        "rig_boxes": [(r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2) for r in inside.itertuples()]})
    return patches, skipped


def load_negatives(dirs, repeat: int = 1):
    """[(image path, training name)] for every patch listed in the ``manifest.json`` of each mined set.

    ``repeat`` copies each patch under distinct names, which is how a few hard negatives are over-weighted.
    """
    import json
    from pathlib import Path

    items = []
    for si, d in enumerate(dirs):
        d = Path(d)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for p in manifest["patches"]:
            path = d / "images" / p["file"]
            if not path.is_file():
                raise FileNotFoundError(f"{path} is listed in {d / 'manifest.json'} but missing")
            items += [(path, f"neg_{manifest['label']}_s{si}_{Path(p['file']).stem}_r{r}.jpg") for r in range(repeat)]
    return items
