"""Does a checkpoint still detect the hanging camera rig as a person?

    python scripts/eval_rig_person_check.py --ckpt data/models/yolox_m_football4_rig/best_ckpt.pth

Criteria are fixed here, before any stage-3b result exists (``docs/finetune_4class.md``, "Rig as a person"):

  R1  a2 evaluation window (rig on a pole, no patch from it in training): samples where a person-class detection
      (player, goalkeeper or referee) at confidence >= ``PERSON_FLOOR`` overlaps a stage-1 rig box with IoU >=
      ``RIG_IOU``: at most ``R1_MAX_SAMPLES`` of the a2 rig samples, and none at confidence >= ``MAX_CONF``.
  R2  PoC samples 396-419, the rig episode kept out of training: the same test, at most ``R2_MAX_SAMPLES``.

Reference rig boxes are the stage-1 person tracks that a person confirmed as the rig by looking at them
(a2 run tracks 22, 24, 27; PoC tracks 156, 160, 164). The in-sample episode (PoC samples 234-299, mined into the
training set) is printed but never counted. R3 (real people kept) is scored through the pipeline, not here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_cv.detection.yolox_backend import CLASS_NAMES, YoloxPredictor  # noqa: E402

PERSON_FLOOR = 0.25                    # configs/default.yaml detection.conf_person
RIG_IOU = 0.30
MAX_CONF = 0.50
R1_MAX_SAMPLES, R2_MAX_SAMPLES = 3, 5
PERSON_CLASSES = {k for k, v in CLASS_NAMES.items() if v != "ball"}
GROUPS = {   # name: (run dir, rig track ids, sample range or None)
    "a2_rig": ("data/processed/inter_napoli_fb4f5p_a2_corner_t2707", [22, 24, 27], None),
    "poc_held_out_rig": ("data/processed/poc_seg1rb", [156, 160, 164], (396, 419)),
    "poc_in_sample_rig": ("data/processed/poc_seg1rb", [83, 87], (234, 299)),   # printed, never counted
}


def iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def best_person_conf(pred, rig_boxes) -> float:
    """Highest confidence of a person-class detection >= PERSON_FLOOR that overlaps a rig box (0 when none)."""
    xyxy, conf, cls = pred
    best = 0.0
    for b, c, k in zip(xyxy, conf, cls):
        if int(k) in PERSON_CLASSES and c >= PERSON_FLOOR and any(iou(b, r) >= RIG_IOU for r in rig_boxes):
            best = max(best, float(c))
    return best


def rig_boxes_by_sample(run: Path, track_ids, span) -> dict[int, list[tuple]]:
    pre = run / "tracks_pre_roles.parquet"
    tracks = pd.read_parquet(pre if pre.exists() else run / "tracks.parquet")
    rows = tracks[(tracks.object_class == "person") & tracks.track_id.isin(track_ids)]
    if span:
        rows = rows[(rows.sample_id >= span[0]) & (rows.sample_id <= span[1])]
    out: dict[int, list[tuple]] = {}
    for r in rows.itertuples():
        out.setdefault(int(r.sample_id), []).append((r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--yolox-repo", default=ROOT / "data/external/YOLOX", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    pred = YoloxPredictor(args.ckpt, args.yolox_repo, device=0, test_size=(704, 1280), conf_floor=0.01)
    result = {"checkpoint": pred.provenance}
    for name, (run, track_ids, span) in GROUPS.items():
        run = ROOT / run
        frames = pd.read_parquet(run / "frames.parquet").set_index("sample_id")
        boxes = rig_boxes_by_sample(run, track_ids, span)
        confs = {sid: round(best_person_conf(pred.predict([str(run / frames.loc[sid, "path"])])[0], rb), 3)
                 for sid, rb in sorted(boxes.items())}
        result[name] = {"n": len(confs), "detected_at_floor": sum(c > 0 for c in confs.values()),
                        "max_conf": max(confs.values(), default=0.0), "samples": confs}
    r1, r2 = result["a2_rig"], result["poc_held_out_rig"]
    result["R1_pass"] = r1["detected_at_floor"] <= R1_MAX_SAMPLES and r1["max_conf"] < MAX_CONF
    result["R2_pass"] = r2["detected_at_floor"] <= R2_MAX_SAMPLES and r2["max_conf"] < MAX_CONF
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
