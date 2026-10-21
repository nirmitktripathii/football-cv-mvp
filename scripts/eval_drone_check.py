"""Does a ball checkpoint still fire on the hanging camera rig, and does it keep the confirmed real balls?

    python scripts/eval_drone_check.py --ckpt data/models/yolox_m_football4_drone/best_ckpt.pth

Criteria are fixed here, before any stage-3 result exists (``docs/finetune_4class.md``, "Drone hard negatives"):

  D1  held-out rig samples (the episode that contributed no training patch, user-confirmed as the rig): ball detections within
      ``RIG_RADIUS_PX`` of the rig at confidence >= ``BALL_FLOOR`` in at most ``D1_MAX_SAMPLES`` of them, and none
      at confidence >= ``D1_MAX_CONF``.
  D2  user-confirmed real balls: a ball detection >= ``BALL_FLOOR`` within ``REAL_RADIUS_PX`` of the position the
      stage-1 pipeline tracked, in at least ``D2_MIN_SAMPLES`` of them.

Reference positions are the stage-1 ball rows of ``--run`` (poc_seg1rb): the rig for D1, the confirmed balls for D2.
In-sample rig samples (mined into the training set) are printed but never counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_cv.detection.yolox_backend import CLASS_NAMES, YoloxPredictor  # noqa: E402

BALL_FLOOR = 0.10                       # configs/default.yaml detection.conf_ball
RIG_RADIUS_PX, REAL_RADIUS_PX = 20.0, 10.0
D1_MAX_SAMPLES, D1_MAX_CONF, D2_MIN_SAMPLES = 2, 0.30, 5
HELD_OUT_RIG = [394, 396, 397, 398, 399, 400, 401, 410, 411]   # episodes with no mined patch
REAL_BALLS = [78, 86, 171, 341, 467, 849]                                 # confirmed real by the user
IN_SAMPLE_RIG = [76, 230, 232, 233, 237, 238, 240, 241, 242, 305, 311, 312, 313, 314, 315]
BALL = [k for k, v in CLASS_NAMES.items() if v == "ball"][0]


def centre(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def nearest_ball(pred, ref_xy: np.ndarray, radius: float) -> float:
    """Highest confidence of a ball detection >= BALL_FLOOR within ``radius`` of ``ref_xy`` (0 when none)."""
    xyxy, conf, cls = pred
    best = 0.0
    for b, c, k in zip(xyxy, conf, cls):
        if k == BALL and c >= BALL_FLOOR and np.linalg.norm(centre(b) - ref_xy) <= radius:
            best = max(best, float(c))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--run", default=ROOT / "data/processed/poc_seg1rb", type=Path)
    ap.add_argument("--yolox-repo", default=ROOT / "data/external/YOLOX", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    frames = pd.read_parquet(args.run / "frames.parquet").set_index("sample_id")
    tracks = pd.read_parquet(args.run / "tracks_projected.parquet")
    ref = tracks[tracks.object_class == "ball"].drop_duplicates("sample_id").set_index("sample_id")
    pred = YoloxPredictor(args.ckpt, args.yolox_repo, device=0, test_size=(704, 1280), conf_floor=0.01)
    result = {"checkpoint": pred.provenance, "run": str(args.run)}
    groups = {"held_out_rig": (HELD_OUT_RIG, RIG_RADIUS_PX), "in_sample_rig": (IN_SAMPLE_RIG, RIG_RADIUS_PX),
              "real_balls": (REAL_BALLS, REAL_RADIUS_PX)}
    for name, (ids, radius) in groups.items():
        ids = [i for i in ids if i in ref.index]
        confs = {}
        for i in ids:
            r = ref.loc[i]
            p = pred.predict([str(args.run / frames.loc[i, "path"])])[0]
            confs[i] = round(nearest_ball(p, centre((r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)), radius), 3)
        result[name] = {"samples": confs, "detected_at_floor": sum(c >= BALL_FLOOR for c in confs.values()),
                        "n": len(ids), "max_conf": max(confs.values(), default=0.0)}
    d1, d2 = result["held_out_rig"], result["real_balls"]
    result["D1_pass"] = d1["detected_at_floor"] <= D1_MAX_SAMPLES and d1["max_conf"] < D1_MAX_CONF
    result["D2_pass"] = d2["detected_at_floor"] >= D2_MIN_SAMPLES
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
