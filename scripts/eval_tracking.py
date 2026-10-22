"""Score tracker output on the hand-corrected evaluation windows (phase 0 metrics, ablations A-F).

    python scripts/eval_tracking.py --config configs/inter_napoli.yaml --label A_baseline
    python scripts/eval_tracking.py --config configs/inter_napoli.yaml --gt-name prefill_cvat.xml --label self_check

Ground truth is ``data/annotations/<clip>/<gt-name>`` (default ``annotations.xml``, the corrected CVAT for
video 1.1 export); predictions are ``data/processed/<clip>/<tracks-name>``. A window without ground truth
is listed under ``missing`` and left out of the pooled score; if none has ground truth, nothing is scored.
Scoring the pre-fill against the tracks it came from must give perfect scores; that is the self-check.

Writes artifacts/<clip.id>/eval_tracking_<label>.json. See football_cv.evaluation.mot for the metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from football_cv import config as C
from football_cv.evaluation import cvat, mot

log = logging.getLogger("eval_tracking")


def load_window(clip: str, gt_name: str, tracks_name: str, ball_radius: float):
    """(WindowResult, provenance) for one window, or (None, reason) when an input is missing."""
    ann, run = C.resolve("data/annotations") / clip, C.resolve("data/processed") / clip
    gt_path, tracks_path, meta_path = ann / gt_name, run / tracks_name, ann / "window.json"
    for p in (gt_path, tracks_path, meta_path):
        if not p.exists():
            return None, f"{p.relative_to(C.REPO_ROOT).as_posix()} not found"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    text = gt_path.read_text(encoding="utf-8")
    gt = cvat.parse_cvat_xml(text)
    result = mot.evaluate_window(gt, pd.read_parquet(tracks_path), meta["num_frames"], ball_radius)
    prov = {"gt_file": gt_path.relative_to(C.REPO_ROOT).as_posix(),
            "gt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "gt_labels": {str(k): int(v) for k, v in gt["label"].value_counts().items()},
            "tracks_file": tracks_path.relative_to(C.REPO_ROOT).as_posix(), "window": meta}
    return result, prov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/inter_napoli.yaml")
    ap.add_argument("--window", action="append", help="window name; repeatable (default: all in the config)")
    ap.add_argument("--gt-name", default="annotations.xml")
    ap.add_argument("--tracks-name", default="tracks.parquet")
    ap.add_argument("--label", default="current", help="run name, e.g. the ablation id")
    ap.add_argument("--ball-radius", type=float, default=mot.BALL_RADIUS_PX)
    args = ap.parse_args()
    C.setup_logging()
    cfg = C.load_config(args.config)
    windows = args.window or list(cfg["evaluation"]["windows"])
    results, provenance, missing = {}, {}, {}
    for w in windows:
        res, info = load_window(f"{cfg['clip']['id']}_{w}", args.gt_name, args.tracks_name, args.ball_radius)
        if res is None:
            missing[w] = info
            log.warning("%s skipped: %s", w, info)
        else:
            results[w], provenance[w] = res, info
    out = {"label": args.label, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "settings": {"iou_match": mot.IOU_MATCH, "ignore_ioa": mot.IGNORE_IOA, "ball_radius_px": args.ball_radius,
                        "gt_name": args.gt_name, "tracks_name": args.tracks_name},
           "missing": missing, "provenance": provenance}
    if results:
        out.update(mot.report(results, fps=float(cfg["sampling"]["fps"])))
    dest = C.artifacts_dir(cfg) / f"eval_tracking_{args.label}.json"
    C.write_json(dest, out)
    if not results:
        log.error("no window has ground truth; nothing scored (see %s)", dest)
        sys.exit(1)
    pooled = out["pooled"]
    log.info("pooled over %s: %s", list(results), json.dumps(pooled["mot"]))
    log.info("ball: %s", json.dumps(pooled["ball"]))
    log.info("written %s", dest)


if __name__ == "__main__":
    main()
