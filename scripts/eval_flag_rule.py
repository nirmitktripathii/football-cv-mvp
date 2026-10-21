"""Score the corner-flag rule: criteria F1-F3 of docs/corner_flags.md (declared before the first run).

    python scripts/eval_flag_rule.py

F1/F2 are on the PoC run (``poc_seg1rb``): which tracks the rule drops, against the labelled flag tracks, with a crop
gallery of every other dropped track for a person to look at. F3 is on the audited windows (run prefix ``FLAG_RUN``,
default ``inter_napoli_fb4f5s``): ground-truth person boxes whose matched prediction sits on a dropped track.
Writes artifacts/flag_check/eval_flag_rule.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("ROLE_RUN", os.environ.get("FLAG_RUN", "inter_napoli_fb4f5s"))
import eval_roles_detector as E  # noqa: E402
from football_cv import config as C  # noqa: E402
from football_cv.calibration.offpitch import drop_corner_flags  # noqa: E402

POC = ROOT / "data/processed/poc_seg1rb"
FLAG_TRACKS = {74, 93, 84, 108}   # labelled by looking at crops (artifacts/flag_check/candidates*.jpg)
PITCH = (105.0, 68.0)
OUT = ROOT / "artifacts/flag_check"


def gallery(run: Path, proj: pd.DataFrame, ids: list[int], path: Path) -> None:
    paths = pd.read_parquet(run / "frames.parquet").set_index("sample_id")["path"]
    tiles = []
    for t in ids:
        g = proj[proj.track_id == t].sort_values("sample_id")
        r = g.iloc[len(g) // 2]
        img = cv2.imread(str(run / paths[r.sample_id]))
        h, w = img.shape[:2]
        cx, cy = (r.bbox_x1 + r.bbox_x2) / 2, (r.bbox_y1 + r.bbox_y2) / 2
        x1, y1, x2, y2 = int(max(cx - 60, 0)), int(max(cy - 60, 0)), int(min(cx + 60, w)), int(min(cy + 60, h))
        c = cv2.resize(img[y1:y2, x1:x2], (240, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(c, f"t{t} s{r.sample_id} n{len(g)} {r.role}", (4, 16), 0, 0.5, (0, 255, 255), 2)
        tiles.append(c)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))


def poc_part() -> dict:
    proj = pd.read_parquet(POC / "tracks_projected.parquet")
    _, dropped = drop_corner_flags(proj, *PITCH)
    others = [t for t in dropped if t not in FLAG_TRACKS]
    OUT.mkdir(parents=True, exist_ok=True)
    if others:
        gallery(POC, proj, others, OUT / "poc_dropped_other.jpg")
    return {"dropped": dropped, "F1_flags_found": sorted(FLAG_TRACKS & set(dropped)),
            "F1_flags_missed": sorted(FLAG_TRACKS - set(dropped)), "F2_dropped_not_labelled_flag": others}


def windows_part() -> dict:
    out, lost_total = {}, 0
    for w in E.WINDOWS:
        run = C.resolve("data/processed") / f"{E.RUN}_{w}"
        proj = pd.read_parquet(run / "tracks_projected.parquet")
        _, dropped = drop_corner_flags(proj, *PITCH)
        gt, sample_of_k, _, after, _ = E.window_inputs(w)
        gt_people = gt[(gt["label"] == "person") & gt["role"].isin(E.CLASSES)]
        people = after[(after["object_class"] == "person") & (after["track_id"] >= 0)]
        m = E.match_rows(gt_people, people, sample_of_k)
        hit = m[m["pred_row"] >= 0]
        lost = int(after.loc[hit["pred_row"], "track_id"].isin(dropped).sum())
        out[w] = {"dropped_tracks": dropped, "gt_boxes": int(len(m)), "matched": int(len(hit)), "gt_matched_on_dropped": lost,
                  "projected_rows_dropped": int(proj["track_id"].isin(dropped).sum())}
        lost_total += lost
    return {"per_window": out, "F3_gt_matched_on_dropped": lost_total, "F3_pass": lost_total == 0}


def main() -> None:
    result = {"poc": poc_part(), "windows": windows_part()}
    result["F1_pass"] = not result["poc"]["F1_flags_missed"]
    text = json.dumps(result, indent=2)
    print(text)
    (OUT / "eval_flag_rule.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
