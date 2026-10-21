"""Stage 1c: pitch-position prior for goalkeepers. Criteria G1-G4 are declared in docs/role_probe.md.

    python scripts/probe_roles_position.py

Needs the cached SRITrack embeddings (scripts/probe_roles.py) and a calibration for every window: a1 from the PoC run
``poc_seg1re``, a2/a3 from the hand-clicked keyframes (``inter_napoli_fb4_calib_a{2,3}.yaml``). Writes
artifacts/role_probe/probe_position.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from football_cv import config as C
from football_cv.calibration.motion import apply_h
from football_cv.calibration.projection import H_of
from football_cv.evaluation import cvat

sys.path.insert(0, str(C.resolve("scripts")))
from probe_roles import CLASSES, SRC_PER_GT, WINDOWS, build, fit_predict, report  # noqa: E402

PITCH_L, PITCH_W = 105.0, 68.0
PEN_DEPTH = 16.5
USABLE = ("keyframe", "propagated", "propagated_unrefined")
HARD_KEEPERS = {"a2_corner_t2707": 15, "a3_open_t4800": 24}  # near-black keepers (stage 1b)
POC_RUN = "poc_seg1re"


def calibration(window: str):
    """(frame_id -> calibration row, first source frame_id) for one window."""
    proc = C.resolve("data/processed")
    frames = pd.read_parquet(proc / f"inter_napoli_fb4f5_{window}" / "frames.parquet")
    first = int(frames["frame_id"].min())
    if window.startswith("a1"):
        cal_run = pd.read_parquet(proc / POC_RUN / "frames.parquet")
        cal = pd.read_parquet(proc / POC_RUN / "frame_calibration.parquet").set_index("sample_id")
        by_frame = cal_run.set_index("frame_id")["sample_id"]
    else:
        cal = pd.read_parquet(proc / f"inter_napoli_fb4f5_{window}" / "frame_calibration.parquet").set_index("sample_id")
        by_frame = frames.set_index("frame_id")["sample_id"]
    return cal, by_frame, first


def pitch_positions(meta: pd.DataFrame) -> pd.DataFrame:
    """Add x_m, y_m (NaN when the frame has no usable calibration) from the GT boxes' bottom centres."""
    out = meta.copy()
    out["x_m"], out["y_m"] = np.nan, np.nan
    for w in WINDOWS:
        gt = cvat.parse_cvat_xml((C.resolve("data/annotations") / f"inter_napoli_fb4_{w}" / "annotations.xml")
                                 .read_text(encoding="utf-8"))
        gt = gt.set_index(["frame", "gt_id"])
        cal, by_frame, first = calibration(w)
        for k, idx in out[out["window"] == w].groupby("frame").groups.items():
            sid = by_frame.get(first + SRC_PER_GT * int(k))
            if sid is None or sid not in cal.index or cal.at[sid, "calibration_status"] not in USABLE:
                continue
            boxes = gt.loc[[(int(k), int(g)) for g in out.loc[idx, "gt_id"]]]
            foot = np.c_[(boxes["x1"] + boxes["x2"]) / 2, boxes["y2"]].astype(float)
            p = apply_h(H_of(cal.loc[sid]), foot)
            out.loc[idx, ["x_m", "y_m"]] = p
    return out


def prior_keepers(meta: pd.DataFrame, probe: np.ndarray) -> np.ndarray:
    """True where the box is the on-pitch non-referee person nearest a goal line, within the penalty-area depth."""
    fires = np.zeros(len(meta), bool)
    ok = meta["on_pitch"].to_numpy() & (probe != "referee")
    for _, idx in meta.groupby(["window", "frame"]).indices.items():
        idx = idx[ok[idx]]
        if not len(idx):
            continue
        x = meta["x_m"].to_numpy()[idx]
        for dist in (x, PITCH_L - x):
            j = int(np.argmin(dist))
            if dist[j] <= PEN_DEPTH:
                fires[idx[j]] = True
    return fires


def per_keeper_recall(meta: pd.DataFrame, pred: np.ndarray) -> dict:
    gk = (meta["role"] == "goalkeeper").to_numpy()
    out = {}
    for (w, g), sub in meta[gk].groupby(["window", "gt_id"]):
        i = sub.index.to_numpy()
        out[f"{w[:2]}_gt{g}"] = {"boxes": int(len(i)), "recall": round(float((pred[i] == "goalkeeper").mean()), 3)}
    return out


def main() -> None:
    C.setup_logging()
    meta, feats, _ = build("sritrack", "cpu", "")
    meta = pitch_positions(meta)
    meta["on_pitch"] = meta["x_m"].between(0, PITCH_L) & meta["y_m"].between(0, PITCH_W)
    y, wins = meta["role"].to_numpy(object), meta["window"].to_numpy(object)
    probe = fit_predict(feats["fc"], y, wins)
    fires = prior_keepers(meta, probe)
    p0 = np.where(fires, "goalkeeper", np.where(probe == "goalkeeper", "player", probe))
    p1 = np.where(fires | (probe == "goalkeeper"), "goalkeeper", probe)
    keep = meta["on_pitch"].to_numpy()
    matched = keep & (meta["pipeline_role"] != "").to_numpy()
    pipe = meta["pipeline_role"].to_numpy(object)
    meta["p1"] = p1
    sub = lambda a: a[keep]  # noqa: E731
    out = {"boxes": int(len(meta)), "calibrated": int(meta["x_m"].notna().sum()), "on_pitch": int(keep.sum()),
           "off_pitch_or_uncalibrated": int((~keep).sum()),
           "by_window": {w[:2]: {"boxes": int((wins == w).sum()), "on_pitch": int((keep & (wins == w)).sum())} for w in WINDOWS},
           "stage1_probe": report(sub(y), sub(probe)), "P0": report(sub(y), sub(p0)), "P1": report(sub(y), sub(p1)),
           "P1_per_window": {w[:2]: report(y[keep & (wins == w)], p1[keep & (wins == w)]) for w in WINDOWS},
           "keeper_recall_P1": per_keeper_recall(meta[keep].reset_index(drop=True), sub(p1)),
           "keeper_recall_stage1": per_keeper_recall(meta[keep].reset_index(drop=True), sub(probe)),
           "pipeline_matched_on_pitch": report(y[matched], pipe[matched]), "P1_matched_on_pitch": report(y[matched], p1[matched])}
    fp = meta[keep & (p1 == "goalkeeper") & (y != "goalkeeper")]
    out["false_keepers_by_true_role"] = fp["role"].value_counts().to_dict()
    miss = meta[keep & (y == "goalkeeper") & (p1 != "goalkeeper")]
    out["missed_keeper_x_m"] = {"min_dist_to_goal_line_median": round(float(np.minimum(miss["x_m"], PITCH_L - miss["x_m"]).median()), 1) if len(miss) else None, "n": int(len(miss))}
    C.write_json(C.resolve("artifacts") / "role_probe" / "probe_position.json", out)
    r = lambda d: (d["accuracy"], {c: (d[c]["recall"], d[c]["precision"]) for c in CLASSES})  # noqa: E731
    for k in ("stage1_probe", "P0", "P1", "pipeline_matched_on_pitch", "P1_matched_on_pitch"):
        print(k, out[k]["n"], *r(out[k]))
    for k in ("boxes", "calibrated", "on_pitch", "by_window", "keeper_recall_stage1", "keeper_recall_P1",
              "false_keepers_by_true_role", "missed_keeper_x_m"):
        print(k, out[k])


if __name__ == "__main__":
    main()
