"""Score the roles stage on detector boxes: criteria H1-H6 of docs/role_probe.md stage 2 (declared before the run).

    python scripts/eval_roles_detector.py

Reads data/processed/inter_napoli_fb4f5p_<window> (relabelled tracks; tracks_pre_roles.parquet holds the f5r tracks) and
the audited ground truth. Writes artifacts/role_probe/eval_detector.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_cv import config as C
from football_cv.evaluation import cvat, mot
from football_cv.tracking import role_probe

WINDOWS = ("a1_open_t600", "a2_corner_t2707", "a3_open_t4800")
RUN = "inter_napoli_fb4f5p"
SRC_PER_GT = 5  # 25 fps source, 5 fps annotation
PITCH = (105.0, 68.0)
CLASSES = role_probe.CLASSES
SMALL_PX = 8


def window_inputs(window: str):
    run = C.resolve("data/processed") / f"{RUN}_{window}"
    frames = pd.read_parquet(run / "frames.parquet")
    first = int(frames["frame_id"].min())
    by_frame = frames.set_index("frame_id")["sample_id"]
    sample_of_k = {(int(f) - first) // SRC_PER_GT: int(by_frame[f]) for f in frames["frame_id"]
                   if (int(f) - first) % SRC_PER_GT == 0}
    gt = cvat.parse_cvat_xml((C.resolve("data/annotations") / f"inter_napoli_fb4_{window}" / "annotations.xml")
                             .read_text(encoding="utf-8"))
    after = pd.read_parquet(run / "tracks.parquet")
    before = pd.read_parquet(run / "tracks_pre_roles.parquet")
    calib = pd.read_parquet(run / "frame_calibration.parquet")
    return gt, sample_of_k, before, after, calib


def match_rows(gt_people: pd.DataFrame, people: pd.DataFrame, sample_of_k: dict) -> pd.DataFrame:
    """Hungarian matches per GT frame; one row per GT box with the matched prediction's row label (-1 if none)."""
    by_sample = dict(tuple(people.groupby("sample_id")))
    rows = []
    for k, g in gt_people.groupby("frame"):
        p = by_sample.get(sample_of_k.get(int(k)))
        hit = {}
        if p is not None and len(p):
            iou = mot.box_iou(g[mot.GT_BOX].to_numpy(float), p[mot.PRED_BOX].to_numpy(float))
            for r, c in zip(*linear_sum_assignment(-iou)):
                if iou[r, c] >= mot.IOU_MATCH:
                    hit[r] = p.index[c]
        for r, (_, gr) in enumerate(g.iterrows()):
            rows.append({"frame": int(k), "gt_id": int(gr["gt_id"]), "gt_role": gr["role"], "pred_row": hit.get(r, -1)})
    return pd.DataFrame(rows)


def q0_roles(after: pd.DataFrame) -> pd.Series:
    """Q0 (prior only) rebuilt from the saved columns: referee stays, prior fires -> keeper, rest player."""
    scored = after["role_source"].isin(["prior", "probe"])
    q0 = np.where(after["role"] == "referee", "referee", np.where(after["role_source"] == "prior", "goalkeeper", "player"))
    return pd.Series(np.where(scored, q0, after["role_pipeline"]), index=after.index)


def role_report(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {"n": int(len(y)), "accuracy": round(float((y == pred).mean()), 4) if len(y) else None}
    for c in CLASSES:
        tp = int(((pred == c) & (y == c)).sum())
        out[c] = {"n": int((y == c).sum()), "predicted": int((pred == c).sum()),
                  "recall": round(tp / max((y == c).sum(), 1), 4), "precision": round(tp / max((pred == c).sum(), 1), 4)}
    return out


def on_pitch(xy: np.ndarray) -> np.ndarray:
    return (xy[:, 0] >= 0) & (xy[:, 0] <= PITCH[0]) & (xy[:, 1] >= 0) & (xy[:, 1] <= PITCH[1])


def score(window: str) -> tuple[pd.DataFrame, dict]:
    gt, sample_of_k, before, after, calib = window_inputs(window)
    gt_people = gt[(gt["label"] == "person") & gt["role"].isin(CLASSES)]
    people = after[(after["object_class"] == "person") & (after["track_id"] >= 0)]
    people = mot.drop_ignored(people.assign(row=people.index), gt, "person").set_index("row", drop=False)
    m = match_rows(gt_people, people, sample_of_k)
    xy = role_probe.pitch_positions(after, calib)
    matched = m[m["pred_row"] >= 0].copy()
    pr = matched["pred_row"].to_numpy()
    q0 = q0_roles(after)
    matched["pipeline"] = before.loc[pr, "role"].to_numpy()
    matched["p1"] = after.loc[pr, "role"].to_numpy()
    matched["p0"] = q0.loc[pr].to_numpy()
    matched["source"] = after.loc[pr, "role_source"].to_numpy()
    matched["x"], matched["y"] = xy[pr, 0], xy[pr, 1]
    w_px = (after.loc[pr, "bbox_x2"] - after.loc[pr, "bbox_x1"]).to_numpy()
    h_px = (after.loc[pr, "bbox_y2"] - after.loc[pr, "bbox_y1"]).to_numpy()
    matched["small"] = (w_px < SMALL_PX) | (h_px < SMALL_PX)
    matched["window"] = window
    extra = people[~people.index.isin(m["pred_row"])]
    xy_extra = xy[extra.index.to_numpy()]
    keeper_or_ref = extra["role"].isin(["goalkeeper", "referee"]).to_numpy()
    stats = {"gt_boxes": int(len(m)), "matched": int(len(matched)), "unmatched_pred": int(len(extra)),
             "unmatched_pred_roles": {k: int(v) for k, v in extra["role"].value_counts().items()},
             "unmatched_pred_f5r_roles": {k: int(v) for k, v in before.loc[extra.index, "role"].value_counts().items()},
             "unmatched_pred_keeper_or_referee_on_pitch": int((keeper_or_ref & on_pitch(xy_extra)).sum()),
             "unmatched_pred_f5r_keeper_or_referee_on_pitch": int((
                 before.loc[extra.index, "role"].isin(["goalkeeper", "referee"]).to_numpy() & on_pitch(xy_extra)).sum())}
    return matched, stats


def per_keeper(matched: pd.DataFrame, col: str) -> dict:
    gk = matched[matched["gt_role"] == "goalkeeper"]
    return {f"{w[:2]}_gt{g}": {"boxes": int(len(s)), "recall": round(float((s[col] == "goalkeeper").mean()), 3)}
            for (w, g), s in gk.groupby(["window", "gt_id"])}


def tracking_block(use_after: bool) -> dict:
    results = {}
    for w in WINDOWS:
        gt, sample_of_k, before, after, _ = window_inputs(w)
        sid_to_k = {s: k for k, s in sample_of_k.items()}
        tr = (after if use_after else before)
        tr = tr[tr["sample_id"].isin(sid_to_k)].assign(sample_id=lambda d: d["sample_id"].map(sid_to_k))
        keep = sorted(sample_of_k)
        results[w] = mot.evaluate_window(gt[gt["frame"].isin(keep)], tr, max(keep) + 1)
    pooled = mot.report(results, fps=5.0)["pooled"]
    lab = pooled["labels"]
    return {"mot": pooled["mot"], "team_outfield_per_box": lab["team_outfield"]["accuracy_per_box"],
            "role_accuracy_all_matched": lab["role_accuracy"]}


def boxes_identical(window: str) -> bool:
    _, _, before, after, _ = window_inputs(window)
    cols = ["sample_id", "track_id", *mot.PRED_BOX]
    return len(before) == len(after) and bool((before[cols].to_numpy() == after[cols].to_numpy()).all())


def changed_referees_other() -> dict:
    n = bad = 0
    for w in WINDOWS:
        after = window_inputs(w)[3]
        ch = after[(after["role"] == "referee") & (after["role_pipeline"] != "referee")]
        n, bad = n + len(ch), bad + int((ch["team_id"] != "other").sum())
    return {"changed_to_referee_rows": n, "without_other": bad}


def show(out: dict) -> None:
    for name, d in out["matched_on_pitch"].items():
        print(name, d["n"], "acc", d["accuracy"], {c: (d[c]["recall"], d[c]["precision"]) for c in CLASSES})
    print("per_keeper", out["per_keeper_recall"])
    print("coverage", out["coverage"])
    print("tracking", out["tracking"])
    print("h4 identical boxes", out["h4_boxes_identical"], "| h5 referee", out["h5_changed_referee_rows_carry_other"])
    print("end_to_end", out["end_to_end"])


def main() -> None:
    C.setup_logging()
    parts, stats = [], {}
    for w in WINDOWS:
        matched, stats[w] = score(w)
        parts.append(matched)
    pairs = pd.concat(parts, ignore_index=True)
    verdict = pairs[on_pitch(pairs[["x", "y"]].to_numpy(float))].reset_index(drop=True)
    y = verdict["gt_role"].to_numpy(object)
    out = {"matched_on_pitch": {c: role_report(y, verdict[col].to_numpy(object)) for c, col in
                                (("f5r", "pipeline"), ("Q1", "p1"), ("Q0", "p0"))},
           "per_keeper_recall": {c: per_keeper(verdict, col) for c, col in (("f5r", "pipeline"), ("Q1", "p1"), ("Q0", "p0"))},
           "per_window_Q1": {w: role_report(verdict.loc[verdict["window"] == w, "gt_role"].to_numpy(object),
                                            verdict.loc[verdict["window"] == w, "p1"].to_numpy(object))
                             for w in WINDOWS},
           "coverage": {"matched_pairs": int(len(pairs)), "scored_on_pitch": int(len(verdict)),
                        "excluded_off_pitch_or_no_calibration": int(len(pairs) - len(verdict)),
                        "matched_without_usable_calibration": int(pairs["x"].isna().sum()),
                        "matched_under_8px": int(pairs["small"].sum()),
                        "role_source_counts": {k: int(v) for k, v in pairs["source"].value_counts().items()}},
           "end_to_end": {"per_window": stats, "gt_person_boxes": int(sum(s["gt_boxes"] for s in stats.values())),
                          "matched": int(len(pairs))},
           "tracking": {"f5r": tracking_block(False), "p": tracking_block(True)},
           "h4_boxes_identical": all(boxes_identical(w) for w in WINDOWS),
           "h5_changed_referee_rows_carry_other": changed_referees_other()}
    C.write_json(C.resolve("artifacts") / "role_probe" / "eval_detector.json", out)
    show(out)


if __name__ == "__main__":
    main()
