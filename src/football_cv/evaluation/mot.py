"""Image-space tracking metrics against hand-corrected ground truth (phase 0, docs/annotation_guide.md).

Ground truth is a ``cvat.parse_cvat_xml`` frame; predictions are ``tracks.parquet`` rows of the same window,
whose ``sample_id`` is the CVAT frame number.

- People: HOTA (DetA, AssA, LocA), CLEAR (MOTA, MOTP, IDSW, Frag, MT/PT/ML) and Identity (IDF1, IDP, IDR),
  computed with TrackEval's own code (vendored in ``trackeval_vendor``) so the numbers are comparable with
  published ones. Each window is one sequence; windows are pooled with TrackEval's ``combine_sequences``.
- Labels: team accuracy and goalkeeper affiliation on ground-truth boxes matched at IoU >= 0.5, and how
  often a referee is given a team (it should be never).
- Ball: per-frame recall and precision at a centre-distance radius, and the longest run of missed frames.

Predictions that fall on an ``ignore`` region and match no ground-truth box of their class are dropped
before scoring, so staff and spare balls are neither rewarded nor punished.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_cv.evaluation.cvat import BALL, IGNORE, PERSON
from football_cv.evaluation.trackeval_vendor.metrics import CLEAR, HOTA, Identity

IOU_MATCH = 0.5          # TrackEval's CLEAR/Identity threshold, also used for label matching
IGNORE_IOA = 0.5         # share of a prediction's area inside an ignore box for it to be dropped
BALL_RADIUS_PX = 10.0
OUTFIELD = ("home", "away")
PRED_BOX = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
GT_BOX = ["x1", "y1", "x2", "y2"]
_QUIET = {"PRINT_CONFIG": False}
_HOTA_FIELDS = ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr")
_CLEAR_FLOAT = ("MOTA", "MOTP", "CLR_Re", "CLR_Pr")
_CLEAR_INT = ("IDSW", "Frag", "MT", "PT", "ML", "CLR_TP", "CLR_FN", "CLR_FP")
PAIR_COLUMNS = ["frame", "gt_id", "gt_role", "gt_team", "track_id", "pred_team", "pred_role", "iou"]


@dataclass
class WindowResult:
    raw: dict[str, dict]       # TrackEval result per metric class name, kept for pooling
    counts: dict[str, int]
    pairs: pd.DataFrame        # matched (ground truth, prediction) person boxes with their labels
    ball: dict


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def _area(a: np.ndarray) -> np.ndarray:
    return np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) of xyxy boxes."""
    inter = _intersection(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def box_ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box in ``a`` covered by each box in ``b``."""
    inter = _intersection(a, b)
    area = _area(a)[:, None]
    return np.divide(inter, area, out=np.zeros_like(inter), where=area > 0)


def drop_ignored(pred: pd.DataFrame, gt: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop predictions inside an ``ignore`` box unless they match a ground-truth ``label`` box."""
    ignore = gt[gt["label"] == IGNORE]
    keep = np.ones(len(pred), bool)
    frames = pred["sample_id"].to_numpy()
    for frame, ig in ignore.groupby("frame"):
        idx = np.flatnonzero(frames == frame)
        if not len(idx):
            continue
        boxes = pred.iloc[idx][PRED_BOX].to_numpy(float)
        inside = box_ioa(boxes, ig[GT_BOX].to_numpy(float)).max(1) >= IGNORE_IOA
        same = gt[(gt["frame"] == frame) & (gt["label"] == label)][GT_BOX].to_numpy(float)
        matched = box_iou(boxes, same).max(1) >= IOU_MATCH if len(same) else np.zeros(len(idx), bool)
        keep[idx[inside & ~matched]] = False
    return pred[keep].reset_index(drop=True)


def build_sequence(gt: pd.DataFrame, pred: pd.DataFrame, num_frames: int) -> dict:
    """TrackEval's per-sequence input: contiguous ids and an IoU matrix per frame."""
    gid = {v: i for i, v in enumerate(sorted(gt["gt_id"].unique()))}
    tid = {v: i for i, v in enumerate(sorted(pred["track_id"].unique()))}
    g_by, p_by = dict(tuple(gt.groupby("frame"))), dict(tuple(pred.groupby("sample_id")))
    data = {"num_timesteps": num_frames, "num_gt_ids": len(gid), "num_tracker_ids": len(tid),
            "num_gt_dets": len(gt), "num_tracker_dets": len(pred),
            "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for t in range(num_frames):
        g, p = g_by.get(t, gt.iloc[:0]), p_by.get(t, pred.iloc[:0])
        data["gt_ids"].append(g["gt_id"].map(gid).to_numpy(int))
        data["tracker_ids"].append(p["track_id"].map(tid).to_numpy(int))
        data["similarity_scores"].append(box_iou(g[GT_BOX].to_numpy(float), p[PRED_BOX].to_numpy(float)))
    return data


def match_people(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """One-to-one IoU matches per frame (Hungarian, IoU >= ``IOU_MATCH``) with both sides' labels."""
    has_role = "role" in pred
    p_by = dict(tuple(pred.groupby("sample_id")))
    rows = []
    for frame, g in gt.groupby("frame"):
        p = p_by.get(frame)
        if p is None:
            continue
        iou = box_iou(g[GT_BOX].to_numpy(float), p[PRED_BOX].to_numpy(float))
        for r, c in zip(*linear_sum_assignment(-iou)):
            if iou[r, c] < IOU_MATCH:
                continue
            gr, pr = g.iloc[r], p.iloc[c]
            rows.append({"frame": frame, "gt_id": gr["gt_id"], "gt_role": gr["role"], "gt_team": gr["team"],
                         "track_id": pr["track_id"], "pred_team": pr["team_id"],
                         "pred_role": pr["role"] if has_role else None, "iou": float(iou[r, c])})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _centres(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    b = df[cols].to_numpy(float)
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def ball_hits(gt: pd.DataFrame, pred: pd.DataFrame, radius: float = BALL_RADIUS_PX) -> dict:
    """Centre-distance matches of ground-truth and predicted balls, frame by frame."""
    g_by, p_by = dict(tuple(gt.groupby("frame"))), dict(tuple(pred.groupby("sample_id")))
    errors, hit_frames = [], set()
    for frame, g in g_by.items():
        p = p_by.get(frame)
        if p is None:
            continue
        d = np.linalg.norm(_centres(g, GT_BOX)[:, None] - _centres(p, PRED_BOX)[None], axis=2)
        r, c = linear_sum_assignment(d)
        ok = d[r, c] <= radius
        errors += d[r, c][ok].tolist()
        if ok.any():
            hit_frames.add(frame)
    return {"gt": len(gt), "pred": len(pred), "tp": len(errors), "errors_px": errors,
            "max_missed_run": _longest_miss(sorted(g_by), hit_frames)}


def _longest_miss(gt_frames: list[int], hits: set[int]) -> int:
    """Longest run of consecutive frames that have a ground-truth ball and no hit."""
    best = run = 0
    prev = None
    for f in gt_frames:
        if f in hits:
            run = 0
        else:
            run = run + 1 if prev is not None and f == prev + 1 else 1
        best, prev = max(best, run), f
    return best


def evaluate_window(gt: pd.DataFrame, pred: pd.DataFrame, num_frames: int,
                    ball_radius: float = BALL_RADIUS_PX) -> WindowResult:
    gt = gt[gt["frame"] < num_frames]
    pred = pred[(pred["sample_id"] >= 0) & (pred["sample_id"] < num_frames)]
    gt_people = gt[gt["label"] == PERSON]
    all_people = pred[(pred["object_class"] == PERSON) & (pred["track_id"] >= 0)]
    people = drop_ignored(all_people, gt, PERSON)
    balls = drop_ignored(pred[pred["object_class"] == BALL], gt, BALL)
    data = build_sequence(gt_people, people, num_frames)
    raw = {m.get_name(): m.eval_sequence(data) for m in (HOTA(), CLEAR(_QUIET), Identity(_QUIET))}
    counts = {"frames": num_frames, "gt_ids": data["num_gt_ids"], "pred_ids": data["num_tracker_ids"],
              "gt_boxes": data["num_gt_dets"], "pred_boxes": data["num_tracker_dets"],
              "pred_boxes_dropped_on_ignore": len(all_people) - len(people)}
    pairs = match_people(gt_people, people)
    return WindowResult(raw, counts, pairs, ball_hits(gt[gt["label"] == BALL], balls, ball_radius))


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return None
    p, zz = k / n, z * z
    centre, half = p + zz / (2 * n), z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n))
    return [round((centre - half) / (1 + zz / n), 4), round((centre + half) / (1 + zz / n), 4)]


def _rate(k: int, n: int) -> dict:
    return {"n": int(n), "k": int(k), "rate": round(k / n, 4) if n else None, "ci95": wilson(int(k), int(n))}


def _affiliation(pairs: pd.DataFrame) -> dict:
    """Coverage (a home/away call was made) and accuracy of the calls, per box and per ground-truth track."""
    called = pairs[pairs["pred_team"].isin(OUTFIELD)]
    per_track = called.groupby(["window", "gt_id"]).agg(
        gt=("gt_team", "first"), pred=("pred_team", lambda s: s.mode().iloc[0]))
    return {"coverage": _rate(len(called), len(pairs)),
            "accuracy_per_box": _rate((called["pred_team"] == called["gt_team"]).sum(), len(called)),
            "accuracy_per_track": _rate((per_track["pred"] == per_track["gt"]).sum(), len(per_track))}


def label_metrics(pairs: pd.DataFrame) -> dict:
    """Team and role metrics on matched boxes. Boxes within a track are not independent, so the per-box
    interval is optimistic; the per-track figure is the conservative one."""
    if "window" not in pairs:
        pairs = pairs.assign(window="")
    sided = pairs[pairs["gt_team"].isin(OUTFIELD)]
    refs = pairs[pairs["gt_role"] == "referee"]
    out = {"matched_boxes": int(len(pairs)),
           "team_outfield": _affiliation(sided[sided["gt_role"] == "player"]),
           "goalkeeper_affiliation": _affiliation(sided[sided["gt_role"] == "goalkeeper"]),
           "referee_given_team": _rate(refs["pred_team"].isin(OUTFIELD).sum(), len(refs))}
    known = pairs[pairs["pred_role"].fillna("").ne("") & (pairs["gt_role"] != "unknown")]
    out["role_accuracy"] = _rate((known["pred_role"] == known["gt_role"]).sum(), len(known)) if len(known) else None
    return out


def ball_summary(ball: dict, fps: float) -> dict:
    e = np.asarray(ball["errors_px"])
    return {"gt_frames": ball["gt"], "pred_frames": ball["pred"], "hits": ball["tp"],
            "recall": round(ball["tp"] / ball["gt"], 4) if ball["gt"] else None,
            "precision": round(ball["tp"] / ball["pred"], 4) if ball["pred"] else None,
            "centre_error_px_median": round(float(np.median(e)), 2) if len(e) else None,
            "centre_error_px_p90": round(float(np.percentile(e, 90)), 2) if len(e) else None,
            "max_missed_run_frames": ball["max_missed_run"],
            "max_missed_run_s": round(ball["max_missed_run"] / fps, 2)}


def mot_summary(raw: dict[str, dict]) -> dict:
    h, c, i = raw["HOTA"], raw["CLEAR"], raw["Identity"]
    out = {k: round(float(np.mean(h[k])), 4) for k in _HOTA_FIELDS}
    out.update({k: round(float(c[k]), 4) for k in _CLEAR_FLOAT})
    out.update({k: int(c[k]) for k in _CLEAR_INT})
    out.update({k: round(float(i[k]), 4) for k in ("IDF1", "IDP", "IDR")})
    return out


def _pool_ball(balls: list[dict]) -> dict:
    return {"gt": sum(b["gt"] for b in balls), "pred": sum(b["pred"] for b in balls),
            "tp": sum(b["tp"] for b in balls), "errors_px": [e for b in balls for e in b["errors_px"]],
            "max_missed_run": max((b["max_missed_run"] for b in balls), default=0)}


def _block(raw: dict, counts: dict, pairs: pd.DataFrame, ball: dict, fps: float) -> dict:
    frag = round(counts["pred_ids"] / counts["gt_ids"], 3) if counts["gt_ids"] else None
    return {"counts": counts, "pred_ids_per_gt_id": frag, "mot": mot_summary(raw),
            "labels": label_metrics(pairs), "ball": ball_summary(ball, fps)}


def report(results: dict[str, WindowResult], fps: float) -> dict:
    """Per-window and pooled metrics, JSON-ready."""
    out = {"windows": {n: _block(r.raw, r.counts, r.pairs.assign(window=n), r.ball, fps) for n, r in results.items()}}
    pooled_raw = {m.get_name(): m.combine_sequences({n: r.raw[m.get_name()] for n, r in results.items()})
                  for m in (HOTA(), CLEAR(_QUIET), Identity(_QUIET))}
    counts = {k: sum(r.counts[k] for r in results.values()) for k in next(iter(results.values())).counts}
    pairs = pd.concat([r.pairs.assign(window=n) for n, r in results.items()], ignore_index=True)
    out["pooled"] = _block(pooled_raw, counts, pairs, _pool_ball([r.ball for r in results.values()]), fps)
    return out
