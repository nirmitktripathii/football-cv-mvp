"""Role probe: can frozen embeddings label player / goalkeeper / referee? Criteria are declared in docs/role_probe.md.

    python scripts/probe_roles.py --device cuda --label sritrack

Embeds every ground-truth person box of the audited windows a1-a3, then scores a leave-one-window-out logistic
regression. Embeddings are cached in artifacts/role_probe/emb_<label>.npz so a second embedder reuses the same boxes
and folds. Writes artifacts/role_probe/probe_<label>.json.
"""
from __future__ import annotations

import argparse
import json
import logging

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from football_cv import config as C
from football_cv.evaluation import cvat
from football_cv.tracking import appearance
from football_cv.tracking.reid_embed import SriTrackEmbedder

log = logging.getLogger("probe_roles")
WINDOWS = ("a1_open_t600", "a2_corner_t2707", "a3_open_t4800")
CLASSES = ("player", "goalkeeper", "referee")
MIN_SIDE = 8
SRC_PER_GT = 5  # 25 fps source, 5 fps annotation
IOU_MATCH = 0.5
PRIMARY = "fc"
SEED = 0


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def window_boxes(window: str):
    """(meta, crops) for the GT person boxes of one window; meta has the pipeline's role for matched boxes."""
    gt = cvat.parse_cvat_xml((C.resolve("data/annotations") / f"inter_napoli_fb4_{window}" / "annotations.xml")
                             .read_text(encoding="utf-8"))
    gt = gt[(gt["label"] == "person") & gt["role"].isin(CLASSES)]
    run = C.resolve("data/processed") / f"inter_napoli_fb4f5_{window}"
    frames = pd.read_parquet(run / "frames.parquet")
    first = int(frames["frame_id"].min())
    by_frame = frames.set_index("frame_id")
    tracks = pd.read_parquet(C.resolve("data/processed") / f"inter_napoli_fb4f5r_{window}" / "tracks.parquet")
    tracks = tracks[tracks["object_class"] != "ball"]
    rows, crops, dropped = [], [], 0
    for k, sub in gt.groupby("frame"):
        fid = first + SRC_PER_GT * int(k)
        if fid not in by_frame.index:
            dropped += len(sub)
            continue
        sample = int(by_frame.at[fid, "sample_id"])
        img = cv2.imread(str(run / by_frame.at[fid, "path"]))
        h, w = img.shape[:2]
        pred = tracks[tracks["sample_id"] == sample]
        pbox = pred[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
        for r in sub.itertuples():
            x1, y1, x2, y2 = max(int(r.x1), 0), max(int(r.y1), 0), min(int(r.x2), w), min(int(r.y2), h)
            if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                dropped += 1
                continue
            role = ""
            if len(pbox):
                o = iou(np.array([r.x1, r.y1, r.x2, r.y2], float), pbox)
                if o.max() >= IOU_MATCH:
                    role = str(pred.iloc[int(o.argmax())]["role"])
            rows.append({"window": window, "frame": int(k), "gt_id": int(r.gt_id), "role": r.role,
                         "pipeline_role": role, "h": y2 - y1, "w": x2 - x1})
            crops.append(img[y1:y2, x1:x2].copy())
    log.info("%s: %d boxes, %d dropped (missing frame or < %d px)", window, len(rows), dropped, MIN_SIDE)
    return pd.DataFrame(rows), crops, dropped


def kit_size_features(meta: pd.DataFrame, crops: list[np.ndarray]) -> np.ndarray:
    kit = np.zeros((len(crops), 9))
    for i, c in enumerate(crops):
        v = appearance.band_colours(c, (0, 0, c.shape[1], c.shape[0]))
        if v is not None:
            kit[i] = v
    return np.c_[kit / 100.0, np.log(meta["h"].to_numpy(float)), (meta["w"] / meta["h"]).to_numpy(float)]


def build(label: str, device: str, checkpoint: str):
    path = C.resolve("artifacts") / "role_probe" / f"emb_{label}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return pd.DataFrame(json.loads(str(z["meta"]))), {k: z[k] for k in z.files if k not in ("meta", "dropped")}, int(z["dropped"])
    metas, crops_all, dropped = [], [], 0
    for w in WINDOWS:
        m, c, d = window_boxes(w)
        metas.append(m); crops_all += c; dropped += d
    meta = pd.concat(metas, ignore_index=True)
    emb = SriTrackEmbedder(C.resolve(checkpoint), (256, 128), device)
    feats = emb.embed(crops_all)
    feats["kit_size"] = kit_size_features(meta, crops_all)
    np.savez(path, meta=json.dumps(meta.to_dict("list")), dropped=dropped, sha256=emb.sha256, **feats)
    return meta, feats, dropped


def fit_predict(x: np.ndarray, y: np.ndarray, windows: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for w in WINDOWS:
        te = windows == w
        sc = StandardScaler().fit(x[~te])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED).fit(sc.transform(x[~te]), y[~te])
        pred[te] = clf.predict(sc.transform(x[te]))
    return pred


def report(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {"n": int(len(y)), "accuracy": round(float((y == pred).mean()), 4) if len(y) else None}
    for c in CLASSES:
        tp = int(((pred == c) & (y == c)).sum())
        out[c] = {"n": int((y == c).sum()), "recall": round(tp / max((y == c).sum(), 1), 4),
                  "precision": round(tp / max((pred == c).sum(), 1), 4)}
    out["confusion"] = {t: {p: int(((y == t) & (pred == p)).sum()) for p in CLASSES} for t in CLASSES}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="sritrack")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default="models/sritrack/model.pth.tar-60")
    args = ap.parse_args()
    C.setup_logging()
    meta, feats, dropped = build(args.label, args.device, args.checkpoint)
    y, wins = meta["role"].to_numpy(object), meta["window"].to_numpy(object)
    matched = (meta["pipeline_role"] != "").to_numpy()
    out = {"label": args.label, "primary": PRIMARY, "classes": CLASSES, "seed": SEED, "boxes": int(len(meta)),
           "dropped_boxes": int(dropped), "matched_to_pipeline": int(matched.sum()), "features": {}}
    for name in (*[k for k in feats if k not in ("kit_size", "sha256")], "kit_size"):
        pred = fit_predict(feats[name], y, wins)
        out["features"][name] = {"pooled": report(y, pred), "matched_subset": report(y[matched], pred[matched]),
                                 "per_window": {w: report(y[wins == w], pred[wins == w]) for w in WINDOWS}}
    out["pipeline"] = report(y[matched], meta["pipeline_role"].to_numpy(object)[matched])
    out["majority_class_accuracy"] = round(float((y == "player").mean()), 4)
    dest = C.resolve("artifacts") / "role_probe" / f"probe_{args.label}.json"
    C.write_json(dest, out)
    log.info("written %s", dest)


if __name__ == "__main__":
    main()
