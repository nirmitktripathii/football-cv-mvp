"""SRITrack appearance evidence for the tracklet linker: per-track embeddings and a bounded log-likelihood-ratio cost.

``track_embeddings`` crops each track's boxes (tall enough to carry detail), embeds them with
``reid_embed.SriTrackEmbedder`` and averages the L2-normalised vectors per track. ``reid_llr`` turns the cosine distance
of two tracks into ``log p(d | different team-mate) - log p(d | same person)`` in nats, using Gaussian models of those two
distances measured on the hand-audited windows (``scripts/reid_calibrate.py`` -> ``artifacts/reid_distance_model.json``).
Lower is better (a small distance makes it negative), so it adds to the linker's cost like the kit-colour term does. It is
clipped to ``+-clip`` because the distance model comes from 3 short windows (45 same-person and 244 different-person
pairs, AUC about 0.81): the cue can nudge a decision, never decide it alone.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_BOX_H = 40.0
MAX_CROPS = 12
MIN_CROPS = 3


def _gauss_logpdf(d: np.ndarray, mean: float, std: float) -> np.ndarray:
    return -0.5 * ((d - mean) / std) ** 2 - np.log(std)


def track_embeddings(tracks: pd.DataFrame, frames: pd.DataFrame, frames_root, embedder, feature: str = "fc",
                     min_box_h: float = MIN_BOX_H, max_crops: int = MAX_CROPS, min_crops: int = MIN_CROPS) -> pd.DataFrame:
    """One row per track with at least ``min_crops`` usable crops: ``track_id``, ``n_crops`` and ``e_0..e_d`` (unit norm)."""
    ppl = tracks[(tracks["object_class"] == "person") & (tracks["track_id"] >= 0)
                 & ((tracks["bbox_y2"] - tracks["bbox_y1"]) >= min_box_h)]
    picked = []
    for _, g in ppl.groupby("track_id"):
        if len(g) >= min_crops:
            g = g.sort_values("timestamp_s")
            picked.append(g.iloc[np.unique(np.linspace(0, len(g) - 1, min(max_crops, len(g))).round().astype(int))])
    if not picked:
        return pd.DataFrame(columns=["track_id", "n_crops"])
    sel = pd.concat(picked, ignore_index=True)
    paths = frames.set_index("sample_id")["path"]
    crops, owner = [], []
    for sid, grp in sel.groupby("sample_id"):
        img = cv2.imread(str(Path(frames_root) / paths[sid]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for r in grp.itertuples():
            x1, y1, x2, y2 = max(int(r.bbox_x1), 0), max(int(r.bbox_y1), 0), min(int(r.bbox_x2), w), min(int(r.bbox_y2), h)
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                crops.append(img[y1:y2, x1:x2].copy())
                owner.append(r.track_id)
    feats = embedder.embed(crops)[feature]
    owner = np.array(owner)
    rows = []
    for tid in np.unique(owner):
        v = feats[owner == tid]
        if len(v) >= min_crops:
            m = v.mean(0)
            rows.append([int(tid), len(v), *(m / (np.linalg.norm(m) + 1e-9))])
    log.info("Re-ID embeddings: %d crops -> %d tracks", len(crops), len(rows))
    return pd.DataFrame(rows, columns=["track_id", "n_crops", *[f"e_{i}" for i in range(feats.shape[1])]])


def reid_llr(pred: pd.Series, succ: pd.Series, emb: pd.DataFrame, model: dict, clip: float = 2.0) -> np.ndarray:
    """Per pair: ``log p(d|different) - log p(d|same)`` clipped to ``+-clip``; 0 when either track has no embedding."""
    cols = [c for c in emb.columns if c.startswith("e_")]
    idx = emb.set_index("track_id")
    a, b = idx.reindex(pred.to_numpy()), idx.reindex(succ.to_numpy())
    d = 1.0 - (a[cols].to_numpy(float) * b[cols].to_numpy(float)).sum(1)
    cost = _gauss_logpdf(d, model["diff"]["mean"], model["diff"]["std"]) - _gauss_logpdf(d, model["same"]["mean"], model["same"]["std"])
    return np.clip(np.nan_to_num(cost, nan=0.0), -clip, clip)
