"""Train the role-probe models from the cached ground-truth embeddings (docs/role_probe.md stage 2).

    python scripts/train_role_probe.py

Writes models/role_probe/lowo_<window>.npz (fit on the other two windows; used to score <window> without leakage) and
models/role_probe/all.npz (fit on all three; the model a deployment would use). Fixed seed, no randomness otherwise.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from football_cv import config as C
from football_cv.tracking.role_probe import CLASSES, RoleModel

log = logging.getLogger("train_role_probe")
EMBEDDINGS = "artifacts/role_probe/emb_sritrack.npz"
FEATURE = "fc"
OUT_DIR = "models/role_probe"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    z = np.load(C.resolve(EMBEDDINGS), allow_pickle=False)
    meta = pd.DataFrame(json.loads(str(z["meta"])))
    x, y = z[FEATURE], meta["role"].to_numpy()
    if not set(y) <= set(CLASSES):
        raise ValueError(f"unexpected roles {set(y) - set(CLASSES)}")
    out = C.resolve(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    for window in sorted(meta["window"].unique()):
        train = (meta["window"] != window).to_numpy()
        RoleModel.fit(x[train], y[train]).save(out / f"lowo_{window}.npz")
        log.info("lowo_%s: trained on %d boxes", window, int(train.sum()))
    RoleModel.fit(x, y).save(out / "all.npz")
    log.info("all: trained on %d boxes", len(y))


if __name__ == "__main__":
    main()
