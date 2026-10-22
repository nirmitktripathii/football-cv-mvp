"""Stage 1d: keeper chosen by distance to the goal centre, probe keeper probability as tie-break (docs/role_probe.md).

    python scripts/probe_roles_centre.py

Same boxes, calibration, folds and criteria as scripts/probe_roles_position.py. Writes artifacts/role_probe/probe_centre.json.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from football_cv import config as C

sys.path.insert(0, str(C.resolve("scripts")))
import probe_roles_position as P  # noqa: E402
from probe_roles import CLASSES, SEED, WINDOWS, build, fit_predict, report  # noqa: E402

PEN_HALF_W = 20.16
GOAL_Y = P.PITCH_W / 2
TIE_M = 2.0


def keeper_proba(x: np.ndarray, y: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Out-of-fold probability of 'goalkeeper' (leave-one-window-out, as fit_predict)."""
    out = np.zeros(len(y))
    for w in WINDOWS:
        te = windows == w
        sc = StandardScaler().fit(x[~te])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED).fit(sc.transform(x[~te]), y[~te])
        out[te] = clf.predict_proba(sc.transform(x[te]))[:, list(clf.classes_).index("goalkeeper")]
    return out


def centre_keepers(meta, probe: np.ndarray, p_gk: np.ndarray) -> np.ndarray:
    fires = np.zeros(len(meta), bool)
    ok = meta["on_pitch"].to_numpy() & (probe != "referee")
    xs, ys = meta["x_m"].to_numpy(), meta["y_m"].to_numpy()
    for _, idx in meta.groupby(["window", "frame"]).indices.items():
        idx = idx[ok[idx]]
        for end_x in (0.0, P.PITCH_L):
            cand = idx[(np.abs(xs[idx] - end_x) <= P.PEN_DEPTH) & (np.abs(ys[idx] - GOAL_Y) <= PEN_HALF_W)]
            if not len(cand):
                continue
            d = np.hypot(xs[cand] - end_x, ys[cand] - GOAL_Y)
            near = cand[d <= d.min() + TIE_M]
            fires[near[int(np.argmax(p_gk[near]))]] = True
    return fires


def main() -> None:
    C.setup_logging()
    meta, feats, _ = build("sritrack", "cpu", "")
    meta = P.pitch_positions(meta)
    meta["on_pitch"] = meta["x_m"].between(0, P.PITCH_L) & meta["y_m"].between(0, P.PITCH_W)
    y, wins = meta["role"].to_numpy(object), meta["window"].to_numpy(object)
    probe = fit_predict(feats["fc"], y, wins)
    fires = centre_keepers(meta, probe, keeper_proba(feats["fc"], y, wins))
    q0 = np.where(fires, "goalkeeper", np.where(probe == "goalkeeper", "player", probe))
    q1 = np.where(fires | (probe == "goalkeeper"), "goalkeeper", probe)
    keep = meta["on_pitch"].to_numpy()
    matched = keep & (meta["pipeline_role"] != "").to_numpy()
    pipe = meta["pipeline_role"].to_numpy(object)
    kept = meta[keep].reset_index(drop=True)
    out = {"on_pitch": int(keep.sum()), "tie_m": TIE_M,
           "stage1_probe": report(y[keep], probe[keep]), "Q0": report(y[keep], q0[keep]), "Q1": report(y[keep], q1[keep]),
           "Q1_per_window": {w[:2]: report(y[keep & (wins == w)], q1[keep & (wins == w)]) for w in WINDOWS},
           "keeper_recall_Q1": P.per_keeper_recall(kept, q1[keep]),
           "pipeline_matched_on_pitch": report(y[matched], pipe[matched]), "Q1_matched_on_pitch": report(y[matched], q1[matched])}
    fp = keep & (q1 == "goalkeeper") & (y != "goalkeeper")
    out["false_keepers"] = {w[:2]: {"prior": int((fires & fp & (wins == w)).sum()),
                                    "probe_only": int((~fires & fp & (wins == w)).sum())} for w in WINDOWS}
    C.write_json(C.resolve("artifacts") / "role_probe" / "probe_centre.json", out)
    for k in ("stage1_probe", "Q0", "Q1", "pipeline_matched_on_pitch", "Q1_matched_on_pitch"):
        d = out[k]
        print(k, d["n"], d["accuracy"], {c: (d[c]["recall"], d[c]["precision"]) for c in CLASSES})
    print("keeper_recall_Q1", out["keeper_recall_Q1"]); print("false_keepers", out["false_keepers"])


if __name__ == "__main__":
    main()
