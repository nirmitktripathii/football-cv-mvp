"""Atomic-VAEP via ``socceraction.atomic.vaep.AtomicVAEP``, trained on public event data.

Why not on the CV clip: Atomic-VAEP rates an *action stream* (who did what,
where, with which body part, and the outcome). The CV pipeline produces
positions only: no player identities, no action types, no outcomes, and the
ball in about half of the frames. Any action stream derived from it would be
invented, so the clip-level Atomic-VAEP result is reported as INCOMPLETE.

What is real here: a model trained on StatsBomb-derived legacy SPADL games
(converted with ``socceraction.atomic.spadl.convert_to_atomic``), evaluated
on a held-out competition (FIFA World Cup 2018) with Brier score and ROC AUC,
and an integration test that rates one held-out game end to end. That is the
component a future CV event-detection stage would plug into.

Legacy SPADL stores lack ``original_event_id``; a placeholder
``"spadl-<action_id>"`` is filled in because socceraction copies that column
through the atomic conversion and its game-state shift cannot fill nulls.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _store_games(path: Path, max_games: int | None, seed: int) -> tuple[pd.HDFStore, pd.DataFrame]:
    st = pd.HDFStore(str(path), "r")
    games = st["games"]
    if max_games is not None and len(games) > max_games:
        games = games.sample(max_games, random_state=seed)
    return st, games


def _atomic(store: pd.HDFStore, game_id: int) -> pd.DataFrame:
    from socceraction.atomic.spadl import convert_to_atomic

    actions = store[f"actions/game_{game_id}"]
    actions = actions.assign(original_event_id="spadl-" + actions["action_id"].astype(str))
    return convert_to_atomic(actions)


def build_dataset(model, stores: list[Path], max_games_per_store: int | None, seed: int):
    """Features and labels for every game of the given stores."""
    Xs, ys, n_games = [], [], 0
    for path in stores:
        st, games = _store_games(path, max_games_per_store, seed)
        try:
            for g in games.itertuples(index=False):
                game = pd.Series(g._asdict())
                at = _atomic(st, game.game_id)
                Xs.append(model.compute_features(game, at))
                ys.append(model.compute_labels(game, at))
                n_games += 1
        finally:
            st.close()
        log.info("Atomic-VAEP features: %s done (%d games so far)", path.name, n_games)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True), n_games


def train_and_evaluate(
    spadl_dir: Path, train_stores: list[str], test_store: str, out_dir: Path,
    max_games_per_store: int | None = None, seed: int = 0,
) -> dict:
    """Fit AtomicVAEP (xgboost) on ``train_stores`` and score it on ``test_store``."""
    from socceraction.atomic.vaep import AtomicVAEP

    t0 = time.perf_counter()
    model = AtomicVAEP(nb_prev_actions=3)
    X_tr, y_tr, n_tr = build_dataset(model, [spadl_dir / s for s in train_stores], max_games_per_store, seed)
    X_te, y_te, n_te = build_dataset(model, [spadl_dir / test_store], None, seed)
    t_feat = time.perf_counter() - t0
    np.random.seed(seed)
    model.fit(
        X_tr, y_tr, learner="xgboost", val_size=0.25,
        tree_params={"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "random_state": seed, "n_jobs": -1},
        fit_params={"verbose": False},
    )
    scores = model.score(X_te, y_te)
    base = {k: float(y_te[k].mean()) for k in ("scores", "concedes")}
    brier_ref = {k: float(base[k] * (1 - base[k])) for k in base}  # Brier of always predicting the base rate
    result = {
        "status": "trained_and_evaluated_on_public_event_data",
        "cv_clip_status": "INCOMPLETE: no valid action stream can be derived from the CV clip (see module docstring)",
        "train_stores": train_stores, "test_store": test_store,
        "train_games": n_tr, "test_games": n_te,
        "train_atomic_actions": int(len(X_tr)), "test_atomic_actions": int(len(X_te)),
        "n_features": int(X_tr.shape[1]), "nb_prev_actions": 3, "learner": "xgboost", "seed": seed,
        "test_scores": {k: {m: round(float(v), 5) for m, v in d.items()} for k, d in scores.items()},
        "test_base_rate": {k: round(v, 5) for k, v in base.items()},
        "test_brier_of_base_rate": {k: round(v, 5) for k, v in brier_ref.items()},
        "seconds_features": round(t_feat, 1), "seconds_total": round(time.perf_counter() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "atomic_vaep_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    integration = integration_test(model, spadl_dir / test_store)
    integration.to_csv(out_dir / "atomic_vaep_integration_top_actions.csv", index=False)
    log.info("Atomic-VAEP: %s", {k: result[k] for k in ("train_games", "test_games", "test_scores")})
    return result


def integration_test(model, store_path: Path, top_n: int = 15) -> pd.DataFrame:
    """Rate one held-out game end to end and return its highest-valued atomic actions."""
    import socceraction.atomic.spadl as aspadl

    st = pd.HDFStore(str(store_path), "r")
    try:
        games = st["games"]
        game = games.sort_values("game_id").iloc[-1]  # deterministic choice: last game id in the store
        at = _atomic(st, game.game_id)
    finally:
        st.close()
    ratings = model.rate(game, at)
    rated = aspadl.add_names(at).join(ratings)
    cols = ["game_id", "period_id", "time_seconds", "team_id", "player_id", "type_name", "x", "y", "dx", "dy",
            "offensive_value", "defensive_value", "vaep_value"]
    return rated.sort_values("vaep_value", ascending=False).head(top_n)[cols].round(4)
