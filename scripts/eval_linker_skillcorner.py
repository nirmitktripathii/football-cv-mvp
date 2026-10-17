"""Score the identity linker on SkillCorner open tracking (true player ids known).

    python scripts/eval_linker_skillcorner.py [--match-dir data/external/skillcorner/1886347] [--windows 150]

Tuning of the defaults used period 1 and seed 100; this script scores the held-out period 2 with seed 0.
The pass criterion (precision >= 0.98 in the reference scenario) is fixed in football_cv.evaluation.linker_eval.
Writes artifacts/<clip>/identity/skillcorner_linker_eval.json.
"""
from __future__ import annotations

import argparse
import json
import logging

from football_cv import config as C
from football_cv.evaluation import linker_eval as LE
from football_cv.evaluation import skillcorner as sc

log = logging.getLogger("eval_linker")
SCENARIOS = {
    "reference": LE.REFERENCE,
    "no_offset": {"offset_sigma_m": 0.0, "team_noise": 0.05},
    "offset_5m": {"offset_sigma_m": 5.0, "team_noise": 0.05},
    "team_noise_15pct": {"offset_sigma_m": 3.0, "team_noise": 0.15},
}
VARIANTS = {                     # name -> identity-section overrides
    "default": {},
    "no_edge_rule": {"exclude_edge_breaks": False},
    "no_unseen_cap": {"max_false_matches": 1e9},
    "strict_margin0": {"ambiguity_margin": 0.0},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--match-dir", default="data/external/skillcorner/1886347")
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--period", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = C.load_config(args.config)
    match_dir = C.resolve(args.match_dir)
    match_id = int(match_dir.name)
    table, views = sc.load_match(match_dir, match_id), sc.load_views(match_dir, match_id)

    results = {"config": {"windows": args.windows, "seed": args.seed, "period": args.period, "match": match_dir.name,
                          "criterion": f"precision >= {LE.PRECISION_TARGET} in reference scenario, default variant"}}
    for name, sc_args in SCENARIOS.items():
        for variant, over in VARIANTS.items():
            r = LE.run_scenario(table, {"identity": over}, seed=args.seed, n_windows=args.windows,
                                period=args.period, views=views, **sc_args)
            results[f"{name}|{variant}"] = r
            log.info("%-18s %-15s links %4d  precision %.3f  recall %.3f  impure %d", name, variant,
                     r.get("links", 0), r["precision"], r["recall"], r.get("impure_identities", 0))
        g = LE.run_scenario(table, None, seed=args.seed, n_windows=args.windows, greedy=True, period=args.period,
                            views=views, **sc_args)
        results[f"{name}|greedy_baseline"] = g
        log.info("%-18s %-15s links %4d  precision %.3f  recall %.3f  impure %d", name, "greedy_baseline",
                 g.get("links", 0), g["precision"], g["recall"], g.get("impure_identities", 0))
    out = C.artifacts_dir(cfg) / "identity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "skillcorner_linker_eval.json").write_text(json.dumps(results, indent=2))
    ref = results["reference|default"]
    log.info("criterion %s (precision %.3f over %d links)", "PASS" if ref["precision"] >= LE.PRECISION_TARGET else "FAIL",
             ref["precision"], ref.get("links", 0))


if __name__ == "__main__":
    main()
