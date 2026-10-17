"""Score inferred regions on SkillCorner open tracking by hiding detected stretches (truth known).

    python scripts/eval_regions_skillcorner.py [--match-dir data/external/skillcorner/1886347] [--trials 150]

Writes artifacts/<clip>/regions/skillcorner_region_eval.json. The pass criterion is fixed in
football_cv.evaluation.region_eval (coverage in [0.85, 0.95]; area < half the shape-only baseline up to 4 s).
"""
from __future__ import annotations

import argparse
import json
import logging

from football_cv import config as C
from football_cv.evaluation import region_eval as RE
from football_cv.evaluation import skillcorner as sc

log = logging.getLogger("eval_regions")
VARIANTS = {
    "motion+shape": {},
    "motion_only": {"regions": {"use_shape": False}},
    "shape_only": {"regions": {"use_motion": False}},
}
# (pitch-mapping offset simulated, calibration_sigma_m assumed by the regions)
SCENARIOS = ((3.0, 3.0), (0.0, 3.0), (5.0, 3.0), (5.0, 5.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--match-dir", default="data/external/skillcorner/1886347")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--period", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = C.load_config(args.config)
    match_dir = C.resolve(args.match_dir)
    table = sc.load_match(match_dir, int(match_dir.name))

    results = {"config": {"trials": args.trials, "seed": args.seed, "period": args.period, "match": match_dir.name,
                          "coverage_band": RE.COVERAGE_BAND, "regions_defaults": dict(RE.regions.DEFAULTS)}}
    for offset, assumed in SCENARIOS:
        for name, variant in VARIANTS.items():
            extra = {"calibration_sigma_m": assumed}
            merged = {"regions": {**(cfg.get("regions") or {}), **variant.get("regions", {}), **extra}}
            for gap in RE.GAP_LENGTHS_S:
                r = RE.run_gap_trials(table, merged, gap, args.trials, args.seed, offset, period=args.period)
                results[f"offset={offset:g}|assumed={assumed:g}|{name}|gap={gap:g}"] = r
                log.info("offset %.0f assumed %.0f %-13s gap %4.0fs  coverage %.3f  area %7.0f m2  centre err %5.1f m  (%d samples)",
                         offset, assumed, name, gap, r.get("coverage", float("nan")), r.get("median_area_m2", float("nan")),
                         r.get("median_centre_err_m", float("nan")), r.get("samples", 0))
    out = C.artifacts_dir(cfg) / "regions"
    out.mkdir(parents=True, exist_ok=True)
    (out / "skillcorner_region_eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
