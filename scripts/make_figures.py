"""Render the README result figures from the local evaluation files (no video frames are used).

    python scripts/make_figures.py [--config configs/default.yaml]

Reads artifacts/<clip>/{regions,identity,calibration_validation}/ and writes docs/images/*.png.
Every number plotted comes from those files; nothing is typed in here. The figures show failures as well
as passes: the linker misses its pre-registered target and is drawn that way.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from football_cv import config as C  # noqa: E402

GAPS = (1, 2, 4, 8, 15)
BLUE, GREY, RED, GREEN = "#1f5fbf", "#8a8a8a", "#c0392b", "#2e7d32"
DPI = 150


def _regions_figure(art: Path, out: Path) -> None:
    final = json.loads((art / "regions" / "skillcorner_region_eval.json").read_text())
    orig = json.loads((art / "regions" / "skillcorner_region_eval_original_design.json").read_text())
    new = [final[f"offset=3|assumed=3|motion+shape|gap={g}"] for g in GAPS]
    old = [orig[f"offset=3|motion+shape|gap={g}"] for g in GAPS]
    shape = [final[f"offset=3|assumed=3|shape_only|gap={g}"] for g in GAPS]
    lo, hi = final["config"]["coverage_band"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.axhspan(lo, hi, color=GREEN, alpha=0.12, label=f"pre-registered band {lo:g}-{hi:g}")
    a.axhline(0.9, color=GREEN, lw=0.8, ls=":")
    a.plot(GAPS, [r["coverage"] for r in old], "o--", color=RED, label="original design")
    a.plot(GAPS, [r["coverage"] for r in new], "o-", color=BLUE, label="final (sigma growth 0.15)")
    a.set(xlabel="hidden stretch (s)", ylabel="share of true positions inside the 90 % region",
          title="Coverage of the inferred region", ylim=(0.7, 1.0), xticks=GAPS)
    a.legend(loc="lower left", fontsize=8)

    b.plot(GAPS, [r["median_area_m2"] for r in new], "o-", color=BLUE, label="motion + formation (final)")
    b.plot(GAPS, [r["median_area_m2"] for r in shape], "s--", color=GREY, label="formation only (baseline)")
    b.set(xlabel="hidden stretch (s)", ylabel="median region area (m$^2$)", title="Region size", xticks=GAPS)
    b.legend(fontsize=8)
    fig.suptitle("Unseen players: held-out SkillCorner match (150 trials per point, detected positions only)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "regions_coverage.png", dpi=DPI)
    plt.close(fig)


def _linker_figure(art: Path, out: Path) -> None:
    res = json.loads((art / "identity" / "skillcorner_linker_eval.json").read_text())
    scen = [("reference", "reference\n(3 m offset, 5 % team noise)"), ("no_offset", "no mapping\noffset"),
            ("offset_5m", "5 m offset"), ("team_noise_15pct", "15 % team-label\nnoise")]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    for i, (key, _) in enumerate(scen):
        for j, (variant, colour, name) in enumerate((("default", BLUE, "conservative linker"),
                                                     ("greedy_baseline", GREY, "greedy nearest-successor"))):
            r = res[f"{key}|{variant}"]
            x = i + (j - 0.5) * w
            ax.bar(x, r["precision"], w, color=colour, label=name if i == 0 else None)
            ax.text(x, r["precision"] + 0.012, f"{r['precision']:.2f}\n{r['links']} links", ha="center", fontsize=7)
    ax.axhline(0.98, color=RED, ls="--", lw=1.2)
    ax.text(len(scen) - 0.55, 0.99, "pre-registered target 0.98", color=RED, fontsize=8, ha="right", va="bottom")
    ax.set(ylabel="share of links that are the same player", ylim=(0, 1.3),
           xticks=range(len(scen)), xticklabels=[s[1] for s in scen],
           title="Tracklet linking: FAILS its target (so it is off by default)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "linker_precision.png", dpi=DPI)
    plt.close(fig)


def _landmark_figure(art: Path, out: Path) -> None:
    df = pd.read_csv(art / "calibration_validation" / "landmark_error.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for status, colour in (("keyframe", GREEN), ("propagated", BLUE)):
        d = df[df.status == status]
        ax.scatter(d.keyframe_dt_s, d.mean_abs_m, color=colour, s=34, label=f"{status} ({len(d)} frames)")
    ax.set(xlabel="seconds from the nearest manual keyframe", ylabel="mean error at hand-labelled landmarks (m)",
           title="Pitch-mapping error, measured by hand (not by line agreement)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pitch_mapping_error.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    art = C.artifacts_dir(cfg)
    out = C.resolve("docs/images")
    out.mkdir(parents=True, exist_ok=True)
    _regions_figure(art, out)
    _linker_figure(art, out)
    _landmark_figure(art, out)
    # a pure top-down rendering (pitch graphics only, no broadcast pixels)
    shutil.copyfile(art / "reconstruction" / "play_frame_8.png", out / "topdown_states.png")


if __name__ == "__main__":
    main()
