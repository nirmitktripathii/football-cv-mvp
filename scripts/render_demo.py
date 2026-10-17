"""Render one candidate sequence as a top-down animation with inferred regions (no broadcast pixels).

    python scripts/render_demo.py [--seq SEQ1] [--out docs/media/topdown_regions_SEQ1.gif]

Faster than the visualize stage (which renders every window and the full broadcast overlay): this draws
only pitch graphics for one window, so the result can be committed. Solid dots are observed players,
translucent dashed circles are interpolated or predicted positions, hatched patches are inferred regions
(not positions).
"""
from __future__ import annotations

import argparse

import pandas as pd

from football_cv import config as C
from football_cv.visualization import pitch_states


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seq", default="SEQ1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    run = C.run_dir(cfg)
    seq = pd.read_parquet(run / "sequences.parquet").set_index("seq_id").loc[args.seq]
    names = {"home": cfg["clip"]["home_team"], "away": cfg["clip"]["away_team"]}
    out = C.resolve(args.out or f"docs/media/topdown_regions_{args.seq}.gif")
    pitch_states.render_state_animation(
        pd.read_parquet(run / "player_states.parquet"), pd.read_parquet(run / "trajectories.parquet"),
        pd.read_parquet(run / "frames.parquet"), pd.read_parquet(run / "shots.parquet"),
        pd.read_parquet(run / "frame_calibration.parquet"), out, float(seq.start_s), float(seq.end_s), names,
        float(cfg["sampling"]["fps"]), title=f"{names['home']} v {names['away']}  {args.seq}",
        regions=pd.read_parquet(run / "inferred_regions.parquet"),
    )
    print(out)


if __name__ == "__main__":
    main()
