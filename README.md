# football-cv-mvp

Broadcast football video → detected and tracked players and ball → 2D pitch coordinates (metres)
→ spatial analytics, exploratory xT, candidate attacking sequences, and an Atomic-VAEP model
trained on public event data. Built and evaluated on one real Indian Super League broadcast clip.

**Status: MVP.** Every output is derived from a single broadcast clip with no ground truth.
Provisional values are labelled as such in the files and the dashboard. Atomic-VAEP on the clip
is **INCOMPLETE** by design: see [Report](#report).

## Pipeline

```
ingest → detect → shots → motion → track → calibrate → reconstruct → analytics → visualize
                                       (atomic_vaep and reconstruction_eval run separately)
```

| stage | does | key output |
|---|---|---|
| ingest | probe and sample the video at 5 fps | `data/manifests/<clip>.json`, frames |
| detect | YOLO11m (COCO) people + ball, FP32, imgsz 1280 | `detections.parquet` |
| shots | cut detection, wide tactical view gate | `shot_table.csv` |
| motion | ORB camera motion between samples, motion segments | `motion_segments.csv` |
| track | ByteTrack people, best ball per frame, torso-colour teams | `track_teams.csv` |
| calibrate | manual keyframes → propagated homography → pitch metres, with rejections | `trajectories.parquet`, `calibration_report.json` |
| reconstruct | fills short gaps in person tracks with a motion model, with provenance, state and uncertainty; never across cuts or uncalibrated frames | `player_states.parquet`, `gaps.parquet` |
| analytics | team shape, ball proximity, ball-zone xT, candidate sequences | `artifacts/<clip>/analytics/` |
| visualize | annotated broadcast, top-down animation, state-aware top-down animation, per-sequence clips | `artifacts/<clip>/videos/` |
| atomic_vaep | socceraction Atomic-VAEP on StatsBomb SPADL stores | `artifacts/atomic_vaep/` |
| reconstruction_eval | masking evaluation of the gap-filling methods | `artifacts/<clip>/reconstruction/masking_eval.csv` |

Design and conventions: [docs/current_architecture.md](docs/current_architecture.md). Audit of the
v0.1 baseline (git tag `baseline-v0.1`): [docs/audit_and_findings.md](docs/audit_and_findings.md).
Table schemas: [docs/data_schemas.md](docs/data_schemas.md).

### Reading the state-aware top-down view

| marker | meaning |
|---|---|
| solid, team colour | observed: detected, tracked and calibrated |
| translucent, dashed edge, ring | interpolated between two observations; ring = 2-sigma |
| faint, dotted edge, ring | predicted from history only (at most 2 s); ring = 2-sigma |
| diamond | manually annotated position |
| `?` in a dashed ring | location unknown; estimate withheld, ring shows how uncertain it is |
| grey band below the touchline | off-pitch region, used by annotated players only |
| red border / `CAMERA CUT` | no calibrated view, or a camera cut; nothing is drawn |

Injury treatment, return to play and confirmed substitution are shown only when a manual annotation
file says so (`reconstruction.annotations_file`); a gap in tracking never produces them.

## Setup

Python 3.11–3.12, a CUDA build of PyTorch already installed (deliberately not a dependency, so pip
cannot swap it for a CPU wheel), and `ffmpeg` on `PATH`.

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
```

The clip is not in the repository (copyrighted broadcast). It is seconds 3120–3480 of the official
full replay, 1280×720 at 30 fps; for example:

```bash
yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --download-sections "*3120-3480" -o data/raw/isl_bfc_scd_2026-10-10_gw1_section.mp4 https://www.youtube.com/watch?v=gDI508P5F3g
```

YOLO11m weights download on first use. Atomic-VAEP needs the SPADL HDF5 stores from the sibling
`VAEP_Football` project (path in `configs/default.yaml`).

## Run

```bash
python -m football_cv.pipeline all
```

```bash
python -m football_cv.pipeline atomic_vaep
```

```bash
streamlit run app/streamlit_app.py
```

```bash
python -m football_cv.pipeline reconstruct
python -m football_cv.pipeline reconstruction_eval
```

```bash
python -m pytest
```

Any config value can be overridden, e.g. `--set calibration.max_keyframe_dt_s=8`. New calibration
keyframes are added with `scripts/annotate_keyframe.py` (draws candidate line intersections to
name) and `configs/calibration_keyframes.json`.

## Report

Clip: Bengaluru FC v Sporting Club Delhi, ISL 2026-27 GW1, 6 minutes from the official full replay
(1-0 goal, replays, kick-off, open play). Full numbers and method: [docs/evaluation.md](docs/evaluation.md).

**What works**

- Players are detected in 98.8 % of sampled frames (median confidence 0.78) and tracked within
  shots; teams are separated by kit colour (blue home, white away; referees and keepers as "other").
- 895 of 1,043 wide-view frames are mapped to pitch metres from 11 manual keyframes. Short-range
  propagation is accurate (0.4 m median over 4 s); frames where drift was measured to grow
  (5–15 m over ~30 s pans) are rejected rather than kept.
- Direction of play is measured from the data (home attacks left in this window).
- Team shape of the visible players: median width 34–38 m, depth 20–22 m.
- Missing positions are reconstructed and kept apart from observations: 11,997 observed rows,
  536 interpolated, 2,379 predicted, 1,954 withheld. On a masking test over 75,140 samples, Kalman +
  RTS recovers a hidden 4 s stretch to 1.0 m MAE (linear 1.2 m, last-position hold 3.3 m) with a
  well-calibrated uncertainty. The truth in that test is the pipeline's own smoothed projection, so
  it measures the motion model, not real pitch error ([docs/evaluation.md](docs/evaluation.md) §9).
- Team labels carry a confidence: 373 of the 450 home/away tracks are confident, 77 uncertain (labels unchanged).
- Five candidate attacking sequences are found and rendered. SEQ1 (47–61 s, Bengaluru FC towards
  the left goal) is the 1-0: on review of the rendered clip the ball is in the net at t 57.6 s. The
  system itself does not detect goals; that check was done by eye.
- Atomic-VAEP trained on 760 Premier League and La Liga games, tested on 64 World Cup 2018 games:
  AUROC 0.869 (scores) and 0.871 (concedes), Brier below the base-rate Brier for both.

**What does not (yet)**

- The ball: detected in about half the frames with low confidence; no dedicated ball tracker, and
  lofted balls project to the wrong place.
- Identity: tracks break at cuts and occlusions; per-track distances are fragments, not player totals.
  No re-identification is implemented (a wrong merge cannot be detected without identity labels);
  the design is in the roadmap.
- Injury, return to play and substitution states are implemented and unit-tested, but this clip has
  none, so they have never run on real data. Positions are never counted as observed when
  reconstructed, and distance and speed are not computed from them.
- Most withheld positions come from calibration coverage (frames too far from a keyframe, replays),
  not from the motion model.
- All five sequences are low confidence (ball gaps, distance from keyframes, weak possession proxy).
- xT on the clip is a zone value of the ball's position (exploratory), not action xT.
- **Atomic-VAEP on the clip: INCOMPLETE.** There is no valid SPADL action stream (no possession,
  no action segmentation), so no VAEP number is reported for the clip.
- **PPDA: not computed.** It needs defensive actions and passes.

What would change that: [docs/future_pipeline_roadmap.md](docs/future_pipeline_roadmap.md).
What was tried and why: [docs/research_notes.md](docs/research_notes.md).

## Repository layout

```
app/               Streamlit dashboard (read-only over artifacts)
configs/           pipeline config, manual calibration keyframes
docs/              architecture, audit, implementation plan, evaluation and protocol, schemas, research notes, roadmap
models/            xT grid (committed), YOLO weights (git-ignored)
scripts/           keyframe annotation helper
src/football_cv/   pipeline package
tests/             geometry, analytics and reconstruction unit tests (42)
data/, artifacts/  local only (git-ignored), except data/manifests
```

## Licences and attribution

- Source video: © the Indian Super League and its broadcaster. Not redistributed.
- Event data behind xT and Atomic-VAEP: [StatsBomb open data](https://github.com/statsbomb/open-data),
  used under its user agreement (attribution required).
- Detector: Ultralytics YOLO11, AGPL-3.0. socceraction: MIT.
