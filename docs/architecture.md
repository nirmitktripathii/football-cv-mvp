# Architecture

A staged batch pipeline. Each stage reads the previous stage's files and writes its own, so any
stage can be re-run alone (`python -m football_cv.pipeline <stage>`). All settings come from
`configs/default.yaml` (override with `--set key.sub=value`).

```
video (data/raw) ──ingest───▶ sampled frames + manifest
                  ──detect───▶ detections.parquet            (YOLO11m, person + sports ball)
                  ──shots────▶ shot_table.csv                (cuts, wide-view gate)
                  ──motion───▶ camera motion G_t per frame    (ORB + RANSAC, motion segments)
                  ──track────▶ tracks + team ids              (ByteTrack, torso-colour k-means)
                  ──calibrate▶ frame_calibration, trajectories.parquet (pitch metres)
                  ──analytics▶ ball xT, team shape, proximity, candidate sequences
                  ──visualize▶ annotated broadcast, top-down animation, per-sequence clips
atomic_vaep (run separately) ─▶ model + metrics on public StatsBomb SPADL data
app/streamlit_app.py ─────────▶ read-only dashboard over all of the above
```

## Modules (`src/football_cv/`)

| module | responsibility |
|---|---|
| `config.py` | YAML loading, overrides, run/artifact directories |
| `ingestion/video.py` | ffprobe metadata, 5 fps sampling, manifest |
| `detection/detector.py` | Ultralytics batch inference, FP32 (FP16 gives NaN boxes on GTX 16xx) |
| `calibration/shots.py` | histogram cut detection; wide view = green ratio, small players, enough people |
| `calibration/motion.py` | ORB matching between consecutive sampled frames → homography G_t, chained per segment |
| `tracking/tracker.py` | ByteTrack for people (ids unique across segments); ball = highest-confidence on-pitch detection per frame, `track_id = -1` |
| `tracking/teams.py` | k=2 kit clusters on torso colour; far from both centroids → "other" |
| `calibration/homography.py` | pitch model (105×68 m), normalised DLT with point-on-line equations, validation, line-chamfer refinement for keyframes |
| `calibration/projection.py` | keyframe fitting, propagation `H_t = H_k · G_k⁻¹ · G_t`, rejection rules, drift checks, bottom-centre projection |
| `trajectories/clean.py` | implied-speed jump flags, runs split at gaps, rolling median over observed samples only, speeds |
| `analytics/xt.py` | socceraction-convention xT grid lookups, attacking-frame mirroring, ball-zone threat |
| `analytics/spatial.py` | team shape (width, depth, hull, nearest-neighbour), nearest player to ball |
| `analytics/atomic_vaep.py` | Atomic-VAEP training/evaluation on SPADL stores; clip status INCOMPLETE |
| `sequences/candidates.py` | final-third ball windows → ranked candidates with confidence notes |
| `visualization/broadcast.py`, `pitch_animation.py` | annotated video; mplsoccer top-down animation and snapshots |
| `pipeline.py` | CLI and stage wiring |

## Pitch frame and conventions

- 105 × 68 m. x = 0 is the goal line on the image-left of the main camera, y = 0 the near touchline.
  These are reference dimensions; the stadium pitch is not surveyed.
- xT follows socceraction: `grid[w-1-yj, xi]` for a team attacking towards x = 105. For a team
  attacking left, positions are mirrored to `(105 - x, 68 - y)` first.
- Direction of play is a config value (`clip.home_attacks`) measured from the data
  (`docs/evaluation.md` §5).

## Calibration design

1. **Manual keyframes.** A human names landmarks (box and goal-area corners, circle extremes,
   halfway-line ends) among Hough-line intersection candidates drawn by
   `scripts/annotate_keyframe.py`, plus optional points that only lie *on* a straight marking
   (`on_line`), each adding one linear equation. Fit: normalised DLT. Validation rejects
   degenerate, mirrored or off-pitch transforms.
2. **Propagation.** Each sampled frame takes the nearest keyframe in its motion segment and
   composes the camera motion between them.
3. **Rejection.** A frame stays uncalibrated when it is not a wide view, its segment has no
   keyframe, it is more than 12 s from its keyframe (measured drift), or the grass visible in the
   image projects far outside the pitch. The reason is kept in `calibration_status`.
4. **Projection.** Every box (player or ball) by its bottom-centre. For an airborne ball this is
   wrong by the ball's height projected onto the ground.

## What is deliberately absent

- No interpolation across gaps: animations show only observed positions.
- No event detection, possession model or player identity.
- No PPDA (needs defensive actions and passes).
- No Atomic-VAEP on the clip (no valid action stream; `docs/evaluation.md` §7).

## Data and artifacts

`data/raw`, `data/processed` and `artifacts/` are git-ignored: the source video is copyrighted
broadcast footage, and everything derived from it stays local. The repository holds code, configs,
manual keyframes, the xT grid (`models/xt_model.json`), tests and docs.
