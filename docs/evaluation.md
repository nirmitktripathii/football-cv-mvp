# Evaluation

What was measured on the MVP clip, how, and what it does and does not show. There is **no
ground truth** for this clip (no tracking data, no event data), so every figure below is either an
internal consistency check or a measurement against manually identified pitch landmarks. Nothing
here is an accuracy figure against an independent reference.

Clip: Bengaluru FC v Sporting Club Delhi, ISL 2026-27 GW1, full-replay seconds 3120-3480
(360 s, 1280×720, 30 fps), sampled at 5 fps → 1,800 frames.

## 1. Detection (YOLO11m COCO, FP32, imgsz 1280)

| | detections | frames with ≥1 | frame coverage | median confidence |
|---|---|---|---|---|
| person | 24,298 | 1,779 / 1,800 | 98.8 % | 0.78 |
| sports ball | 1,690 | 891 / 1,800 | 49.5 % | 0.22 |

- Coverage counts *all* frames, including replays and close-ups, where there may be no ball in view.
- Ball detections are low-confidence (median 0.22, threshold 0.10). Multiple ball candidates per
  frame (mean 1.9 when present) are common: heads, boots, ad-board circles. The trajectory stage keeps
  one per frame and flags implausible jumps; it is not a dedicated ball tracker.
- FP16 inference on the GTX 1650 returned NaN boxes. The invalid FP16 run is kept locally in
  `data/processed/<clip>/detections_fp16_invalid.parquet` as evidence, and the config pins `half: false`.

## 2. Shots and wide-view gating

17 shots by colour-histogram cut detection. Four of them (0, 7, 10, 15) are wide tactical views
(wide share ≥ 0.70); the other 13 are replays, close-ups and crowd shots and are never calibrated.
68–123 s is the replay sequence after the 1-0 goal; 301–335 s are close-ups.

## 3. Pitch calibration

Eleven manual keyframes (`configs/calibration_keyframes.json`), propagated by ORB camera-motion
chains within each motion segment.

**Fit residuals at the keyframes** (pitch metres, at the clicked points):

| keyframe | points | line points | manual RMS (m) | max error after refinement (m) |
|---|---|---|---|---|
| 20 | 5 | 3 | 0.34 | 0.65 |
| 40 | 5 | 3 | 0.07 | 0.54 |
| 140 | 5 | 0 | 0.35 | 1.53 |
| 290 | 10 | 0 | 0.91 | 2.99 |
| 680 | 5 | 0 | 0.40 | 1.33 |
| 795 | 10 | 0 | 0.46 | 1.48 |
| 920 | 9 | 0 | 0.32 | 0.85 |
| 1065 | 7 | 0 | 0.23 | 0.53 |
| 1160 | 7 | 2 | 0.20 | 0.49 |
| 1310 | 5 | 2 | 0.45 | 0.85 |
| 1450 | 6 | 0 | 0.63 | 1.09 |

These are in-sample residuals: they show the points are mutually consistent with one homography,
not that the clicks are right. A mislabelled landmark in an exactly determined fit (4 points) would
show zero error, which is why a keyframe tried at sample 1250 was removed.

**Propagation drift, out of sample.** Where two keyframes share a segment, the homography propagated
from the earlier one is evaluated at the later one's own clicked landmarks (`landmark_*`), and at a
fixed grid of image probe points (`probe_*`):

| from → to | Δt (s) | landmark median (m) | landmark max (m) | probe median (m) | probe max (m) |
|---|---|---|---|---|---|
| 20 → 40 | 4 | 0.42 | 0.62 | 0.57 | 1.02 |
| 1160 → 1310 | 30 | 10.22 | 16.80 | 14.59 | 17.02 |
| 1310 → 1450 | 28 | 4.26 | 19.10 | 7.76 | 16.17 |

Over a few seconds drift is sub-metre. Over a long pan (segment 13 sweeps from the right box to the
halfway line) it reaches 5–15 m. Line-chamfer refinement of propagated homographies was tried and
**did not** recover drift of that size (it locks onto the wrong lines), so it is not used on
propagated frames. Instead, frames more than `max_keyframe_dt_s = 12 s` from their nearest keyframe
are rejected. Within 12 s, expect errors of roughly 1–5 m, worse on the far side and at image edges.
This is an extrapolation from the three drift checks, not a measured bound.

**Frames by calibration status** (1,043 wide-view frames):

| status | frames |
|---|---|
| keyframe | 11 |
| propagated | 884 |
| rejected: too far from keyframe | 95 |
| rejected: visible grass projects far outside pitch | 14 |
| no keyframe (segment 14, different camera angle) | 39 |

Trajectories: 12,865 projected rows; 147 implied-speed jumps (> 11 m/s) are flagged and excluded
from speed and distance.

## 4. Tracking and teams

- ByteTrack (`supervision`) produced 654 track ids. Torso-colour k-means (k=2) assigned 262 to the
  home team, 188 to away, 42 to "other" (referees, goalkeepers) and left 162 "unknown" (too few or
  too small torso crops). After the minimum-length filter, 390 tracks remain in `track_summary.csv`.
- **Identities are not stable.** A track breaks at every camera cut and often at occlusions, so
  per-track distance (median 7.5 m) is a fragment, not a player's match distance. No
  re-identification or shirt-number reading is attempted.
- Team colours were spot-checked on the annotated broadcast video, not measured against labels.

## 5. Analytics

- **Team shape** (visible players only, ≥ 4 per team): median width 34.1 m (away) / 38.1 m (home),
  depth 21.7 / 20.2 m, convex hull 401 / 381 m². A broadcast shows part of the pitch, so these
  describe the visible block, not the whole team.
- **xT:** the grid lookup matches `socceraction.xthreat.ExpectedThreat.rate` on random actions
  (`tests/test_analytics.py`). Values on the clip are *ball-zone* xT, not action xT, because there
  is no pass/carry segmentation. They are labelled exploratory everywhere.
- **Direction of play** (home attacks left) is measured, not assumed: near each goal the deepest
  outfield player belonged to the defending team in 149 v 33 frames (left goal: away) and
  263 v 27 (right goal: home).
- **PPDA** is not computed: it needs defensive actions and opponent passes, which the CV stream does
  not contain.

## 6. Candidate attacking sequences

Five windows where the observed ball reaches a final third. All are rated **low** confidence by the
rule in `sequences/candidates.py` (ball coverage, keyframe distance, nearest-player proxy).

| id | t (s) | probable attack | note from review |
|---|---|---|---|
| SEQ1 | 47–61 | Bengaluru FC → left | ball reaches x ≈ −1 m at t 59.4, then the replay block starts at 68 s and the kick-off keyframe is at 136 s. Reviewed by eye on `broadcast_SEQ1.mp4`: at t 57.6 s the ball is in the left goal past the diving keeper with the scoreboard still 0-0, i.e. the 1-0. **The system does not detect goals**; this is a human check. |
| SEQ2 | 151–173 | Sporting Club Delhi → right | |
| SEQ3 | 220–245 | Sporting Club Delhi → right | |
| SEQ4 | 248–276 | Bengaluru FC → left | long ball from x ≈ 47 to 5 m at t 262–265; by t 270 a Sporting Club Delhi player has the ball at the edge of their box (likely cleared), matching the low proximity share (0.13) |
| SEQ5 | 276–301 | Sporting Club Delhi → right | |

## 7. Atomic-VAEP (public event data, not the clip)

`socceraction.atomic.vaep.AtomicVAEP`, XGBoost, 154 features, 3 previous actions, seed 0.
Trained on 760 games (Premier League + La Liga, StatsBomb open data; 1.71 M atomic actions),
tested on 64 held-out World Cup 2018 games (145 k atomic actions).

| label | AUROC | Brier | base rate | Brier of predicting the base rate |
|---|---|---|---|---|
| scores | 0.869 | 0.00799 | 0.0100 | 0.00991 |
| concedes | 0.871 | 0.00252 | 0.0028 | 0.00276 |

Both labels beat the base-rate Brier. The integration test rates one held-out game end to end; its
top-valued actions are that game's goals, as expected.

**On the CV clip: INCOMPLETE.** No valid SPADL action stream can be derived from this clip: there is
no reliable possession assignment, no pass/shot/tackle segmentation, and about half of the frames
have no usable ball. VAEP numbers from such a stream would be fabricated.

## 8. Tests

`pytest`: 17 tests, all passing (geometry and homography 8, analytics 9). They test the maths on
synthetic inputs; they do not validate the pipeline on video.
