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

`pytest`: 83 tests, all passing (geometry and homography 8, analytics 9, reconstruction 25, identity
linker 16, inferred regions 14, SkillCorner loader and scoring 11). They
test the maths on synthetic inputs; they do not validate the pipeline on video. The real-data numbers
for the linker and the regions are in section 11. The reconstruction
tests cover the motion models, view geometry, the missing-observation cases (interior gap, trailing
prediction, gap too long, uncalibrated frame, camera cut, rejected jump), uncertainty growth,
annotation rules (annotation-only states never come from a gap; observations are never
overwritten), team confidence, the masking harness (including a leakage check) and a renderer smoke
test.

## 9. Missing-position reconstruction

Protocol: `docs/evaluation_protocol.md` §4. Truth is the system's own smoothed calibrated
projection, **not** real pitch positions, so these are relative accuracies of the motion models.
75,140 scored samples; MAE in metres with a 95 % bootstrap interval over windows; `rts cov` is the
share of samples inside the method's own 95 % ellipse.

| gap | windows | linear | rts | cv | kalman | hold | rts cov |
|---|---|---|---|---|---|---|---|
| 0.6 s | 666 | **0.32** [0.30, 0.33] | 0.44 [0.40, 0.45] | 0.68 [0.63, 0.72] | 0.84 [0.79, 0.88] | 1.07 [1.01, 1.10] | 0.96 |
| 1 s | 549 | **0.41** [0.38, 0.43] | 0.50 [0.46, 0.52] | 0.96 [0.88, 1.00] | 1.08 [1.00, 1.14] | 1.39 [1.31, 1.46] | 0.94 |
| 2 s | 343 | 0.70 [0.64, 0.72] | 0.71 [0.64, 0.73] | 1.54 [1.39, 1.60] | 1.58 [1.44, 1.68] | 2.08 [1.93, 2.18] | 0.95 |
| 4 s | 177 | 1.24 [1.12, 1.34] | **1.04** [0.94, 1.13] | 3.05 [2.71, 3.36] | 2.92 [2.64, 3.16] | 3.33 [2.92, 3.57] | 0.96 |
| 8 s | 58 | 3.42 [3.05, 3.81] | **2.37** [2.08, 2.61] | 6.46 [5.64, 7.37] | 5.59 [4.94, 6.33] | 6.44 [5.56, 7.03] | 0.97 |

Reading it:

- Between two observations `linear` is best up to 1 s and tied with `rts` at 2 s; `rts` wins from
  4 s, and its reported uncertainty is calibrated (coverage 0.94–0.97 against a nominal 0.95). The
  other methods' sigmas are over-conservative (coverage ≈ 1.0, mean sigma several times the error).
  The default is `rts` because only it also carries a trustworthy sigma, which the publish/withhold
  rule needs.
- History-only, `cv` beats `kalman` clearly up to 1 s and ties it at 2 s (overlapping intervals).
  Both are only modestly better than `hold` beyond 2 s, with errors near 3 m at 4 s, which is why
  prediction stops at 2 s.
- The `cv` default was chosen on this same data (see the limitation in
  `docs/audit_and_findings.md`).

**Applied to the clip** (`artifacts/<clip>/reconstruction/reconstruction_summary.json`): 390
identities, 16,866 state rows, 622 gaps.

| source | rows | | state | rows |
|---|---|---|---|---|
| OBSERVED | 11,997 | | VISIBLE_ON_PITCH | 11,997 |
| PREDICTED | 2,379 | | OCCLUDED | 1,974 |
| INTERPOLATED | 536 | | OUTSIDE_CAMERA_VIEW | 399 |
| no position | 1,954 | | UNKNOWN_MISSING | 2,496 |

Failure analysis of the 4,869 non-observed rows (2,915 published, 1,954 withheld): 694 rows are on
frames rejected for being too far from a keyframe, 459 on non-wide views and 111 on a segment with no
keyframe; 588 were estimated but withheld because sigma exceeded 5 m (long predictions); 102 are
detections rejected as implausible jumps. Most withholding is therefore calibration coverage, not
the motion model. Of 622 gaps, 232 are interior (148 filled) and 390 trailing (316 filled, mostly
2 s predictions). No gap crosses a shot boundary because ByteTrack ids are not reused across shots;
the `across_shots` path is covered by a unit test only.

No injury, return or substitution exists in this clip; the three annotation-only states have never
been exercised on real data.

**Team confidence** (same clip): 152 away and 221 home tracks `confident`, 36 and 41 `uncertain`, 42
`other`, 162 `unknown`. Mean crop agreement 0.969; 37 of 450 tracks fall below 0.8.

## 10. Pitch-mapping error at held-out landmarks

Question: how many metres off is a projected position? Method (`scripts/landmark_error.py`, labels in
`configs/overlay_labels.json`): points that lie on a straight pitch marking (halfway line, touchlines,
penalty-area front line) were clicked by eye in 14 frames that were **not** used to fit anything for
that frame. Each point is mapped through the frame's homography; the error is its distance to the
marking, perpendicular to the line (metres). A point on a line only constrains one direction, so this
**understates** the 2-D error.

| frames | status | labelled points | mean error (m) | median frame (m) | worst frame (m) | worst point (m) |
|---|---|---|---|---|---|---|
| 2 | keyframe | 11 | 0.18 | 0.18 | 0.20 | 0.41 |
| 12 | propagated, 2–12 s from a keyframe | 54 | 4.06 | 3.86 | 8.10 | 17.98 |

Reading it:

- At keyframes the mapping is right to a few decimetres. On propagated frames it is typically about
  4 m off, and one frame in two has a mean over 4 m. Error is not clearly monotone in the distance to
  the keyframe (correlation of frame error with keyframe distance 0.26 on 12 frames).
- **This replaces the earlier statement "0.4 m median over 4 s".** That figure came from one
  keyframe pair (20 → 40) and from an automatic score comparing projected lines with detected white
  pixels (`calibration_validation/overlay_summary.json`: median 0.44 m on 884 propagated frames, but only
  58 % of probe points are inliers). The automatic score only sees line pixels it can match, so it
  cannot see a mapping that is shifted by several metres onto a neighbouring line.
- Consequence for everything downstream: a position in pitch metres on a propagated frame carries an
  error of roughly 4–6 m, which is larger than the reconstruction sigma. The linker and inferred regions
  therefore add a calibration term (`calibration_sigma_m`) that this section justifies.

Limits: 12 propagated frames, labelled by one person by eye, on straight markings only; frames were
picked where markings are visible, which are likely easier than average. Treat 4 m as an estimate with
a wide interval, not a bound. More keyframes (clicking landmarks in the worst frames) is the direct fix.

## 11. Identity linking and inferred regions, scored on public tracking data

Data: SkillCorner open data, match 1886347 (MIT licence, credit SkillCorner), 10 fps tracking of all
players including those outside the broadcast view. Files stay local (`data/external/`, git-ignored).
**Only positions flagged `is_detected` are used as truth**; the remainder is SkillCorner's own
extrapolation. Loader: `evaluation/skillcorner.py`. Period 1 was used for tuning, period 2 only for the
numbers below. Everything is seeded (seed 0); settings and raw results are written to
`artifacts/<clip>/{identity,regions}/skillcorner_*_eval.json`.

### 11.1 Tracklet linker

Harness (`scripts/eval_linker_skillcorner.py`, `evaluation/linker_eval.py`): 150 random 30 s windows. In
each, every player's detected stretches of at least 1 s become tracklets, degraded to broadcast
conditions (per-tracklet pitch-mapping offset of 3 m, noise, 5 % wrong team labels, ends at the view
boundary flagged). The linker sees only team, position and time; true identity only scores it. A link is
right if both tracklets belong to the same player; 1,020 true adjacent pairs exist in the reference run.
**Pass criterion, fixed before the first run:** link precision at least 0.98 in the reference scenario
(offset 3 m, 5 % team noise) with the default settings. Baseline: greedy nearest-successor linking.

| scenario | links | precision | recall of true continuations | impure identities | greedy: links / precision / recall |
|---|---|---|---|---|---|
| reference (offset 3 m, 5 % team noise) | 36 | **0.861** (31/36; 95 % Wilson interval 0.71 to 0.94) | 0.030 | 5 | 490 / 0.494 / 0.219 |
| no mapping offset | 41 | 0.902 | 0.036 | 4 | 503 / 0.598 / 0.283 |
| offset 5 m | 24 | 0.833 | 0.020 | 4 | 467 / 0.418 / 0.174 |
| 15 % team-label noise | 32 | 0.781 | 0.025 | 7 | 477 / 0.415 / 0.181 |

**Result: FAIL.** Precision 0.861 against a target of 0.98, and only about 3 % of true continuations are
found. Even with a perfect mapping (second row) precision is 0.90. The ambiguity refusal does what it is
for (greedy linking is right about half the time), but it cannot remove errors caused by team-mates who
are in no tracklet: an unseen player can fit a gap as well as the true one, and position and team do not
tell them apart.

Ablations on the same data (reference scenario): without the view-edge rule 74 links at 0.878; without
the unseen-team-mate cap 39 links at 0.846; with ambiguity margin 0, 34 links at 0.853. None approaches 0.98.

History, so the number is read correctly:

1. The first design (gates, ambiguity refusal, kit-colour veto) did not reach the target.
2. Two rules were then added and tuned on period 1: do not link across a view-edge break (an entry is a
   new person, not a continuation), and mark a lone candidate `weak` when the expected number of unseen
   team-mates that fit equally well exceeds 0.5. On period 1 this gave precision 1.0, but at only 23 links
   and 3.5 % recall.
3. On held-out period 2 the same settings gave 0.861. The period 1 figure was a small-sample selection
   effect. The held-out figure is the result.

Conclusions and limits:

- **Linking by team, position and a kit-colour veto is not reliable enough to switch on.** Team-mates
  share a kit, so appearance can only veto; what is missing is something that separates team-mates
  (jersey number or a learned re-identification embedding). `identity.apply` is therefore `false`, and a
  merge is never applied silently.
- On the broadcast clip the linker finds 604 candidate pairs: 12 linked, 79 ambiguous, 509 not best,
  4 weak; 529 tracks would become 517 identities. They are written to `artifacts/<clip>/identity/` but
  not used downstream.
- The kit-colour veto is untested against reality: SkillCorner has no video, so it is exercised only by
  unit tests and by the clip itself.
- A split-track check on the real clip (cut one long track in two, see whether it is re-joined) was
  contaminated by duplicate fragments of the same person and is inconclusive. No figure is claimed.
- The scenario is built from stretches the SkillCorner camera followed, so it is not identical to
  off-screen motion in the broadcast.

### 11.2 Inferred regions for unseen players

Harness (`scripts/eval_regions_skillcorner.py`, `evaluation/region_eval.py`): hide a stretch of one
player's detected positions (1, 2, 4, 8 or 15 s), give the system the observed team-mates and the
player's own positions before and after, a drifting pitch-mapping offset and noise, and score the 90 %
region against the hidden truth. 150 trials per cell, period 2, seed 0. **Pass criteria, fixed first:**
coverage within [0.85, 0.95] at every gap length, and for gaps up to 4 s a median area under half that of
the shape-only baseline (team centroid plus the player's usual offset, no motion).

| gap | original design: coverage | final: coverage | final: median area (m²) | shape-only area (m²) | centre error (m) |
|---|---|---|---|---|---|
| 1 s | 0.930 | 0.920 | 134 | 651 | 3.7 |
| 2 s | 0.877 | 0.880 | 137 | 651 | 3.9 |
| 4 s | 0.899 | 0.916 | 149 | 651 | 3.6 |
| 8 s | 0.859 | 0.899 | 211 | 651 | 4.4 |
| 15 s | **0.778** | 0.881 | 430 | 651 | 5.7 |

(Simulated mapping offset 3 m, regions assume 3 m.) **The original design failed** the coverage band at
15 s: the motion model's uncertainty grows too slowly over long gaps. One parameter was added,
`sigma_growth_per_m` (motion σ becomes σ(1 + g·σ)), chosen on period 1 only. The period 1 grid, coverage at
1 / 2 / 4 / 8 / 15 s:

| g | 1 s | 2 s | 4 s | 8 s | 15 s |
|---|---|---|---|---|---|
| 0.00 | 0.921 | 0.859 | 0.929 | 0.839 | 0.807 |
| 0.05 | 0.921 | 0.859 | 0.933 | 0.861 | 0.852 |
| 0.10 | 0.921 | 0.859 | 0.935 | 0.874 | 0.885 |
| 0.15 | 0.921 | 0.859 | 0.939 | 0.891 | 0.908 |
| 0.20 | 0.921 | 0.859 | 0.940 | 0.903 | 0.920 |

g = 0.15 was taken. On held-out period 2 it **meets both criteria**: coverage 0.88 to 0.92 at every
length, and area about a fifth of the shape-only baseline up to 4 s. The two columns come from runs of
different size (120 and 150 trials), so they are not paired.

What the formation prior buys: nothing measurable up to 4 s; at 15 s it cuts the area from 768 m²
(motion only) to 430 m² at similar coverage. Shape alone is calibrated (coverage 0.89 to 0.96) but its
area is constant at 651 m² and its centre is 6 to 8 m off.

Sensitivity to the pitch-mapping error, which section 10 shows to be large on propagated frames:

| mapping offset simulated | region assumes | coverage at 1 / 2 / 4 / 8 / 15 s | median area at 1 s (m²) |
|---|---|---|---|
| 0 m | 3 m | 1.00 / 1.00 / 1.00 / 0.98 / 0.94 | not recorded here |
| 3 m | 3 m | 0.92 / 0.88 / 0.92 / 0.90 / 0.88 | 134 |
| 5 m | 3 m | 0.55 / 0.52 / 0.65 / 0.67 / 0.75 | not recorded here |
| 5 m | 5 m | 0.93 / 0.88 / 0.92 / 0.92 / 0.89 | 366 |

If the mapping is worse than assumed the regions are overconfident, badly so at 5 m against 3 m. Section
10 measures about 4 m error on propagated frames, so the shipped default is `calibration_sigma_m = 5.0`,
which makes regions about 2.7 times larger at short gaps (366 against 134 m²). Only the 3 m / 3 m row was
used in the tuning protocol; the 5 m / 5 m row is a check on the shipped default, not a tuned result.

Limits:

- Truth exists only where SkillCorner detected the player. Those stretches favour play the camera
  followed, so real off-screen movement may be harder than the test.
- On the clip every track fragment gets a region, because there is no identity linking. A display rule
  (`plausible_unseen`: hide a region whose 50 % ellipse probably holds an observed team-mate, and keep at
  most 11 minus the observed count per team) thins them: 28,792 region rows over 390 identities, 11,102
  flagged, median area 858 m² (large: 5 m mapping error and long gaps). **That rule is not scored against
  ground truth.** A drawn region means "an unseen player of this team is probably around here", not
  "this player is here". Referees and goalkeepers labelled `other` are not counted.
- One match, one tracking provider, 150 trials per cell.
