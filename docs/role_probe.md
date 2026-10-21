# Role probe: can frozen embeddings label player / goalkeeper / referee?

Motivation: kit colour cannot tell a goalkeeper from a referee (both wear yellow or orange), and the detector's
role calls are wrong often enough that referees ride inside player tracks and keepers are tagged as players.
A small classifier on frozen appearance embeddings is a cheap way to ask whether the information is in the crop
at all. This is a probe, not a pipeline change.

Stage 1 (this document): SRITrack embeddings, which the pipeline already computes. Stage 2, only after the
verdict and only with the user's go-ahead for the model download: SigLIP, scored on exactly the same split and
the same criteria.

## Declared before the run

**Data.** Every `person` box of the hand-audited windows a1, a2, a3 (`data/annotations/inter_napoli_fb4_<w>`),
cropped from the 5 fps frames of `data/processed/inter_napoli_fb4f5_<w>`. Ground-truth frame `k` is source frame
`first + 5k`. Boxes smaller than 8 px on a side are dropped and counted. `ignore` boxes (13) are not scored.
Classes: `player`, `goalkeeper`, `referee`.

**Features.** Primary: SRITrack `fc` (512-d, the Re-ID neck), input 256x128, checkpoint sha256 logged.
Secondary, reported but not allowed to change the verdict: `cls`, `mean`. Choosing the best of three after
seeing results would be tuning on the test windows, so the primary is fixed here.

**Classifier.** L2 logistic regression, `C = 1.0`, `class_weight="balanced"`, features standardised on the
training folds, `random_state = 0`. No tuning of any kind.

**Split.** Leave one window out (3 folds). Metrics are pooled over the three held-out folds, per box.

**Baselines on the same boxes and folds.**
- `kit+size`: the 9-d shirt/shorts/socks Lab descriptor (`appearance.band_colours`) plus log box height and
  width/height ratio, same classifier. This is "what colour and shape alone can do".
- `pipeline`: the role the current tracker gives the best-IoU (>= 0.5) predicted box from
  `inter_napoli_fb4f5r_<w>/tracks.parquet`, on the ground-truth boxes that have such a match.

**Pass criteria (all must hold for the primary `fc` probe):**

| # | Check | Pass |
|---|-------|------|
| E1 | pooled held-out role accuracy | >= 0.95 |
| E2 | referee recall and precision | both >= 0.80 |
| E3 | goalkeeper recall and precision | both >= 0.80, and in window a2 at most 10% of goalkeeper boxes predicted as referee |
| E4 | adds value | pooled accuracy beats `kit+size` by >= 0.02 and beats `pipeline` (on the matched subset) by >= 0.02 |

Reference points already measured: the pipeline's role accuracy on matched boxes is 0.918 and its referee
recall 0.342 (`eval_rate_*` on these windows).

**Verdict rule.** Pass only if E1-E4 all hold. Anything else is reported as a fail with the numbers; a near miss
is not rounded up.

## Limits stated up front

- Same match and the same people in every window. Leave-one-window-out stops a crop leaking between folds but
  not a person: the probe may learn *this* referee and *this* keeper. That is legitimate for per-match enrolment
  and says nothing about a new match.
- Ground-truth boxes are cleaner than the detector's. Occlusion and box jitter will make the real figure lower.
- No class for non-person detections: corner flags and the camera rig are not annotated, so this probe cannot
  speak to them. Only 13 `ignore` boxes exist.
- Three windows, 153 frames, one keeper visible per frame. Intervals are wide, and per-box intervals are
  optimistic because boxes within a track are not independent.

## Result: SRITrack `fc` probe (one run, nothing tuned) - FAIL on E3

3,487 ground-truth person boxes (2 dropped under 8 px); 3,306 matched a pipeline box. Leave-one-window-out,
pooled. `artifacts/role_probe/probe_sritrack.json`, script `scripts/probe_roles.py`.

| # | Check | Result | Verdict |
|---|-------|--------|---------|
| E1 | accuracy >= 0.95 | 0.9748 (always-"player" = 0.8707) | pass |
| E2 | referee recall and precision >= 0.80 | 0.9765 / 0.9864 | pass |
| E3 | goalkeeper recall and precision >= 0.80; a2 keeper->referee <= 10% | recall **0.6078**, precision 0.8158; a2 keeper->referee 1 of 51 | **fail** (recall) |
| E4 | beats `kit+size` and `pipeline` by >= 0.02 | 0.9748 vs 0.6349; 0.9779 vs 0.9186 on the matched subset | pass |

Verdict: **fail**, because goalkeeper recall is 0.61. 57 of 153 keeper boxes are called players, mostly in the
corner window a2 (35 of 51, recall 0.29; a1 recall 1.00, a3 0.53). Referee-for-keeper confusion is small (3 of 153),
so the embedding does separate the two; the miss is keeper vs ordinary player.

Secondary features (not allowed to change the verdict): `mean` 0.9779 accuracy, referee 0.977/0.980, keeper
0.647/0.892; `cls` 0.9679, referee 0.977/0.904, keeper 0.588/0.874. Same keeper shortfall.

Reading the numbers:
- Referee is the clear win: the probe finds 97.7% of referee boxes at 98.6% precision on held-out windows, against
  10.9% recall (precision 1.0) for the pipeline's track role on the matched boxes.
- The `kit+size` baseline is below the always-"player" rate because `class_weight="balanced"` trades accuracy for
  rare-class recall; beating it says little. The pipeline comparison is the meaningful one.
- Keeper recall is probably limited by training data, not the embedding: when a window is held out the probe sees
  about 100 keeper boxes from the other two windows. That is a hypothesis; it was not tested.
- The per-window referee recall (1.00, 0.99, 0.96) and keeper recall (1.00, 0.29, 0.53) are three points each.

Not tested here: SigLIP (needs a model download, awaiting the user's go-ahead), predicted rather than
ground-truth boxes, a new match, non-person boxes (corner flag, rig).

---

# Stage 1b: track-level voting and the detector's class (declared before the run)

Stage 1 failed E3 on goalkeeper recall. Two cheap ideas, no download: a track's boxes should agree on a role, and
the detector's own class is evidence the embedding does not have. Same data, split, features (`fc`), classifier
and baselines as above; the stage-1 numbers are the reference ("S0").

**Variants.**
- `S1` voted probe: for every predicted pipeline track (`inter_napoli_fb4f5r_<w>/tracks.parquet`) in the held-out
  window, average the probe's class probabilities over the ground-truth boxes that matched it (best IoU >= 0.5),
  and give that average to every one of those boxes. Boxes with no matching track keep their own probability.
  Voting runs on predicted tracks, not ground-truth tracks, so identity switches smear votes exactly as they would
  in the pipeline.
- `S2` voted probe plus detector (the primary for the verdict): a second logistic regression (same settings) on
  `[voted probabilities (3), share of the track's rows the detector called goalkeeper, share it called referee]`,
  the shares taken over all rows of the track in that window (`detector_role`). The stacker is trained on the two
  training windows with inner cross-fitting (each training window's probabilities come from a probe trained on the
  other training window only), so it never sees a probability the probe produced on its own training data.

**Pass criteria for `S2`** (E1-E3 as in stage 1, E4 changed to ask whether voting and the detector add anything):

| # | Check | Pass |
|---|-------|------|
| F1 | pooled accuracy | >= 0.95 |
| F2 | referee recall and precision | both >= 0.80 |
| F3 | goalkeeper recall and precision | both >= 0.80, and in window a2 at most 10% of goalkeeper boxes predicted as referee |
| F4 | adds value | goalkeeper recall at least 0.05 above S0 (0.608), pooled accuracy not more than 0.005 below S0, and pooled accuracy at least 0.02 above `pipeline` on the matched subset |

`S1` is reported as an ablation and cannot change the verdict. If `S2` fails, the next step is SigLIP, and only with
the user's go-ahead for the download.

**Extra limits.** Predicted tracks are the pipeline's `fb4f5r` tracks, tuned for this clip's referee fix, so the
detector shares are as good as that pipeline's; the stacker has 2 windows x ~51 keeper frames to learn from, and
the inner probes are trained on a single window, so they are weaker than the outer probe.

## Result: stage 1b (voting + detector) - FAIL on F3 and F4

One run, nothing tuned. `scripts/probe_roles_vote.py`, `artifacts/role_probe/probe_sritrack_vote.json`. 3,306 of 3,487
boxes matched a pipeline track; the rest kept their box-level probability and detector shares of 0.

| Variant | accuracy | referee R / P | goalkeeper R / P | a2 keeper->referee |
|---------|----------|---------------|------------------|--------------------|
| S0 box probe (stage 1) | 0.9748 | 0.977 / 0.986 | 0.608 / 0.816 | 1 of 51 |
| S1 voted (ablation) | 0.9753 | 0.973 / 0.997 | 0.516 / 0.952 | 0 of 51 |
| **S2 voted + detector (verdict)** | 0.9538 | 0.977 / 0.980 | **0.360 / 0.491** | 0 of 51 |
| pipeline, matched subset | 0.9186 | 0.109 / 1.000 | 0.507 / 1.000 | - |

F1 pass (0.954), F2 pass, F3 **fail** (keeper recall 0.36, precision 0.49), F4 **fail** (keeper recall is 0.25
*below* S0, not 0.05 above; pooled accuracy is 0.021 below S0). Voting raised keeper precision and lowered recall:
the misses are consistent across a track, not noise. Adding the detector's share made it worse: the stacker
trained on one-window probes learned to distrust the probe on keepers.

**Why: the keepers are two different looks.** Keeper recall by ground-truth keeper (stage 1 probe):

| window | keeper | kit | recall |
|--------|--------|-----|--------|
| a1 | 12 | yellow | 1.000 |
| a3 | 18 | yellow | 0.917 |
| a2 | 15 | near-black | 0.294 |
| a3 | 24 | near-black | 0.185 |

The probe recognises yellow keepers and does not recognise near-black ones: that keeper is dressed like the home
outfield players, and the only examples are 78 boxes from two tracks in two different windows, so whichever is held
out leaves one to learn from. This is read from four keeper tracks and three windows; it is a strong pattern, not a
proven mechanism. Referee, in a distinct neon kit, is solved by the embedding either way.

What it means for SigLIP: a different embedder would still face a keeper who looks like his teammates at
15-40 px, with the same two training tracks. The data limit is more likely the binding constraint than the
embedder, though that is untested.

---

# Stage 1c: pitch-position prior for goalkeepers (declared before calibration and before the run)

Stage 1b showed the embedding finds yellow keepers and misses the near-black one, who is dressed like the home
outfield players. Colour cannot fix that; where he stands can: a keeper is the person nearest his own goal line.
The referee labeller stays the SRITrack `fc` probe (leave-one-window-out, stage 1).

**Needs pitch coordinates for the ground-truth boxes.** Window a1 lies inside the calibrated 3-minute clip
(`poc_seg1re/frame_calibration.parquet`, joined on `frame_id`). Windows a2 and a3 have no calibration, so they are
hand-calibrated with the same landmark workflow as the 3-minute clip (`scripts/annotate_keyframe.py`, landmarks read
by eye, 3 keyframes per 30 s window, files `configs/inter_napoli_a2_keyframes.json` and `..._a3_...`). Calibration
is done and checked (landmark residuals, no use of roles) before the prior is run; the prior is not adjusted
afterwards. Calibration error of a few tenths of a metre is expected and is small against the thresholds below.

**The prior (fixed, no tuning).** Position = projected bottom-centre of the ground-truth box, in pitch metres
(105 x 68). Candidates in a frame: boxes on the pitch (0 <= x <= 105, 0 <= y <= 68) that the stage-1 probe does not
call referee. For each goal line separately, the candidate nearest to it fires as goalkeeper if its distance to
that line is <= 16.5 m (the penalty-area depth). At most one per end, so a frame showing both ends can have two.

**Variants** (same boxes and folds as stage 1):
- `P0` prior alone: goalkeeper if the prior fires, otherwise the stage-1 probe's call with goalkeeper mapped to
  player. Ablation.
- `P1` (verdict): goalkeeper if the prior fires **or** the stage-1 probe says goalkeeper; referee from the probe;
  otherwise player.

**Pass criteria for `P1`:**

| # | Check | Pass |
|---|-------|------|
| G1 | pooled accuracy | >= 0.95 |
| G2 | referee recall and precision | both >= 0.80 (unchanged by construction; checked) |
| G3 | goalkeeper recall and precision | both >= 0.80 pooled, and recall >= 0.60 on each near-black keeper (a2 gt 15, a3 gt 24) |
| G4 | adds value | goalkeeper recall at least 0.05 above the stage-1 probe (0.608) and pooled accuracy at least 0.02 above `pipeline` on the matched subset |

If a window cannot be calibrated well enough (any keyframe with a landmark residual above 1.5 m after the
pipeline's refinement), that window is reported as not scored and the verdict says so; it is not replaced by a
looser check.

**Limits.** The windows were calibrated by hand by the same person who defined the rule; a keeper standing outside
the penalty area (sweeping) or an attacker nearer the goal line than the keeper (a corner at the near post) is a
failure by design and will show up as a miss. One keeper per end per frame is assumed. The prior knows nothing about
which goal belongs to which team; it does not need to. Three windows, four keeper tracks: intervals are wide.

**Amendments, written after calibration and before any prior was run** (nothing was scored yet):
1. The windows are 10 s (51 samples), not 30 s, so each was calibrated with 2 hand-clicked keyframes (samples 0 and 50,
   `configs/inter_napoli_a2_keyframes.json`, `..._a3_...`), chained with the pipeline's optical-flow motion. Calibration
   result: all keyframe landmark residuals <= 0.47 m (limit 1.5 m, passed); a2 drift check 0.17 m median, a3 0.76 m
   median / 1.29 m max. Roles were not consulted. a1 uses `poc_seg1re` as declared.
2. User instruction (2026-10-21): boxes beyond the pitch boundaries do not matter. All of G1-G4 are therefore scored
   on ground-truth boxes whose projected foot point is on the pitch (0 <= x <= 105, 0 <= y <= 68). Off-pitch boxes
   (coaches, substitutes, ball boys) are excluded from the prior's candidates and from every denominator; their count is
   reported. The baselines (stage-1 probe and `pipeline`) are re-scored on the same on-pitch subset so the G4 comparison
   stays like for like.

## Stage 1c result (scripts/probe_roles_position.py, artifacts/role_probe/probe_position.json)

One run, no tuning. Calibrated: 3487 of 3487 boxes; on the pitch: 3330 (a1 1122, a2 1080, a3 1128); 157 off-pitch boxes
excluded (a2 77, a3 80). Statuses `keyframe`, `propagated` and `propagated_unrefined` were all accepted (a3 has 26 of 51
frames unrefined, about 1 m drift; against a 16.5 m threshold that is immaterial). This was decided before scoring but
is a change from `project_tracks`, which drops unrefined frames.

| Variant | accuracy | keeper recall / precision | referee recall / precision |
|---|---|---|---|
| stage 1 probe | 0.9748 | 0.608 / 0.823 | 0.983 / 0.978 |
| P0 prior alone | 0.9805 | 0.765 / 0.824 | 0.983 / 0.978 |
| **P1 prior or probe** | **0.9799** | **0.863 / 0.759** | 0.983 / 0.978 |
| `pipeline` (matched, on pitch) | 0.9286 | 0.507 / 1.000 | 0.030 / 1.000 |
| P1 (same matched subset) | 0.9840 | 0.870 / 0.799 | 1.000 / 0.976 |

Verdict for P1: **G1 pass** (0.980 >= 0.95). **G2 pass** (0.983 / 0.978). **G3 FAIL**: keeper recall 0.863 passes
and both near-black keepers pass (a2 gt 15: 0.294 -> 0.745; a3 gt 24: 0.185 -> 0.778), but keeper precision is 0.759
against the 0.80 bar. **G4 pass** (recall +0.255 over the probe; accuracy +0.055 over `pipeline`). The criteria
needed all four, so the experiment fails on one check, by 0.04 on precision.

Where the 42 false keepers come from (all true players, none referees):
- 25 are the prior firing in the a2 corner: a defender standing on the goal line (median 0.3 m from it) is nearer the
  line than the keeper, who stands a little in front. This is the near-post failure the Limits paragraph predicted; a
  GT keeper is present in every one of those 25 frames, so the prior picked the wrong person, not an empty goal.
- 17 are the stage-1 probe alone (a1 15, a3 2): outfielders it calls keeper. The prior does not cause these.
- a1 and a3 prior false positives: 0.

Reading: position clearly fixes what colour could not (the dark keepers go from 0.29 / 0.19 to 0.75 / 0.78 recall), but
"nearest to the goal line" is the wrong tie-break during a corner. Not run, and not to be run without a new declared
stage: choosing among the candidates near the goal by the probe's keeper probability, or requiring the keeper to be
the one nearest the goal centre rather than the line. Doing it now would mean tuning against the three windows.
Limits: three windows, four keeper tracks, one corner; calibration was hand-made by the same person who wrote the rule.

---

# Stage 1d: goal-centre distance with the probe's keeper probability as tie-break (declared before the run)

Requested by the user after stage 1c failed G3 on precision (0.759). Cause found there: during the a2 corner a defender
on the goal line is nearer the *line* than the keeper, who stands a little in front. Rule changes only the choice
among candidates; calibration, on-pitch filtering, usable statuses, folds and the referee labeller are unchanged.

**The rule (fixed now, one run, no tuning).** Per frame and per end, candidates are the on-pitch boxes the stage-1
probe does not call referee that lie within the penalty area of that end (x within 16.5 m of the goal line and
|y - 34| <= 20.16 m). Let d be a candidate's distance to the goal centre ((0, 34) or (105, 34)). The keeper is the
candidate with the smallest d; if other candidates have d within 2.0 m of that smallest value, the one with the highest
out-of-fold probe probability of "goalkeeper" among them is chosen instead. At most one per end per frame. The 2.0 m
tie window is a guess made after seeing the stage 1c failure but before seeing any centre distances.
The probe probability is the same leave-one-window-out logistic regression as stage 1 (`fc`, C=1, balanced, seed 0),
`predict_proba` instead of `predict`.

**Variants.** `Q0`: prior alone (otherwise player, referee from the probe). `Q1` (verdict): prior **or** the stage-1
probe says goalkeeper, as `P1` in stage 1c so the two are comparable.

**Pass criteria for `Q1`:** the same G1-G4 as stage 1c, on the same on-pitch boxes: G1 accuracy >= 0.95; G2 referee
recall and precision >= 0.80; G3 keeper recall and precision >= 0.80 pooled and recall >= 0.60 on a2 gt 15 and a3 gt 24;
G4 keeper recall >= stage-1 probe + 0.05 and accuracy >= `pipeline` + 0.02 on the matched on-pitch subset.

**If it fails, the line stops:** no further tie-break variants are tried on these three windows (that would be fitting
the rule to them); the next step would be a larger audited set or SigLIP, and the user decides. If it passes, it is a
result on three windows, four keeper tracks and one corner, and the rule was shaped after seeing that corner.

## Stage 1d result (scripts/probe_roles_centre.py, artifacts/role_probe/probe_centre.json) - run once, as declared

| Variant | accuracy | keeper recall / precision | referee recall / precision |
|---|---|---|---|
| stage 1 probe | 0.9748 | 0.608 / 0.823 | 0.983 / 0.978 |
| 1c `P1` (nearest the goal line) | 0.9799 | 0.863 / 0.759 | 0.983 / 0.978 |
| `Q0` prior alone | 0.9943 | 0.915 / 0.986 | 0.983 / 0.978 |
| **`Q1` prior or probe** | **0.9892** | **0.935 / 0.867** | 0.983 / 0.978 |
| `pipeline` (matched, on pitch) | 0.9286 | 0.507 / 1.000 | 0.030 / 1.000 |
| `Q1` (same matched subset) | 0.9908 | 0.932 / 0.877 | 1.000 / 0.976 |

Verdict for `Q1`: **PASS on all four.** G1 0.989 >= 0.95. G2 0.983 / 0.978 >= 0.80. G3 keeper recall 0.935 and precision
0.867 >= 0.80; hard keepers a2 gt 15 0.961 and a3 gt 24 0.778 (bar 0.60); the other two are a1 gt 12 1.000, a3 gt 18
0.917. G4 keeper recall +0.327 over the probe (bar +0.05); accuracy +0.062 over `pipeline` (bar +0.02).

False keepers by source: the prior 2 (both a2), the probe alone 20 (a1 15, a2 3, a3 2). So the prior itself is now
nearly clean; the precision cost of `Q1` is the stage-1 probe's own keeper calls. `Q0` (prior alone) scores higher on
accuracy and precision (0.994, 0.986) and loses recall only on the a3 gt 24 keeper and a few frames. `Q0` was not the
declared verdict variant, and choosing it now after seeing the numbers would be a post-hoc pick, so it is reported, not
adopted; if wanted it needs its own declaration on new data.

Limits, unchanged and sharpened: the rule was reshaped after seeing the a2 corner, so a2's gain (0.745 -> 0.961 for gt 15)
is partly in-sample; a3 gt 24 (0.778) and a1 gt 12 were not the reason for the change and are the cleaner evidence. Three
windows, four keeper tracks, one corner; boxes are ground truth, not detector output (a keeper missing from the detections
cannot be labelled); calibration was hand-made, and the prior needs a calibrated frame, so it does nothing on frames with
no homography (in the pipeline, close-ups and replays). Not wired into the pipeline: it is an offline probe on GT boxes.

## Stage 2: the probe and the position prior on detector boxes (declared 2026-10-21, before any run)

Requested by the user: wire the SRITrack referee probe and the goal-centre keeper prior (`Q1`) into the pipeline behind a
config flag, and declare criteria on **detector** boxes before running. Stage 1d used ground-truth boxes; this is the first
test on what the pipeline actually outputs.

**What is wired.** A new stage `roles` (`python -m football_cv.pipeline roles --config ...`) runs after `calibrate` and
before `identity`. It is off unless `role_probe.enabled: true`. For every person box of `tracks.parquet` it embeds the crop
(SRITrack `fc`), takes the probe probabilities (player / goalkeeper / referee), and projects the box foot through the frame
calibration (usable statuses: keyframe, propagated, propagated_unrefined). Rule, identical to `Q1`:

- referee if the probe says referee; the original detector role is kept in `role_pipeline`;
- per frame and per goal end, among on-pitch (0-105 m, 0-68 m, no margin) non-referee boxes within 16.5 m of the goal line
  and |y - 34| <= 20.16 m, the box nearest the goal centre is the keeper; any box within 2.0 m of that minimum competes on the
  probe keeper probability;
- goalkeeper if the prior fires or the probe says goalkeeper (`role_probe.keeper_rule: prior_or_probe`; `prior_only` is the
  `Q0` ablation, reported but not the verdict);
- otherwise player.

Frames with no usable calibration, boxes under 8 px on a side, and `ball` rows keep their detector role. Roles are written
to `tracks.parquet`, `tracks_projected.parquet` and `trajectories.parquet`; `team_id` follows `teams.apply_roles` (referee ->
`other`, goalkeeper -> `unknown`). The row is labelled per box, not per track: no voting (stage 1b rejected it).

**Models.** Logistic regression (C=1, balanced, standardised, seed 0) on the cached GT-box SRITrack `fc` embeddings of the
audited windows. For window W the model is trained on the **other two windows only**, so no box of W is in its training
set. Saved as `models/role_probe/lowo_<W>.npz` (no pickle).

**Runs.** Clip id `inter_napoli_fb4f5p_<W>`, started from the `inter_napoli_fb4f5r_<W>` run (same detector, same tracker, same
tracks, outlier runs and goalkeeper gate), with fresh `calibrate` (a1 gets a one-keyframe file from the PoC keyframe at
t = 600 s; a2 and a3 use the keyframes of stage 1c) and then `roles`. Baseline for every comparison: the `f5r` tracks, same
matched pairs.

**Scoring.** `scripts/eval_roles_detector.py`: per frame, Hungarian match of GT person boxes to output person boxes at
IoU >= 0.5; keep pairs whose output box is on pitch by its own projected foot; pool a1-a3. Reported two ways: *matched*
(isolates labelling; this is the verdict view, comparable to the stage 1d table) and *end to end* (unmatched GT boxes count
as misses, unmatched output boxes labelled keeper or referee count as false calls).

**Pass criteria (verdict variant `Q1`; all must hold, matched view, pooled):**

| # | Check | Pass |
|---|---|---|
| H1 | role accuracy | >= 0.95 and >= the `f5r` role accuracy on the same pairs + 0.02 |
| H2 | referee recall and precision | both >= 0.80 |
| H3 | keeper recall and precision | both >= 0.80; per-keeper recall >= 0.60 for the two hard keepers (a2 gt 15, a3 gt 24) |
| H4 | tracking untouched | HOTA, IDF1 and ID switches of the `p` tracks equal the `f5r` values exactly (roles must not change boxes or ids) |
| H5 | teams not worse | `team_outfield` correct share within 0.01 of `f5r` or better; changed referee rows carry `other` |
| H6 | coverage reported | share of person boxes on frames without usable calibration, and of boxes under 8 px, reported (no threshold) |

**If it fails:** report which checks and by how much. No new rule or tie-break variant is tried on these three windows (that
would be fitting to them). Next steps would be a larger audited set or SigLIP, and the user decides. `Q0` and the end-to-end
view are reported as context and cannot change the verdict.

**Limits stated in advance.** The probe is trained on ground-truth crops and applied to detector crops (domain shift is the
point of the test). The prior parameters (16.5 m, 20.16 m, 2.0 m) were fixed after seeing these same windows in stage 1c/1d,
so the position part is in-sample on geometry even though the probe weights are leave-one-window-out. Three 10 s windows, four
keeper tracks, one match, one corner. Calibration for a1 is one hand-clicked keyframe chained 10 s with LK.

### Stage 2 result (run 2026-10-21/22; criteria above were not changed)

Runs: `inter_napoli_fb4f5p_<W>` from the `f5r` tracks, fresh `calibrate`, then `roles` with the leave-one-window-out model
for that window (`models/role_probe/lowo_<W>.npz`). Scored by `scripts/eval_roles_detector.py`
(`artifacts/role_probe/eval_detector.json`). Matched view, on pitch by the box own projected foot, pooled a1-a3:
3,240 pairs of 3,298 matched (58 off pitch excluded; all matched boxes had a usable calibration).

| # | Check | Result | Verdict |
|---|---|---|---|
| H1 | role accuracy | Q1 0.9886; `f5r` 0.9284 (+0.0602) | PASS |
| H2 | referee recall / precision | 1.000 / 0.976 (`f5r`: 0.030 / 1.000) | PASS |
| H3 | keeper recall / precision | 0.925 / 0.844 (`f5r`: 0.507 / 1.000); per keeper: a1 gt12 1.00, a2 gt15 0.933, a3 gt18 0.913, a3 gt24 0.778 (`f5r`: 1.00, 0, 1.00, 0) | PASS |
| H4 | tracking untouched | boxes and ids identical row for row; HOTA 0.8574, IDF1 0.8599, IDSW 89 for both | PASS |
| H5 | teams not worse | `team_outfield` per box 0.9848 vs 0.9846 (`f5r`); 233 rows changed to referee, 0 without team `other` | PASS |
| H6 | coverage (no threshold) | 0 matched boxes without usable calibration; 9 matched boxes under 8 px; 2 rows had no crop and kept the detector role; 58 matched pairs off pitch | reported |

Context, not part of the verdict:

- **Q0 (prior only, no probe keeper calls):** accuracy 0.9929, keeper recall 0.904 / precision 0.943, per keeper a3 gt24 0.667. Q1 has
  more recall on the hardest keeper and lower precision; the verdict variant stays Q1 as declared.
- **End to end:** unmatched output boxes (not matching any annotated person) labelled keeper or referee on pitch rose from 3 (`f5r`)
  to 26 (a1 2, a2 17, a3 7). These may be coaches, substitutes or ball boys the annotation does not cover, or false calls; this
  was not audited. The matched view cannot see them.
- Per window Q1 accuracy and the confusion tables are in the JSON.

Limits that still apply: three windows, one match, four keeper tracks; probe is leave-one-window-out but the prior geometry (16.5, 20.16,
2.0 m) was fixed after seeing these windows; calibration for a1 is one clicked keyframe. A held-out window is the next real test.

Implementation notes: a first scorer run mis-indexed rows after `mot.drop_ignored` reset the index (accuracy 0.80, wrong); fixed
and rerun before any number above was read. The a3 `roles` process also died once without output and was rerun with the same
config, no code change.

## Audit of the 26 unmatched keeper/referee calls (stage 2)

Scope: the 26 on-pitch output boxes that matched no ground-truth box (IoU >= 0.5) and were labelled goalkeeper or
referee after the roles stage (a1 2, a2 17, a3 7). I looked at a context crop of each and checked its best IoU with the
audited boxes in the same frame (list: `artifacts/role_probe/audit/extra_calls.csv`, crops in the same folder).
Visual classification by one reviewer, no second annotator: treat the sub / photographer split as likely, not certain.

| Class | Boxes | Detail | Role label correct? |
|---|---|---|---|
| Camera rig on a pole (not a person) | 9 | a2, one fixed spot at about (22 m, 48-52 m), tracks 22/24/27, called keeper by the probe (p 0.51-0.99); outside the penalty area, so the prior never fired | No, not a person |
| Photographer or camera operator crouched at the touchline | 4 | a1 s35/s36 (referee, goalkeeper), a3 s25/s26 (referee) | No, staff |
| Substitute in a lime bib | 7 | a2 tracks 21 (3 rows, already referee before this stage), 43, 44, 45, 46; three are only a head at the bottom frame edge | No, the bib colour resembles the referee kit |
| Assistant referee on the touchline | 1 | a3 s34 | Yes (the ground truth does not label them) |
| Duplicate or poorly fitting box on a true person | 3 | a2 s4 t11: true keeper, IoU 0.57 with the GT keeper but another box won the match; a3 s39/s40: true referee, IoU 0.16 / 0.20 | Yes, box is redundant |
| Partial person cut by the bottom frame edge | 2 | a3 s47/s48 t33, head and white shirt only, called keeper | Unverifiable, most likely wrong |

Totals: 4 correct labels (1 assistant referee, 3 duplicate boxes on true keeper/referee), 13 real people with a wrong
or unverifiable role (4 staff, 7 substitutes, 2 edge fragments), 9 non-person false positives. Before this stage the
same count was 3 (the three substitute rows on track 21), so 23 are new: 9 rig, 4 photographers, 4 substitutes,
1 assistant referee, 3 duplicates, 2 edge fragments.

What it means:
- The prior produced 1 of the 26 (the true keeper, a2 t11). The probe alone produced 25, and 12 of them are goalkeeper
  calls outside what the prior accepts (the rig and the two edge fragments, plus the a1 photographer). That is the
  `prior_or_probe` rule being too permissive away from the goal; the stage-1d warning ("the probe alone produced the
  false keepers") applies here too.
- None of the 26 touches a scored box, so H1-H6 are unaffected. They would corrupt anything counted per label
  (keeper tracks, referee count, per-role distance).
- Possible fixes, **not applied** (each needs criteria declared first, and tuning them on these same windows would be
  in-sample): require a probe-only keeper call to lie within a distance of a goal; ignore boxes touching the frame
  edge or under a minimum height for the role relabel; suppress the camera rig as a non-person class or by its fixed
  position; NMS between a role box and a box it overlaps. Note that `prior_only` (Q0) removes the 12 probe-only
  goalkeeper calls but costs keeper recall (0.904 vs 0.925; a3 gt24 0.667 vs 0.778).

## Stage 3: zone and edge guards on the role relabel (criteria declared before running)

Motivation: the audit above. 24 of 26 extra keeper/referee calls came from the probe alone, not the position prior.
Disclosure: the thresholds below were chosen after seeing those 26 boxes (rig at 21-22 m from a goal line, 17-19 px
photographers, substitutes cut by the bottom frame edge). They are fixed now at the single values written here and
are **not** swept; a failure is reported as a failure. The rig, photographers and substitutes are all in these three
windows, so this test cannot show that the rules generalise; it only shows what they cost on the scored boxes.

Rules (new config keys, all off by default, run as `keeper_rule: prior_or_probe_in_area`):
- G, zone: a goalkeeper call that rests on the probe alone (prior did not fire) is accepted only when the box lies
  inside a penalty-area rectangle (same `penalty_depth_m` / `penalty_half_width_m` as the prior, no new number).
  Otherwise the row is a player.
- E, edge: a person box whose raw detector box touches the frame edge (`edge_margin_px: 2`) is not relabelled and keeps
  its detector role.
- S, short: a person box under `min_height_px: 20` is not relabelled and keeps its detector role.
Referee calls outside E/S are unchanged (no zone rule on referees: they legitimately roam the pitch).

Test: copy the three stage 2 run dirs, rerun only the `roles` stage with these keys (same leave-one-window-out probes,
same tracks, same calibration), score with `scripts/eval_roles_detector.py` on detector boxes, matched, on pitch.
Pass only if **all** hold (pooled over the three windows, 3,240 pairs):
- N1 (the point): unmatched on-pitch keeper/referee calls <= 10 (stage 2: 26; the 4 correct ones in the audit may stay).
- N2 (keeper): recall >= 0.90 (stage 2: 0.925) and precision >= 0.844 (stage 2 value, no loss); no single keeper track
  loses more than 0.05 recall against stage 2 (a1 gt12 1.00, a2 gt15 0.933, a3 gt18 0.913, a3 gt24 0.778).
- N3 (referee): recall >= 0.98 (stage 2: 1.000) and precision >= 0.970 (stage 2: 0.976).
- N4 (overall): role accuracy >= 0.980 (stage 2: 0.9886).
- N5 (unchanged): boxes and ids identical to stage 2 (H4), HOTA/IDF1/IDSW identical, changed-to-referee rows still carry
  team "other" (H5).
- N6 (cost, reported): matched boxes left unrelabelled by E or S, as a share of matched pairs; the pass line is <= 3%.
If N1 passes but N2-N4 fail, the guards are rejected as written, and the next step is a different rule, not a looser
threshold on these windows.

### Stage 3 result: guards REJECTED as declared (N2 per-track clause fails)

Run: `configs/inter_napoli_fb4_f5q_a{1,2,3}.yaml` on copies of the stage 2 run dirs (`inter_napoli_fb4f5q_<window>`),
`scripts/eval_roles_detector.py` with `ROLE_RUN=inter_napoli_fb4f5q`, scored on detector boxes, matched, on pitch
(3,240 pairs). Thresholds were not changed after the run.

| Criterion | Declared | Stage 2 | Stage 3 | |
|---|---|---|---|---|
| N1 unmatched on-pitch keeper/referee calls | <= 10 | 26 | **6** | pass |
| N2 keeper recall / precision | >= 0.90 / >= 0.844 | 0.925 / 0.844 | 0.911 / 0.930 | pass |
| N2 per keeper track, drop <= 0.05 | a1 gt12 1.00, a2 gt15 0.933, a3 gt18 0.913, a3 gt24 0.778 | | 1.00, 0.933, 0.957, **0.667** | **FAIL** (a3 gt24 -0.111) |
| N3 referee recall / precision | >= 0.98 / >= 0.970 | 1.000 / 0.976 | 0.994 / 0.982 | pass |
| N4 role accuracy | >= 0.980 | 0.9886 | 0.9923 | pass |
| N5 boxes, HOTA, IDF1, IDSW, referee team | identical | 0.8574, 0.8599, 89 | identical, 0 referee rows without "other" | pass |
| N6 matched boxes not relabelled (E or S) | <= 3% | 2 of 3,298 | 26 of 3,298 (0.8%) | pass |

Why it fails: a3 gt24 is a real keeper standing 17-21 m from the right goal line (pitch x 84-88 m, y 31 m), outside
the 16.5 m penalty rectangle. Nine of its boxes are probe-only keeper calls; the zone rule turns six of them back into
players (21 -> 18 of 27). The camera rig sits 22-26 m from the goal line, so no distance or zone rule separates the rig
from this keeper: the rig needs its own treatment (non-person class or fixed-position suppression), not a tighter
goal-distance gate. The prior-only ablation (Q0) also scores 0.667 on that track.

What the guards did get right: the 6 remaining extra calls are all either already present before this stage (3 rows of
the substitute on a2 track 21, referee from the detector) or correctly labelled (the a2 keeper duplicate, the a3
assistant referee, one duplicate box on the a3 referee). No false camera-rig, photographer or substitute call remains
that the probe introduced, and aggregate accuracy rose from 0.9886 to 0.9923. That is in-sample: the thresholds were
set after viewing these boxes.

Decision: defaults stay as they are (guards off, `keeper_rule: prior_or_probe`). Options for the next step, each needing
new criteria: (a) accept the guards but relax N2's per-track clause (a decision about the criterion after seeing the
result, so it would need a held-out window to count); (b) keep only the edge/height rules (E, S), which fix the
substitutes, photographers and edge fragments, and handle the rig separately; (c) keep prior_or_probe and treat the rig
as a detector problem.

## Stage 3b: edge and height guards only (criteria declared before running)

Decision after stage 3: drop the zone rule G (it removed real keeper boxes for a3 gt24, who stood 17-21 m out). Keep
E (`edge_margin_px: 2`) and S (`min_height_px: 20`) with `keeper_rule: prior_or_probe` unchanged. The camera rig is
**not** addressed here: it is a detector problem (a person class fires on a non-person), see `pipeline_status.md`.
Disclosure: the stage 3 run already contained E and S together with G, so their combined effect on the scored boxes
is partly known (26 of 3,298 matched boxes unrelabelled, mostly by E/S). The thresholds are the same single values as
before, not swept. This run only separates E/S from G.

Test: copies of the stage 2 run dirs (`inter_napoli_fb4f5s_<window>`), roles stage only, same leave-one-window-out probes,
scored on detector boxes, matched, on pitch (3,240 pairs). Pass only if **all** hold:
- M1 (the point): of the 23 new extra calls in the audit, these 11 are gone: photographers a1 s35/s36 and a3 s25/s26,
  substitutes a2 s11 t43, s14 t44, s15 t45, s16 t46, edge fragments a3 s47/s48 t33, short box a3 s39 t34; and total
  unmatched on-pitch keeper/referee calls <= 15 (26 minus those 11). The rig calls (9) are expected to remain.
- M2 (keeper): recall >= 0.90 and precision >= 0.844 (stage 2: 0.925 / 0.844); per keeper track recall within 0.05 of
  stage 2 (a1 gt12 1.00, a2 gt15 0.933, a3 gt18 0.913, a3 gt24 0.778).
- M3 (referee): recall >= 0.98 and precision >= 0.970 (stage 2: 1.000 / 0.976).
- M4: role accuracy >= 0.985 (stage 2: 0.9886).
- M5: boxes and ids identical, HOTA/IDF1/IDSW identical to stage 2, changed-to-referee rows carry team "other".
- M6 (cost, reported): matched boxes not relabelled, <= 3% of matched pairs.
A fail is reported as a fail; no threshold is changed afterwards.

### Stage 3b result: edge and height guards PASS (all six criteria)

Run: `configs/inter_napoli_fb4_f5s_a{1,2,3}.yaml` (edge_margin_px 2, min_height_px 20, `keeper_rule: prior_or_probe`) on
copies of the stage 2 run dirs; scored with `ROLE_RUN=inter_napoli_fb4f5s scripts/eval_roles_detector.py`
(`artifacts/role_probe/eval_detector_inter_napoli_fb4f5s.json`). Thresholds were not changed after the run.

| Criterion | Declared | Stage 2 | Stage 3b | |
|---|---|---|---|---|
| M1 the 11 targeted calls gone; total unmatched on-pitch keeper/referee calls | all 11 gone; <= 15 | 26 | all 11 gone; **11** | pass |
| M2 keeper recall / precision | >= 0.90 / >= 0.844 | 0.925 / 0.844 | 0.932 / 0.845 | pass |
| M2 per keeper track (within 0.05 of stage 2) | | 1.00, 0.933, 0.913, 0.778 | 1.00, 0.933, 0.957, 0.778 | pass |
| M3 referee recall / precision | >= 0.98 / >= 0.970 | 1.000 / 0.976 | 0.994 / 0.982 | pass |
| M4 role accuracy | >= 0.985 | 0.9886 | 0.9886 | pass |
| M5 boxes, HOTA 0.8574, IDF1 0.8599, IDSW 89, referee rows carry "other" | identical | | identical, 0 without "other" | pass |
| M6 matched boxes not relabelled | <= 3% | 2 | 26 of 3,298 (0.8%) | pass |

The remaining 11 unmatched on-pitch calls: 3 already present before this stage (substitute on a2 track 21, detector
referee), 3 with a correct label (a2 keeper duplicate box, a3 assistant referee, a3 duplicate box on the referee) and
**5 camera-rig boxes** (a2 tracks 24, 22, 27; the other 4 rig boxes are under 20 px high and are now left as players).
The rig is a detector problem and stays open. The stage 3 variant that added the zone rule G is rejected (see above).

Caveats: in-sample (the two numbers were set after viewing these boxes), three windows of one match, no held-out
window. The guards drop role probing for boxes that touch the frame edge or are under 20 px tall: they keep their
detector role, so a referee or keeper at the frame edge or far away is never corrected. 26 matched boxes (0.8%) are
in that state here, and the 3% line is the limit. Decision: defaults for `role_probe.edge_margin_px` and `min_height_px`
set to 2 and 20 (the stage stays off unless `role_probe.enabled`).
