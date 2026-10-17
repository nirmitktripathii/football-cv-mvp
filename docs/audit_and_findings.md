# Audit and findings

Audit of the v0.1 prototype (git tag `baseline-v0.1`, commit `b3de7a4`) done before any upgrade
work, and what the upgrade changed. Numbers are from the one ISL clip (Bengaluru FC v Sporting Club
Delhi, 360 s, sampled at 5 fps = 1,800 frames). There is no ground truth for this clip, so every
figure is an internal consistency check unless stated otherwise (see `docs/evaluation_protocol.md`).

Baseline outputs were copied, not overwritten, to `artifacts/<clip>/baseline_v0.1/` (local,
git-ignored). The original observed-only renderer (`visualization/pitch_animation.py`) is unchanged.

## 1. What already worked and was kept

| area | status in v0.1 | action |
|---|---|---|
| Detection (YOLO11m, FP32, imgsz 1280) | person 98.8 % frame coverage; ball 49.5 % | kept unchanged |
| Shot / wide-view gating | 17 shots, 4 wide-view | kept unchanged |
| Calibration (manual keyframes + ORB propagation) | 884 propagated + 11 keyframe of 1,043 wide-view frames | kept unchanged; validated, not rewritten |
| ByteTrack tracking | 654 ids, 390 after the length filter | kept unchanged |
| Team labels (torso-colour k-means) | 262 home / 188 away / 42 other / 162 unknown tracks | labels kept identical; confidence added |
| Analytics, xT, candidate sequences, dashboard, Atomic-VAEP on public data | working | kept unchanged |

## 2. Findings

Severity is the effect on downstream metrics, not the effort to fix.

| # | finding | evidence | severity | what was done |
|---|---|---|---|---|
| F1 | No model of missing observations at all. A player who is occluded, outside the camera frame or on an uncalibrated frame simply vanishes from the top-down view. | `docs/architecture.md` "No interpolation across gaps"; renderer reads observed rows only | high | new `reconstruction/` package: evidence-gated reconstruction with provenance, state and uncertainty |
| F2 | Gaps are short but not negligible: median 1.3 s, p90 4 s. | 622 gaps over 390 identities | medium | motion models chosen per gap length by masking evaluation |
| F3 | Position drift under long pans reaches 5–15 m; frames > 12 s from a keyframe are already rejected. | `docs/evaluation.md` §3 | high | unchanged; the reconstruction refuses to invent coordinates on rejected frames and reports `calibration_status` and `keyframe_dt_s` per row |
| F4 | Team labels carry no confidence. 37 of 450 tracks have < 80 % of their crops agreeing with their label (mean crop agreement 96.9 %). | `team_stability.json` | medium | `agreement` and `team_status` columns added (confident / uncertain / other / unknown); labels unchanged |
| F5 | No re-identification: a track breaks at every cut and many occlusions, so one player is many ids. | 654 ids for about 22 people on the pitch | high | **not fixed**; see §4 |
| F6 | Ball is projected by its bottom-centre, which is wrong for an airborne ball. | `docs/architecture.md` | medium | not changed; ball rows are not reconstructed |
| F7 | `research_notes.md` stated "no interpolation to fill gaps in animations". | grep | low | updated: reconstruction exists, and is visually distinct from observation |
| F8 | An earlier audit pass (commit `eeaa839`) fixed crashes, wrong valuations and deployment problems. | git log | n/a | not repeated here |

## 3. Defects the obvious approach would introduce, avoided on purpose

- **Gap length alone must not decide a state.** A long gap could be an injury, a substitution or a
  tracker break. `OFF_PITCH_INJURY_TREATMENT`, `RETURNING_TO_PLAY` and `SUBSTITUTED_CONFIRMED` are
  assigned only from a manual annotation file; a gap never produces them
  (test `test_no_gap_ever_produces_an_annotation_only_state`).
- **Smoothed positions leak.** `x_s` is a rolling median, so masking one sample and scoring it
  against `x_s` leaks neighbouring values. The masking harness hides one guard sample on each side
  of the scored window.
- **Cuts are not gaps.** A player seen before and after a replay is not one observation run;
  intervals across shots are listed in `gaps` (`kind = across_shots`, `filled = False`) and are never
  bridged.

## 4. Known limitations that remain

- **No re-identification.** `identity_id` equals `track_id`. `reconstruct()` accepts a pre-linked
  `identity_id` column, but no linker is shipped: without ground truth a linker's merges cannot be
  verified, and a wrong merge silently fuses two players. Design and evaluation plan are in
  `docs/future_pipeline_roadmap.md` §3.
- **Truth in the masking evaluation is the system's own smoothed projection**, not a surveyed
  position. The numbers show how well a motion model recovers a tracked, projected path, including
  its calibration jitter. They are not metric errors against real pitch positions.
- **The `cv` default for history-only prediction was chosen on the data it is reported on.** At
  0.6 s `cv` beats `kalman` clearly (MAE 0.68 vs 0.84 m, non-overlapping 95 % intervals); at 2 s the
  intervals overlap (1.54 vs 1.58 m). Treat the ranking at the longer end as indicative.
- **No annotated injury, return or substitution exists for this clip**, so those states are
  implemented and unit-tested on synthetic annotations only.
- **One clip, one stadium, one camera.** Nothing here generalises beyond it.
