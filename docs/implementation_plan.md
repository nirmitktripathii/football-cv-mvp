# Implementation plan and status

The upgrade from v0.1 to the reconstruction-aware prototype, phase by phase: what was done, how it
was validated on the real clip, and what was deliberately not done. Rollback: `git checkout
baseline-v0.1` restores the original pipeline; the new stages are additive and `all` still runs the
old stages in the same order, with `reconstruct` inserted after `calibrate`.

| phase | goal | status | where | validated by |
|---|---|---|---|---|
| 1 | Audit and baseline | done | `docs/audit_and_findings.md`, tag `baseline-v0.1`, `artifacts/<clip>/baseline_v0.1/` | baseline outputs copied before any change |
| 2 | Calibration and homography | **kept**, not rewritten | `calibration/*` | existing keyframe, drift and coverage checks (`docs/evaluation.md` §3); the reconstruction refuses to place players on uncalibrated frames |
| 3 | Team classification | **kept**, confidence added | `tracking/teams.py` (`add_team_confidence`) | labels diffed against the previous run (unchanged); 37 of 450 tracks flagged by crop agreement |
| 4 | Tracking continuity and re-identification | **not implemented** | roadmap §3 | n/a; see below |
| 5 | Missing-position reconstruction | done | `reconstruction/` | unit tests on synthetic tracks; masking evaluation on the clip |
| 6 | State-aware renderer | done | `visualization/pitch_states.py` | frames inspected by eye (cut, uncalibrated and play frames); renderer smoke test |
| 7 | Evaluation | done where ground truth allows | `reconstruction/evaluate.py`, `docs/evaluation_protocol.md` | masking harness, 75,140 scored samples |
| 8 | Schemas | done | `docs/data_schemas.md` | column list copied from `STATE_COLUMNS` in `reconstruct.py` |

## Why re-identification was not built

A linker that merges tracklets needs a way to know when it is wrong. This clip has no identity
labels, and a wrong merge fuses two players into one trajectory without any visible symptom, which is
worse than the current fragmentation. The reconstruction already accepts an `identity_id` column, so
a linker can be added without changing the rest. The proposed design (gated by time, distance in
pitch metres, team and appearance, leaving ambiguous pairs unlinked) and its self-supervised
evaluation are in `docs/future_pipeline_roadmap.md` §3.

## Design decisions

- **Reconstruction is a separate stage and separate tables.** `trajectories.parquet` is untouched, so
  every existing analytic still sees observed data only.
- **Per identity, per shot, never across a cut.** A replay between two sightings is not a gap that can
  be interpolated.
- **State from geometry, not from gap length.** The 2-sigma disc of the estimate is projected into
  the frame. Injury, return and substitution come only from manual annotations.
- **Withhold rather than guess.** Beyond 8 s, or at sigma above 5 m, the published position is empty
  and only the estimate and its uncertainty are stored.
- **Defaults come from the masking evaluation:** `rts` between observations (lowest error from 4 s,
  calibrated uncertainty at every length); `cv` for history-only prediction capped at 2 s.
- **CPU only.** The reconstruction and renderer need no GPU; the GPU stays confined to detection.

## Configuration

`configs/default.yaml`, section `reconstruction`: `interpolate`, `predict`, `max_gap_s`,
`predict_horizon_s`, `max_occlusion_s`, `max_sigma_m`, `view_margin_px`, `image_size`,
`annotations_file`, `eval_gap_lengths_s`. Override on the command line with `--set
reconstruction.max_gap_s=6`.

## New dependencies

None. Everything uses the existing numpy, pandas, scipy, matplotlib, mplsoccer and Pillow stack.
