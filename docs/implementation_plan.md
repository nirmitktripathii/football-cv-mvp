# Implementation plan and status

The upgrade from v0.1 to the reconstruction-aware prototype, phase by phase: what was done, how it
was validated on the real clip, and what was deliberately not done. Rollback: `git checkout
baseline-v0.1` restores the original pipeline; the new stages are additive and `all` still runs the
old stages in the same order, with `reconstruct` inserted after `calibrate`.

The tactical-camera upgrade (Inter–Napoli, YOLOX/ByteTrack/SRITrack) is planned separately in
[implementation_plan_tactical.md](implementation_plan_tactical.md).

| phase | goal | status | where | validated by |
|---|---|---|---|---|
| 1 | Audit and baseline | done | `docs/audit_and_findings.md`, tag `baseline-v0.1`, `artifacts/<clip>/baseline_v0.1/` | baseline outputs copied before any change |
| 2 | Calibration and homography | **kept**, not rewritten | `calibration/*` | existing keyframe, drift and coverage checks (`docs/evaluation.md` §3); the reconstruction refuses to place players on uncalibrated frames |
| 3 | Team classification | **kept**, confidence added | `tracking/teams.py` (`add_team_confidence`) | labels diffed against the previous run (unchanged); 37 of 450 tracks flagged by crop agreement |
| 4 | Tracking continuity and re-identification | linker built, **off by default**; unseen players shown as inferred regions | `tracking/identity.py`, `tracking/appearance.py`, `reconstruction/regions.py` | SkillCorner open tracking, `docs/evaluation.md` §11: linker did **not** meet its precision target; regions meet coverage up to 8 s |
| 5 | Missing-position reconstruction | done | `reconstruction/` | unit tests on synthetic tracks; masking evaluation on the clip |
| 6 | State-aware renderer | done | `visualization/pitch_states.py` | frames inspected by eye (cut, uncalibrated and play frames); renderer smoke test |
| 7 | Evaluation | done where ground truth allows | `reconstruction/evaluate.py`, `docs/evaluation_protocol.md` | masking harness, 75,140 scored samples |
| 8 | Schemas | done | `docs/data_schemas.md` | column list copied from `STATE_COLUMNS` in `reconstruct.py` |

## Why re-identification is not switched on

A linker that merges tracklets needs a way to know when it is wrong, and this clip has no identity
labels. A public tracking dataset with labelled identities (SkillCorner open data) was used instead:
tracklets were cut from its detected positions, degraded to broadcast-like noise, and linked, so
precision and recall are measured. The linker (gated by team, shot, time, pitch distance, a kit-colour
veto, refusal on any second plausible partner, and a cap on the expected number of look-alike unseen
team-mates) did **not** reach the pre-registered precision of 0.98 on held-out data
(`docs/evaluation.md` §11). It ships, tested, with `identity.apply: false`. The reconstruction accepts
the resulting `identity_id` column, so enabling it later needs no other change.

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
