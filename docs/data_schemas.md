# Data schemas

Tables written by the pipeline under `data/processed/<clip>/` (git-ignored). Times are seconds from
the start of the clip; positions are metres in the pitch frame (105 × 68 m, x = 0 at the image-left
goal of the main camera, y = 0 at the near touchline).

## `player_states.parquet`

One row per person identity and sample inside the identity's life span (first observation to last
observation plus the prediction horizon, per shot). Written by `reconstruction/reconstruct.py`.

| column | type | meaning |
|---|---|---|
| `sample_id` | int | sampled-frame index (5 fps); joins to `frames`, `shots`, `frame_calibration` |
| `timestamp_s` | float | clip time of the frame; the video and the pitch view share this clock |
| `identity_id` | int | person identity; equals `track_id` (no re-identification) |
| `track_id` | int | ByteTrack id |
| `team_id` | str | `home`, `away`, `other` or `unknown` (see `track_teams.csv` for confidence) |
| `shot_id` | int | camera shot; gaps are never bridged across shots |
| `x_m`, `y_m` | float | **published** position. NaN when the position is unknown or withheld |
| `x_est_m`, `y_est_m` | float | the model's estimate even when withheld; NaN if there is no calibrated view |
| `sigma_m` | float | 1-sigma per axis. Observed rows carry the measurement noise (1.0 m) |
| `position_source` | str | `OBSERVED`, `INTERPOLATED`, `PREDICTED`, `MANUALLY_ANNOTATED`, or empty (nothing published) |
| `state` | str | one of the seven states below |
| `visibility` | str | `visible` or `not_visible` |
| `confidence` | float | detector confidence for observed rows; `1 - sigma/max_sigma` for reconstructed rows; 1.0 for annotated positions |
| `calibration_status` | str | `keyframe`, `propagated`, `rejected:*`, `no_keyframe`, `not_wide_view`, `n/a` |
| `keyframe_dt_s` | float | seconds from the frame's calibration keyframe |
| `time_since_obs_s`, `time_to_next_obs_s` | float | distance to the neighbouring observations; the latter is NaN when there is none |
| `gap_s` | float | length of the surrounding gap (0 for observed rows) |
| `method` | str | `observed`, `rts`, `linear`, `cv`, `kalman`, `hold` or `manual` |
| `reason` | str | why a position is absent or withheld (uncalibrated frame, rejected jump, `max_gap_s`, `max_sigma_m`) |

### States

| state | assigned when |
|---|---|
| `VISIBLE_ON_PITCH` | observed, tracked and calibrated |
| `OCCLUDED` | missing, estimate's 2-sigma disc wholly inside the frame, gap ≤ `max_occlusion_s` |
| `OUTSIDE_CAMERA_VIEW` | missing, 2-sigma disc wholly outside the frame |
| `UNKNOWN_MISSING` | none of the above can be shown: uncalibrated frame, straddling the border, long in-view gap, rejected jump |
| `OFF_PITCH_INJURY_TREATMENT` | **manual annotation only** |
| `RETURNING_TO_PLAY` | **manual annotation only** |
| `SUBSTITUTED_CONFIRMED` | **manual annotation only** |

A gap, however long, never produces the last three. Substitution and injury cannot be told apart
from a tracker break by video alone.

### Annotation file (`reconstruction.annotations_file`)

JSON list; the file is optional.

```json
[{"identity_id": 7, "state": "OFF_PITCH_INJURY_TREATMENT", "start_s": 120.0, "end_s": 185.0,
  "x_m": 52.5, "y_m": -3.0, "note": "treated by physio"}]
```

`x_m`, `y_m` are optional; without them the position stays unknown. Annotations are applied only to
samples where the identity was not observed; an observation inside an annotated interval wins and is
reported as a warning. A state outside the three annotation-only states is rejected at load.

## `gaps.parquet`

One row per missing interval, including intervals that were *not* filled.

| column | meaning |
|---|---|
| `identity_id` | person identity |
| `start_s`, `end_s`, `duration_s` | interval and length |
| `kind` | `interior` (observations on both sides, same shot), `trailing` (after the last observation) or `across_shots` (camera cut or replay between two shots) |
| `state` | most frequent state inside the gap |
| `filled` | whether any position was published |
| `reason` | joined reasons why positions were withheld or never attempted |

## Other tables used by the new stages

- `frame_calibration.parquet`: `sample_id`, `calibration_status`, `keyframe_id`, `keyframe_dt_s`,
  `h00` … `h22` (image → pitch homography, row-major).
- `track_teams.csv`: `track_id`, `team_id`, `colour_distance`, `n_colour_samples`, and the new
  `agreement` (share of the track's torso crops whose nearest kit cluster matches its label) and
  `team_status` (`confident`, `uncertain`, `other`, `unknown`). `confident` needs agreement ≥ 0.8 and
  at least 5 crops. Labels are never changed by this check.
- `artifacts/<clip>/reconstruction/masking_eval.csv`: per gap length and method, windows, MAE with a
  95 % bootstrap interval, RMSE, median, p90, mean sigma and 95 % coverage.

## Interoperability

The tables are flat and per-identity-per-sample so they can be mapped to other formats:

| target | mapping | caveat |
|---|---|---|
| SoccerNet tracking / MOT | `(timestamp_s, identity_id, x_m, y_m)` for `OBSERVED` rows; image-space boxes stay in `trajectories` | no HOTA/IDF1 can be computed without labelled boxes |
| kloppy tracking dataset | one frame per `sample_id` at 5 fps, players from `team_id` and `identity_id`, pitch 105 × 68 m | a provider-agnostic export is not implemented; skip rows with empty `position_source` |
| ELASTIC / similar | positions plus `time_since_obs_s` as a mask | the interpolation in this repo is a motion model, not a learned one |
| SPADL / Atomic-VAEP / xT / PPDA | none: they need events | no valid action stream exists for the clip (`docs/evaluation.md` §7) |

Rules for any consumer: `INTERPOLATED` and `PREDICTED` positions must never be counted as observed;
distance, speed and acceleration must not be computed from reconstructed positions; rows with an
empty `position_source` have no position.
