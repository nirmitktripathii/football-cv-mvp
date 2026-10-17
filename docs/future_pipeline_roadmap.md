# Future pipeline roadmap

Ordered by what unblocks the most. Each step names the measurement that would show it worked.

## 1. Calibration without manual keyframes

- Train or adopt a pitch-keypoint / line-segmentation model (e.g. the SoccerNet camera-calibration
  baselines, or a keypoint head trained on broadcast frames) to produce landmarks every frame.
- Use camera-motion propagation only between detections, never across 30 s pans.
- **Measure:** landmark reprojection error on hand-clicked frames held out from several matches and
  stadiums; share of wide-view frames calibrated (today 86 %, 895 of 1,043).

## 2. Ball

- Fine-tune a detector on football broadcast ball annotations; add a dedicated ball tracker with a
  motion model and an in-air flag (a bottom-centre ground projection is wrong for lofted balls).
- **Measure:** ball precision and recall on labelled frames; ball coverage within candidate windows
  (today 36–69 % of calibrated frames).

## 3. Identity

- Appearance re-identification across occlusions and cuts; jersey-number OCR; team-sheet linking.
- **Measure:** ID switches and IDF1 against a labelled tracking sequence (e.g. SoccerNet tracking).
- **Tracklet linker (designed, not built).** Candidate pairs are tracklets of the same team with a
  gap of at most a few seconds inside one shot, whose reconstructed estimate (`reconstruct()` already
  gives mean and sigma) lies within 2-3 sigma of the next tracklet's first position, plus an
  appearance distance on torso crops. A pair is linked only if it is the unique mutual best match
  with a margin; any pair with a second plausible partner stays unlinked and is reported as
  ambiguous. Output is an `identity_id` column that `reconstruct()` already accepts.
- **Measure before shipping:** on labelled tracking data, the rate of wrong merges (must stay near
  zero, since a wrong merge is invisible) and recall of true links. On this clip only the
  self-supervised check in `docs/evaluation_protocol.md` §5 is possible.
- Across a camera cut a person cannot be linked by position at all; only appearance or a team sheet
  can do it.

## 3b. Reconstruction (shipped; next steps)

- Annotate a clip that contains an injury or a substitution and run the annotation-only states on
  real data; none exists in the current clip.
- Replace the constant-velocity prior with a context-conditioned model (ball, teammates, role) and
  report it against the masking baselines in `docs/evaluation.md` §9.
- Evaluate against surveyed tracking data (e.g. a public tracking dataset replayed through the same
  masking harness) so errors are against real positions, not the system's own projection.
- Export `player_states` as a kloppy tracking dataset (`docs/data_schemas.md`, Interoperability).

## 4. Events → SPADL → Atomic-VAEP on video

- Possession from ball–player proximity plus ball kinematics; action spotting for pass, shot,
  dribble and tackle (SoccerNet action-spotting models are a starting point).
- Convert to SPADL with socceraction, then rate with the Atomic-VAEP model already trained here.
- **Measure:** event precision and recall against a match with public event data; agreement of
  video-derived VAEP with event-data VAEP for the same match. Only then report VAEP for clips.

## 5. Metrics that need events

- PPDA, pressing intensity, pass networks, xT of actions rather than zones.

## 6. Engineering

- Run per match, not per clip: chunked processing, resumable stages, one manifest per match.
- Export trajectories in a standard tracking format so tracking-data tools (e.g. kloppy) can read them.
- CI running the unit tests plus a short synthetic end-to-end smoke test.
- A per-platform lock file including the CUDA torch build.

## Data and licensing constraints

- Broadcast footage is copyrighted: keep raw video and derived frames local; publish only code,
  configs, manual annotations and aggregate metrics.
- StatsBomb open data requires attribution; the YOLO11 weights are AGPL-3.0.
