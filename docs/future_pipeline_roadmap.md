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
