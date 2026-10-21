# Pipeline status: what detects and tracks the four classes, what was accepted, what was rejected

Scope: the Inter v Napoli tactical camera, classes `player`, `goalkeeper`, `referee`, `ball`.
Everything below is taken from the project docs and run logs (`detector_4class.md`, `finetune_4class.md`,
`poc_replica.md`, `role_probe.md`, `evaluation.md`). The scored numbers come from **three 10 s audited windows**
(a1 open play, a2 corner, a3 open play; 153 frames at 5 fps, one match). Most settings were chosen on those same
windows, so every figure is optimistic until a held-out window is annotated. Where something was not measured,
it says so.

## 1. Data flow

`shots` -> `motion` -> `detect` -> `track` (tracker, kit colours, teams, roles, gates) -> `calibrate` (pitch homography,
projection, off-pitch drop) -> `identity` (tracklet linking, Re-ID) -> `replica` (speed, distance, possession).
Stages run as `python -m football_cv.pipeline <stage> --config <yaml>`; `--set clip.id=...` selects a per-window run dir.

## 2. Module by module

Status: **ACCEPTED** (in the working config), **REJECTED** (measured, not adopted), **OFF** (built, default off),
**OPEN** (not solved or not scored).

### Detector (people and ball)

| Item | Status | Evidence |
|---|---|---|
| 4-class YOLOX-m fine-tune, stage-1 `best_ckpt.pth` (default) | ACCEPTED | Role accuracy 0.936; referee role recall 0.342 (matched boxes); ball hits 52/137 (0.380). HOTA 0.691 against 0.737 for the Ultralytics 4-class model, so it is **worse on DetA** (0.710 vs 0.790) though better on IDF1 (0.794 vs 0.738). The DetA gap is unexplained. |
| Ultralytics 4-class weights (reference weights) | Comparison, and used on the 3-minute clip | HOTA 0.737; referee recall 0.086; ball 29/137 |
| Person confidence floor 0.10 / 0.15 instead of 0.25 | REJECTED | HOTA 0.668 vs 0.691; fewer output boxes. Floors above 0.25 were not tried. |
| Ball confidence floor 0.01 to 0.05 | REJECTED (not adopted) | 5 extra hits of 137 but precision 0.675 -> 0.50-0.65; no uncertainty estimate. Default stays 0.10. |
| Tracker activation threshold 0.20 to 0.50 | No effect | HOTA moves at most 0.0006 |
| Stage-2 fine-tune (20 epochs from stage 1) | REJECTED | Never beat stage 1: HOTA 0.643-0.689 vs 0.691; referee recall 0.20-0.32 vs 0.34 |
| Input size 1920 / 2560 for the ball | REJECTED | Ball top-1 20% / 10% vs 25% at 1280 |
| Horizontal-flip test-time augmentation for the ball | REJECTED | No gain (0.380 vs 0.416 top-1) |
| Hybrid: 4-class people + YOLOX ball (`detection.ball_detector`) | ACCEPTED where used | Ball hits 29 -> 53 of 137 at the same HOTA 0.857 / IDF1 0.860; costs a second pass over every frame |
| Rig-as-person hard negatives (`v3_rig`, includes the drone patches) | OPEN, **dataset built, not trained** | 13 rig person-box patches from one 13 s episode (PoC 234-299) x 5 added to the drone set; notebook and zip ready; criteria R1-R3 declared (`finetune_4class.md`, "Rig as a person"). Stage-1 baseline fails R1 (13 of 17 a2 frames) and R2 (11 of 23) |
| Drone hard negatives (stage-3 training) | OPEN | Dataset built (15 patches, four hover episodes, correlated); the model is **not scored**; criteria D1-D6 declared |
| Ball detection overall | OPEN | The detector scores the true ball below 0.001 in 40% of frames; slow or still balls and balls at feet are missed; window a2 (ball at the corner arc under the taker) scores 0 for every model |

### Tracker

| Item | Status | Evidence |
|---|---|---|
| ByteTrack family with tracklet recovery | ACCEPTED | base |
| Buffered matching (C-BIoU style, `box_buffer` 0.6) | ACCEPTED | HOTA 0.737 -> 0.850, IDF1 0.738 -> 0.862, ID switches 127 -> 70 on the Ultralytics detections |
| `recover` (re-attach swallowed detections, keep 0.8) | ACCEPTED with the buffer | +2.3 recall points for -0.2 IDF1; hurts the crowded a2 window (precision 0.97 -> 0.90) |
| `recover` without the buffer | REJECTED | IDSW 127 -> 361 |
| Buffer capped by distance to the nearest player | REJECTED | a2 HOTA up (0.796 -> 0.789-0.805) but a1 IDF1 down to 0.79-0.83 |
| Tracking at 25 fps (confirmation 10 frames) | Not adopted | IDF1 0.837, below 0.862 from the buffer at 5 fps; 5x detection cost |
| 10 fps sampling | RECOMMENDED, **not yet the default** | HOTA +1.4, IDF1 +1.9, IDSW -22% over 5 fps for about 1.3x cost; 15 fps adds nothing; tuned on the same windows, so a pointer |
| Static-ball filter | ACCEPTED | drops still balls; a still ball at a corner arc would also be dropped even if detected |
| Outlier-run marking and goalkeeper gate (`teams.outlier_runs`, `goalkeeper_gate`) | ACCEPTED | removes implausible goalkeeper tracks |
| Non-person false positives (corner flags, camera rig) | OPEN, **a detector problem** | tracks 74, 93 (flag), 160, 83, 87 (rig). In the a2 evaluation window a camera on a pole at about (22 m, 48-52 m) is tracked as a person (tracks 22, 24, 27; 9 boxes) and the role probe calls it goalkeeper (p 0.51-0.99). No position rule separates it from a real keeper standing 21 m out (a3 gt24), so the fix belongs in the detector (a rig/camera hard-negative class or training negatives, as for the drone), not in the role stage |

### Teams and roles

| Item | Status | Evidence |
|---|---|---|
| Kit-colour clustering, goalkeeper-aware, kit mask | ACCEPTED | per-crop colour 94.3% right vs 86.7% for the track label |
| Per-row Viterbi team smoothing (`teams.smoothing`) | ACCEPTED | correct team 86.7% -> 91.4%, wrong 6.1% -> 1.4%; **tuned in-sample over 12 settings** |
| Cutting a track where the kit changes (`split_tracks`) | REJECTED | mixed result (HOTA 0.860, IDF1 0.857, IDSW 89 -> 91) |
| Cutting tracks by crop-colour run | REJECTED | HOTA 0.857 -> 0.825-0.843; colour too noisy |
| Outlier kit group dropped before the fit (under 15% of crops) | ACCEPTED | restores 68 home / 69 away on the 3-minute clip |
| Detector role used for referee / goalkeeper | ACCEPTED as the default | referee recall only 0.34 (YOLOX); goalkeepers confused during corners |
| SRITrack embedding role probe (logistic regression on `fc`) | Offline: PASS on referee, FAIL on goalkeeper | referee 0.977 / 0.986 recall / precision; keeper recall 0.608 (stage 1) |
| Track-level voting + detector class (stage 1b) | REJECTED | keeper recall 0.36, precision 0.49 |
| Pitch-position goalkeeper prior, nearest the goal line (stage 1c) | REJECTED | keeper precision 0.759 < 0.80 (a defender stood on the goal line in the a2 corner) |
| Goal-centre distance prior + probe tie-break (stage 1d, `Q1`) | ACCEPTED **offline only** | accuracy 0.989, keeper recall 0.935 / precision 0.867, referee 0.983 / 0.978. On **ground-truth boxes**, three windows, four keeper tracks; reshaped after seeing a2. **Now wired into the pipeline behind a flag (stage 2)** |
| Probe-only keeper must lie in a penalty area (stage 3 zone rule G) | REJECTED | removed the rig calls but also 3 of 27 boxes of a real keeper standing 17-21 m out (a3 gt24 0.778 -> 0.667, N2 per-track clause failed) |
| Roles stage wired behind `role_probe.enabled` (stage 2, detector boxes) | ACCEPTED, default off | role accuracy 0.9886 vs 0.9284, referee recall 1.00 vs 0.03, keeper recall 0.925 vs 0.507; boxes and tracking metrics unchanged; 3 windows, in-sample prior geometry |
| Edge and height guards on the relabel (stage 3b, `edge_margin_px` 2, `min_height_px` 20) | ACCEPTED, in-sample | removed all 11 targeted false calls (photographers, substitutes, edge fragments); extra keeper/referee calls 26 -> 11 (5 are the camera rig); keeper recall 0.932, precision 0.845, referee 0.994 / 0.982, accuracy 0.9886; 0.8% of matched boxes left unrelabelled |
| SigLIP embeddings | NOT DONE | only after explicit go-ahead; the download size must be stated first |

### Calibration and projection

| Item | Status | Evidence |
|---|---|---|
| Hand-clicked landmark keyframes + LK-chain refinement + line refinement | ACCEPTED | keyframe residual <= 0.47 m on a2/a3; 10 s drift checks 0.17-0.76 m median; 3-minute clip: 30 s drift 0.18-0.46 m median except the zoom-out at 690-705 s (2.5 m median, 4.8 m max) |
| ORB feature chain instead of optical flow | REJECTED (earlier) | it was feature matching, not optical flow; replaced by pyramidal LK + RANSAC |
| Off-pitch exclusion (people median > 2 m outside the lines) | ACCEPTED | removed 29 tracks on the 3-minute clip; referees and goalkeepers exempt; **not scored against ground truth** |
| Ball spare-ball / penalty-spot rule | ACCEPTED | dropped 5 of 601 ball rows; confirmed real balls survive |
| Calibration in the evaluation windows | NOW AVAILABLE | a2 and a3 keyframes were added; they were missing before, which is why the position prior could not run on them |

### Identity and re-identification

| Item | Status | Evidence |
|---|---|---|
| Tracklet linker with SRITrack Re-ID embeddings | ACCEPTED as an option, **not applied to reconstruction** | `applied_to_reconstruction: False`; 3-minute clip: 612 -> 312 identities with buffer + recovery |
| Identity persistence over 180 s | OPEN | not achieved; only about 140 tracks have enough tall crops to embed; unknown-team tracks are never linked |
| Linker on public SkillCorner data | Held-out precision 0.86 against a 0.98 target, about 3% of true continuations linked | older baseline; that is why linking was left off |
| Ground truth for speed, distance or possession | OPEN | coverage figures only |

## 3. What is still weak, in the order it hurts

1. **Referee recall from the detector (0.34)** and **goalkeeper confusion at corners**. The probe plus position prior fix this on ground-truth boxes; the detector-box test has not been run.
2. **Ball**: the detector does not see it in 40% of frames, and a2 is zero. This is a data problem (slow or still balls, balls at feet).
3. **DetA gap** between the YOLOX model (0.710) and the Ultralytics model (0.790) is unexplained.
4. **Non-person false positives** (flags, rig) and the unscored drone model.
5. **Held-out window**: nothing here has been scored on a window that was not used to choose settings.
