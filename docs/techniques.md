# Techniques used in the pipeline, and how far each one is actually proven

Written 2026-10-22. "Proven" is deliberately narrow: **every number comes from three 10 s windows of one match (153
frames at 5 fps), and most settings were chosen on those same windows.** Nothing here has been scored on a window held
out from tuning. Where a technique was borrowed, the source is named; where I built it, it says so. The per-item evidence
and the rejected alternatives are in `pipeline_status.md`.

Legend: **in use** = in the working config; **off** = built, default off; **rejected** = measured and not adopted.

## 1. Sampling and shots
| Technique | Origin | Status | Honest note |
|---|---|---|---|
| Frame sampling at 10 fps (was 5) | own | in use since 2026-10-22 | HOTA +1.4 and IDF1 +1.9 points over 5 fps on the three windows. Identity linker, calibration drift and replica speeds were not tested at 10 fps. |
| Shot-boundary detection, camera-motion estimate | own, OpenCV | in use | Cuts are never linked across. Not scored against annotated cuts. |

## 2. Detection
| Technique | Origin | Status | Honest note |
|---|---|---|---|
| 4-class detector (ball, player, goalkeeper, referee), Ultralytics weights | supplied by the reference repo | in use for people | HOTA 0.737 plain tracker. Referee recall 0.086. |
| YOLOX-m fine-tuned on our 4 classes (Colab, seed 0) | YOLOX (Apache-2.0); our data | in use for the ball | Ball hits 52 of 137 (0.380) against 29 for the Ultralytics model. The ball is still missed in 40% of frames, and 0 of 17 in the corner window. |
| Hybrid: people from one model, ball from the other | own | in use | Two detector passes over every frame. |
| Person floor 0.25, ball floor 0.10 | own, tuned on windows | in use | Lower floors were tried and rejected. |
| Hard negatives for the drone and the camera rig | own | **dataset built, never trained** | The Colab GPU quota ran out at epoch 35 of 60, so there is no scored model. The baseline fails the rig criteria, which is why it is still open. |
| Stage-2 fine-tune, larger input sizes, flip TTA | own | rejected | None beat stage 1 on the ball or on tracking. |

## 3. Tracking
| Technique | Origin | Status | Honest note |
|---|---|---|---|
| ByteTrack on camera-stabilised coordinates | ByteTrack, via `supervision` | in use | |
| Buffered box matching (C-BIoU style, buffer 0.6) | published idea, own implementation | in use | HOTA 0.737 -> 0.850 and ID switches 127 -> 70 on the Ultralytics detections. Hurts the crowded corner window. |
| Recovery of detections the tracker swallows | own | in use with the buffer | +2.3 recall points, costs a little IDF1. Without the buffer it tripled ID switches. |
| Static-ball filter and spare-ball / penalty-spot rule | own | in use | A still ball at a corner would also be dropped, even if detected. |
| Per-ball state: one ball per frame, no ball tracker | own | in use | There is no ball track, no physics and no Kalman filter on the ball. |

## 4. Teams and roles
| Technique | Origin | Status | Honest note |
|---|---|---|---|
| Kit-colour clustering (Lab, shirt mask), goalkeeper-aware | the reference repo's KMeans idea, own mask and gates | in use | Per-crop colour 94.3% right. |
| Per-track Viterbi team smoothing | own | in use | Wrong team 6.1% -> 1.4%, **chosen over 12 settings on the same windows**. |
| Roles from SRITrack embeddings (logistic regression) plus a goal-centre keeper prior | SRITrack weights (a Re-ID model); own probe and prior | **off by default**, accepted in-sample | Role accuracy 0.9886 against 0.9284 on the three windows. Prior geometry was reshaped after seeing window a2. |
| Edge and height guards on the role relabel | own | accepted in-sample | Removed 11 false keeper/referee calls (photographers, substitutes). 5 remaining are the camera rig. |
| Voting by track, pure position prior, zone rule | own | rejected | Each failed its declared criterion. |

## 5. Calibration and projection
| Technique | Origin | Status | Honest note |
|---|---|---|---|
| Hand-clicked pitch landmarks on keyframes | manual, by a person | in use | A person has to click keyframes for every new clip. This does not scale to a new match without effort. |
| Pyramidal Lucas-Kanade + RANSAC chain between keyframes, line refinement | OpenCV, own chain | in use | 0.17-0.76 m median drift over 10 s; 2.5 m median at one zoom-out in the 3-minute clip. |
| Off-pitch exclusion of people (median more than 2 m outside the lines) | own | in use | Removed 29 tracks on the 3-minute clip. **Not scored against ground truth.** |
| Corner-flag exclusion (still person on a corner) | own | **rejected** (2026-10-22) | Found all four flags, but also dropped the referee and a waiting corner taker. See `corner_flags.md`. |

## 6. Identity and analytics
| Technique | Origin | Status | Honest note |
|---|---|---|---|
| Conservative tracklet linker with chi-square gating and a Re-ID term | own, SRITrack features | in use as an option, **not applied to reconstruction** | Only a synthetic split test exists (84% recall at 85% precision with Hungarian assignment, a best case). On public SkillCorner data its precision was 0.86 against a 0.98 target. Identity over 180 s is not achieved. |
| Kalman + RTS gap filling for trajectories | textbook filter, own implementation | in use for gaps | Beaten by constant velocity at short gaps. Kalman end states for the linker gave no reliable gain. |
| Speed, distance, possession (3 m nearest player) | the reference repo's definitions, own implementation | in use | **No ground truth exists for any of these**; only coverage is reported. |

## What is not here
No held-out window, no 60 s ground truth (plan: `ground_truth_60s.md`), no ball tracker, no jersey-number reading, no
SigLIP (needs your go-ahead for the download), and no trained model that rejects the drone, the rig or the corner flags.
