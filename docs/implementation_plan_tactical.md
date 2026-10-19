# Tactical-camera upgrade: Phase 1 reconnaissance and plan

Target footage: `data/Inter Milan vs Napoli Tactical Camera.mp4`. This is a high, wide tactical camera
that pans and zooms, recorded at 1280×720 and 25 fps: 151,768 frames, 101 min. This plan covers the
upgrade asked for in the master prompt (YOLOX → ByteTrack → SRITrack Re-ID → teams/roles → ball →
calibration → speed). It is **Phase 1 only**: audit, measured baseline, feasibility checks and a gated
phase plan. No pipeline code has changed. The earlier ISL-clip upgrade (phases 1–8) is documented in
[implementation_plan.md](implementation_plan.md) and is not repeated here.

Every number below was measured in this repository on the machine described in §2, unless it is
marked *estimate*. The scratch scripts and logs live outside the repository; the commands are
given so each number can be reproduced.

## 1. What exists and what the reference repo offers

### 1.1 This repository (kept)

Pipeline: `python -m football_cv.pipeline <stage> --set key=value`, stages in order:

- ingest
- detect
- shots
- motion
- track
- calibrate
- identity
- reconstruct
- regions
- analytics
- visualize

Frames are sampled at `ingest.sample_fps` (5 fps). Every stage reads and writes parquet under
`data/processed/<clip>/`.

| area | current implementation | fit for the tactical cam |
|---|---|---|
| detection | Ultralytics YOLO11m COCO (person, sports ball), imgsz 1280, FP32 (`half: false`: FP16 gave NaN boxes on this GPU) | works, but has no goalkeeper/referee classes; the ball is weak (§3) |
| camera motion | ORB + RANSAC homography per frame pair, chained per segment (`calibration/motion.py`) | robust, unlike the reference's single-feature shift; kept |
| tracking | `supervision` ByteTrack in camera-stabilised coordinates; the ball is not tracked (best detection per frame) | heavy fragmentation (§3) |
| teams | torso CIELAB, KMeans k=2 plus an "other" class; kit hints from config | kit hints are for the ISL clip and wrong here; goalkeepers and referees all fall into "other" |
| calibration | manual keyframes plus ORB-propagated homographies; `scripts/landmark_error.py` validates them | correct, but manual keyframes for 101 min are impractical |
| identity | conservative tracklet linker, off by default: it missed its precision target on SkillCorner | the gating logic is reusable; the appearance part is colour-only |
| reconstruction, regions, analytics, visualisation | done, with tests | unchanged by this plan |

### 1.2 Reference repo (abdullahtarek/football_analysis): reuse ideas, not code

| component | what it does | decision | reason |
|---|---|---|---|
| video I/O | loads the whole video into RAM | **omit** | 151,768 × 1280×720×3 bytes ≈ 420 GB |
| tracker | ByteTrack; goalkeeper relabelled as player | **rewrite as a policy** | the goalkeeper must stay a role (`role`), not be erased |
| ball | fixed track ID 1, unlimited `interpolate()` + `bfill` | **omit** | invents long gaps; the master prompt forbids this |
| team assigner | KMeans per crop, team locked on first sight, hard-coded `if player_id == 91: team_id = 1` | **omit** | the existing CIELAB per-track clustering is strictly better |
| camera estimator | translation of the single max-displacement feature; hard-coded mask columns | **omit** | the existing ORB/RANSAC homography is robust |
| view transformer | hard-coded pixel trapezoid, 23.32×68 m | **omit** | valid for one frame of one video only |
| speed/distance | hard-coded 24 fps, 5-frame window | **rewrite** | must use real timestamps and calibrated positions |
| ball assigner | nearest player foot within 70 px | **reuse the idea** | as a possession *estimate*, in pitch metres, with a distance gate |
| overlays | ellipses, triangles, possession bar, at fixed 1080p coordinates | **reuse the style** | scale to the frame size |

The reference also does not ship its weights (`best.pt`, YOLOv5 on a 4-class Roboflow dataset).

## 2. Hardware and dependencies (inspected, not assumed)

- **GPU:** NVIDIA GeForce **GTX 1650, 4 GB** (the master prompt says "1650 Ti"; it is not a Ti). It is a
  Turing TU117 chip without tensor cores.
- **Packages:** torch 2.5.1+cu121, timm 1.0.24, ultralytics 8.4.178 (AGPL-3.0), supervision 0.30.9,
  opencv-python 4.11, onnxruntime 1.23.2 (CPU build).
- **Not installed:** TensorRT, `onnxruntime-gpu` and YOLOX.
  - YOLOX is Apache-2.0, from the Megvii GitHub repo. It is not a maintained PyPI package.
- **Storage:** E: has 233 GB free. C: has only 19 GB free, so caches (HF, torch hub, pip) should
  point to E:.
- **Python start-up** in this venv takes about 60–90 s per process. That overhead is paid by every
  pipeline stage.

## 3. Baseline: existing pipeline on a 60 s excerpt (ablation A)

Clip: `data/raw/inter_napoli_t600_60s.mp4`. It was stream-copied from 600 s, so it has 1,510 frames.
The pipeline sampled 301 of them at 5 fps. All settings were the defaults.

| stage | wall time | notes |
|---|---|---|
| ingest | 21.6 s | 301 frames |
| detect | 182.1 s | **0.605 s/frame (1.65 FPS)**, YOLO11m FP32, imgsz 1280, batch 8; peak GPU memory 2,148 MiB; mean GPU use ≈ 33 % while busy, so I/O/CPU-bound |
| shots | 129.6 s | 1 shot, 301/301 wide frames |
| motion | 61.1 s | 1 motion segment |
| track | 41.2 s | |

**Detections.**

| class | total | per frame | median confidence | frames covered |
|---|---|---|---|---|
| person | 7,216 | 23.97 | 0.681 | — |
| ball | 278 | — | 0.189 | 199 of 301 (66.1 %) |

7,236 of the 7,494 detections are on the pitch.

- **Person box height:** median 41 px, 10th percentile 33 px, 90th percentile 49 px.

**Tracking.**

- **Fragmentation: 237 person tracks** in 60 s, with no camera cuts, for about 25 visible people.
  That is about 9.5 tracks per person.
- 5,120 of the 7,216 person detections end up in a track.
- The ball is present in 180 frames (59.8 %).

**Teams.** 79 tracks away, 67 home, 69 unknown, 16 other. The kit hints are wrong for this match, so
these labels are not meaningful yet.

**Full-match extrapolation (*estimate*).** About 30,350 frames at 5 fps × 0.605 s ≈ 5.1 h for
detection alone, before Re-ID.

Reproduce:

```
python -m football_cv.pipeline ingest --config configs/inter_napoli.yaml --set clip.id=inter_napoli_t600_60s --set clip.video_path=data/raw/inter_napoli_t600_60s.mp4 --set clip.source_offset_s=600
```

Then run `detect`, `shots`, `motion` and `track` the same way. The manifest is
`data/manifests/inter_napoli_t600_60s.json`.

The numbers above came from a run with `configs/default.yaml`, so the ISL clip's kit hints were used.
That is why the team labels are meaningless. Detection and tracking run before the kit hints are read,
so those counts are unaffected. The phase-0 windows (§6) use `configs/inter_napoli.yaml`.

**Independent corroboration.** `Qussay/DEFECTS_SUMMARY_EN.md` is a third-party report from a
different system run on the same match. Its findings are treated as hypotheses only:

- the ball is visible about 65 % of the time;
- 1,738 track IDs for 32 players;
- 88 ID switches in 10 min;
- goalkeepers are misassigned by colour;
- spare balls near the touchline are taken for the match ball;
- players are about 37 px and the ball about 5 px at 720p.

These match what was measured here.

## 4. SRITrack feasibility

### 4.1 What SRITrack is

Source: [kaoyuyukao/SRITrack](https://github.com/kaoyuyukao/SRITrack), MIT licence. The paper reports
85.2 HOTA on SportsMOT.

SRITrack is a **complete online tracker**, not a Re-ID plug-in. It builds on ByteTrack, Deep-EIoU and
BoT-SORT.

**Interface.**

- `Tracker(cfg, img_sizes, frame_rate).update(dets, embeddings, frame, reid)` takes:
  - detections `[x1, y1, x2, y2, score(, N1, N2)]`;
  - one 512-d embedding per detection.
- It returns the online tracks.
- External detections are accepted: its public-tracking path builds exactly these arrays.
- It is **single-class**: there is no class, role or team logic.

**Association.**

- Expanded-IoU (EIoU) is fused with a Re-ID cost.
- The cost uses visual/position-aware thresholds: `track_vc_th` 0.5, `track_vf_th` 0.25,
  `track_p_th` 0.5 and `match_th` 0.8.
- Features are smoothed with an EMA (α = 0.9).
- The track buffer is 1000 at 30 fps. At 25 fps it scales to 833 frames, about 33 s.

**Camera-motion compensation.** It runs its own sparse optical-flow CMC.

**Quirks found while reading the code.**

- The disappearance penalty is normalised by the *global* frame index, so it weakens as a long video
  goes on.
- The `ris` option drops detections near the image border.
- Embeddings are computed for **every detection on every frame**.
- BGR OpenCV crops go straight into `ToPILImage`, so the training colour order needs checking (§4.3).

**Its own detector.** It ships YOLOX-X `sports_mix`, which is person-only and runs at 800×1440. That
is too heavy for 4 GB and has no ball or role classes, so it is **not** used here.

**Consequence for the plan.** "ByteTrack + SRITrack" cannot mean "ByteTrack with SRITrack features
bolted on". The honest ablation is:

- **B:** ByteTrack on our detections;
- **D:** SRITrack's tracker on the *same* detections, with its Re-ID.

A persistent-identity manager with role and team constraints sits above both (E). It maps
`local_track_id` → `persistent_id`.

### 4.2 Re-ID checkpoint

The user supplied it: `models/sritrack/model.pth.tar-60`, 1,104,745,338 bytes. It is git-ignored.

**Contents.**

- The file is a torch zip containing `state_dict`, `epoch` (60), `optimizer` and `scheduler`.
- It loads with `torch.load(..., weights_only=True)`.

**Architecture.**

- timm `vit_base_patch16_dinov3.lvd1689m`: a DINOv3 ViT-B/16 with RoPE and register tokens.
- A `fc = Linear(768→512) + BatchNorm1d(512)` head.
- An unused `classifier` with 1,278 training IDs.
- 86.7 M parameters in total.

**Loading.**

- SRITrack's `backbones.py` hard-codes `pretrained=True`, which triggers a Hugging Face download.
- Rebuilding with `pretrained=False` and loading the checkpoint (minus `classifier.*`) gives
  **missing = [], unexpected = []**.
- So no download of the gated upstream weights is needed.

**Input.** SRITrack resizes every crop to a 384×384 square, ignoring aspect ratio, then applies
ImageNet normalisation. The output is L2-normalised.

**Licence caveat.** The SRITrack code is MIT. The backbone weights derive from DINOv3, which is under
Meta's DINOv3 License, not an OSI licence. Fine for this MVP; review before any redistribution or
commercial use.

### 4.3 Measured cost and separability on this video

**Setup.**

- Clip: the 60 s clip, every 3rd sampled frame (0.6 s apart), giving 1,700 person crops.
- Crop size: median 41 px tall, upscaled to 384×384, as SRITrack does.
- Batch: 24 crops on the GTX 1650.
- Scratch script: `reid_feasibility.py`, kept outside the repository.

**Pairs.** There is no identity ground truth, so this is a *weak* check:

- **Positives:** 4,000 pairs of the same ByteTrack track, 1–5 s apart.
- **Negatives:** 4,000 pairs of two different tracks in the same frame.
- Track switches add label noise to the positives.
- The negatives mix team-mates with opponents, so they are easier than the hard case (same team,
  same kit).

| input order | precision | ms/crop | peak VRAM | finite | median cos, positive | median cos, negative | AUC |
|---|---|---|---|---|---|---|---|
| BGR (as SRITrack feeds it) | FP32 | 90.6 | 829 MiB | yes | 0.842 | 0.543 | 0.941 |
| BGR | FP16 autocast | 291.2 | 829 MiB | yes | 0.842 | 0.543 | 0.941 |
| **RGB** | **FP32** | 127.4 | 829 MiB | yes | 0.843 | **0.316** | **0.958** |
| RGB | FP16 autocast | 359.3 | 829 MiB | yes | 0.843 | 0.316 | 0.957 |

Colour order does not change the computation. The 90 → 127 ms spread between otherwise identical
FP32 runs is therefore run-to-run variance. Treat FP32 as about **90–130 ms per crop**.

**What this shows.**

- **The checkpoint is useful on 41 px tactical-cam crops.** With RGB input, different people drop to
  a median cosine of 0.32 while the same track stays at 0.84.
- **Feed it RGB.** RGB separates better than the BGR that SRITrack's inference code passes. The
  adapter will convert BGR to RGB, and this will be stated as a deviation from upstream.
- **FP16 is not a speed-up on this card.** It stays finite but is about 3× *slower*. Keep FP32. This
  also lowers expectations for TensorRT FP16 in phase 8.
- **Every-detection Re-ID is unaffordable here.** SRITrack embeds every detection on every frame.
  At about 24 people per frame × about 90–130 ms, that is **about 2–3 s per frame** for Re-ID alone.
  At 5 fps that is about 18–26 h per match (*estimate*).
  - Ablation D can only be run *as published* on short evaluation windows.
  - For full matches, embeddings must be computed only on triggers: new or re-activated tracks,
    ambiguous matches, and periodic gallery refresh. Phase 5 is built around this.
- **This is not an identity accuracy score.** That needs the phase-0 annotations.

## 5. Problems this plan must solve, by evidence

| # | problem | evidence | planned phase |
|---|---|---|---|
| P1 | track fragmentation | 237 tracks / ~25 people / 60 s | 4–5 |
| P2 | no goalkeeper or referee classes; GK colour ≠ team colour | COCO detector; third-party report | 2, 3 |
| P3 | ball weak and spare balls | 66 % frame coverage, median confidence 0.19; third-party report | 6 |
| P4 | calibration is manual | keyframes per shot; 101 min of panning | 7 |
| P5 | slow | 0.6 s/frame detection, GPU 33 % busy | 8 |
| P6 | no identity ground truth for this match | nothing annotated | **0 (blocker)** |
| P7 | kit config tied to the ISL clip | team labels meaningless here | 3 |

## 6. Phase plan with gates

Each phase follows the same protocol:

- it lands behind a config flag;
- it is measured on the **same** evaluation clips as the baseline;
- it is kept only if its gate passes.

Thresholds are fixed **before** each run.

| phase | work | gate (decided now) |
|---|---|---|
| **0. Evaluation set** | Annotate 2–3 windows of this match (≈30 s each, 5 fps) with identity, role, team and ball boxes: bootstrap from tracker output, correct by hand in CVAT. Add a per-match config (kits, halves). Optionally SportsMOT/SoccerNet-tracking as a public MOT proxy. | annotation exists and is reviewed; metric scripts (HOTA/IDF1 via TrackEval) reproduce on a toy case |
| 1. Feasibility | **this document** | — |
| 2. Detector | YOLOX (Apache-2.0) with COCO weights: person + sports ball, same classes as today → ablation B is a detector swap only. 4-class fine-tune (player/GK/referee/ball) **only if** phase 3's role logic fails its gate; needs a licence-checked dataset and a Kaggle T4, not this GPU. | person recall ≥ current YOLO11m and ball recall not worse, at ≤ current latency |
| 3. Teams and roles | per-match kit clustering with "other"; goalkeeper = "other" track that stays in a penalty area, affiliated **by position** (which goal it defends, from team centroids per half), not colour; referee = remaining "other", **no team**. | team accuracy ≥ 95 % with coverage reported; GK affiliation ≥ 95 % on confident observations |
| 4. Tracking | class-specific ByteTrack policies (people vs ball); then SRITrack's tracker on the same detections via an adapter (ablation D). | IDF1 and ID switches better than B on the eval windows; Re-ID cost fits the runtime budget |
| 5. Persistent identity | a manager over `local_track_id`: gallery per role+team, Re-ID only on triggers (new track, re-entry, ambiguous match), refusal on ambiguity, a role-incompatibility veto. The ball never enters the gallery. | **zero** role-incompatible merges; recovery precision ≥ 0.95 (the linker's 0.98 target was missed with colour alone) |
| 6. Ball | dedicated ball pass (higher input resolution or tiles around the last position), pitch-polygon and continuity filter against spare balls, Kalman with a **short** gap limit (≤ 0.5 s, *proposed*), possession as an estimate with a "contested/unknown" state. | ball recall and continuity vs annotations; zero interpolations across gaps longer than the limit |
| 7. Calibration | automatic pitch-keypoint or line detection to replace manual keyframes; candidates are the Roboflow `sports` keypoint model and SoccerNet calibration methods (e.g. PnLCalib); **licence and accuracy to verify**. Validated with `scripts/landmark_error.py`. Speed and distance only on frames with a valid calibration. | landmark error not worse than the manual keyframes |
| 8. Optimisation | in order: decode/crop I/O overlap (the GPU is idle 2/3 of the time), batch size, ONNX; TensorRT FP16 **only if** installed, finite, and faster, since FP16 already produced NaNs on this card | measured end-to-end FPS on the 60 s clip, outputs within tolerance of FP32 |
| 9–13 | pipeline/overlay output, JSONL/CSV export, tests listed in the master prompt, README | the master prompt's acceptance table |

**Ablations**, all on identical clips:

- **A** existing pipeline (§3).
- **B** YOLOX + ByteTrack.
- **C** B + teams/roles.
- **D** C with SRITrack's tracker.
- **E** D + persistent identity manager + CMC.
- **F** E with the optimised inference configuration.

### 6.1 Phase 0 status (2026-10-19): tooling done, annotation pending

- **Built:**
  - `configs/inter_napoli.yaml` with three windows;
  - `scripts/make_eval_window.py` (runs ablation A on a window and writes a CVAT pre-fill);
  - `scripts/eval_tracking.py` (scores a run against corrected annotations);
  - `football_cv.evaluation.cvat` and `football_cv.evaluation.mot`, using TrackEval's HOTA/CLEAR/Identity, vendored and pinned.
- **Guide:** the annotation rules and the CVAT workflow are in [annotation_guide.md](annotation_guide.md).
- **Tests:** a toy case with one ID switch reproduces the hand-computed scores: MOTA 0.75, IDF1 0.5, AssA 0.5, HOTA √0.5.

Pre-fills exported (ablation A, `configs/inter_napoli.yaml` kit hints):

| window | frames | tracker tracks | person boxes | ball frames | detect time |
|---|---|---|---|---|---|
| `w1_open_t600` | 151 | 151 | 2,411 | 128 | 109.2 s (includes model load) |
| `w2_corner_t2695` | 151 | 124 | 2,552 | 33 | 27.9 s |
| `w3_open_t4800` | 151 | 124 | 2,574 | 81 | 33.3 s |

- **Self-check.** Scoring the pre-fills against the tracks they came from gives HOTA = MOTA = IDF1 = 1.0 over 7,537 boxes, with no misses or false positives. LocA is 0.9996 because the XML stores coordinates to 2 decimals. This only shows that export, import and scoring agree; it is **not** a tracking score.
- **`w2` ball.** In 80 of the 113 frames where the detector found a ball, every ball box lay outside the pitch mask: spare balls and board clutter along the far touchline. The match ball sits at the corner flag, so most `w2` ball boxes must be added by hand.
- **Detection runtime (observation, not yet re-measured).** In a warm process, detection took 0.18–0.22 s per frame (`w2`, `w3`). The 0.605 s/frame in §3 included model load and warm-up, so the 5 h-per-match figure in §7 is probably too high. Phase 8 re-measures it properly.

The gate stays open until the three `annotations.xml` files exist and have been reviewed.

## 7. Open blockers and decisions

1. **Ground truth.** Without the phase-0 annotations, the tracking, Re-ID and team targets cannot be
   scored, and no score will be reported. Annotation needs a person's time: 151 frames × about 25 boxes
   per window. The pre-fills and the guide are ready (§6.1, [annotation_guide.md](annotation_guide.md)).
2. **YOLOX install.** It requires cloning the Megvii repo (Apache-2.0) and building it into the venv.
   It is a small download, but a new dependency.
3. **4-class detector.** No public YOLOX checkpoint has player, goalkeeper, referee and ball classes. A
   fine-tune is deferred until phase 3 shows that position/colour role logic is not enough.
4. **Runtime budget.** On this GPU, detection takes about 5 h per match (§3). Per-detection Re-ID would
   add about 18–26 h (§4.3), so Re-ID must be triggered rather than run on every detection. The
   realistic targets are offline processing or a rented GPU, not real time.
