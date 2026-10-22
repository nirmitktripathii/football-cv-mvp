# Fine-tuning the 4-class detector (YOLOX-m, Colab)

Status: **trained once and scored once** (2026-10-20). The YOLOX notebook ran on a Colab T4 (60 epochs, seed 0); the
scored checkpoint is `best_ckpt.pth` (epoch 34, SHA-256 `55ef734a…`, COCO AP 0.423 on the validation windows), fixed
before any pipeline result was seen. The YOLOX backend is `src/football_cv/detection/yolox_backend.py`
(`configs/inter_napoli_yolox4.yaml`). `notebooks/sam3_annotate_colab.ipynb` has not been executed. See "Results".

## Why

The current detector (`models/football_4class/best.pt`, Ultralytics v8, AGPL) matches 152 of the 298 audited referee
boxes in the three evaluation windows, and only 13 of those 152 carry the role "referee" (role recall 0.086, measured
over matched boxes), so referees are scored as players or missed. The ball recall ceiling is about 0.21 (window a2 has no ball candidates at all). Colour rules were
rejected for referees (neon-yellow referee and neon goalkeeper kits overlap), so the fix is more labelled referee and
ball examples. YOLOX is Apache-2.0, which removes the AGPL constraint on the weights.

## Data

| Role | Source | Status |
|---|---|---|
| Validation | The three audited windows `a1`, `a2`, `a3` (153 frames, CVAT-audited) | Ready. Never used for training |
| Training | One or more Roboflow Universe YOLOv8 exports (`--source`, repeatable) | **Not downloaded.** Needs the user's yes, and the Roboflow key should be rotated first (it was pasted in chat) |
| Training (planned) | SAM 3 pseudo-labels on own footage, audited, with condition tags (day/night, weather) | Not built. See the SAM 3 note below |

Each source's licence is copied into `manifest.json`; check it before training on or redistributing the weights.
Class names map to `ball / goalkeeper / player / referee` through `CLASS_ALIASES`. Side and assistant referees are
deliberately not mapped (the pipeline scores them as off-pitch staff). Anything unmapped is counted in the manifest
under `unmapped_classes`, never guessed.

## Leakage and duplicates

- A training image whose 64-bit dHash is within 6 bits of any validation frame is removed (`--leak-hamming`).
- Near-duplicates (within 2 bits) of an already kept image are removed (`--dup-hamming`).
- dHash catches re-encodes and mild resizes, **not crops or different frames of the same match**. Training on other
  broadcasts of the same fixture would still leak, so do not add sources from Inter v Napoli.
- `--prefer ball,referee` keeps every image that has one of those classes and a seeded `--keep-other-frac` of the rest.

## Run

```bash
python scripts/build_finetune_dataset.py --dry-run --out data/finetune/v1 --source <export_dir>
python scripts/build_finetune_dataset.py --out data/finetune/v1 --source <export_dir> --prefer ball,referee
```

Zip `data/finetune/v1` as `finetune_v1.zip` into `MyDrive/football_cv/`, then run the notebook top to bottom on a T4.
The notebook pins YOLOX to commit `6ddff482…`, starts from COCO `yolox_m.pth`, trains with a fixed seed, evaluates
COCO AP on the validation windows, and saves the checkpoint, experiment file, logs, manifest and SHA-256 together.
Hyper-parameters (60 epochs, batch 8, 704x1280 input, no mixup) are starting points, not tuned.

### Nothing is lost if Colab disconnects

The experiment file sets `output_dir` to `MyDrive/football_cv/yolox_runs`, so YOLOX writes straight into Drive
(behaviour read from the YOLOX source at the pinned commit, not yet observed in a run):

| File (in `yolox_runs/yolox_m_football4/`) | Written | Use |
|---|---|---|
| `latest_ckpt.pth` | after every epoch | resume point |
| `best_ckpt.pth` | each evaluation (every 2 epochs) if AP50-95 improved | the model to keep |
| `last_epoch_ckpt.pth` | each evaluation | last evaluated weights |
| `train_log.txt`, `tensorboard/` | continuously | history |
| `football4_exp.py`, `run_config.json`, `dataset_manifest.json`, `sessions.log` | start of every session | the config used |

Re-running the notebook after a disconnect resumes from `latest_ckpt.pth` (at most one epoch repeated). The last cell
copies the best checkpoint, config, logs and a SHA-256 into a timestamped `final_*` folder. Drive sync is
asynchronous, so the last seconds before a crash can be missing.

## Pass criteria (proposed before the run; change them now, not after)

Scored with `scripts/eval_tracking.py` on the three windows after the checkpoint is wired in, against the current
4-class model with the same tracker settings:

1. Referee role recall at least 0.60 (baseline 0.086).
2. Ball hit recall (within 10 px) at least 0.35 with static-drop on (baseline about 0.21).
3. Role accuracy not below the 0.928 baseline.
4. HOTA on players not more than 0.01 below the baseline.

The audit error rate of the validation windows is reported next to every number. COCO AP from the notebook is a
training signal only. The validation windows are three clips of one match, so a pass here does not show the model
works in other stadiums, kits, light or weather.

## Results (YOLOX `best_ckpt.pth`, three audited windows, same tracker settings, ball static-drop on)

| Criterion | Threshold | Current 4-class | YOLOX | Verdict |
|---|---|---|---|---|
| 1 Referee role recall (matched boxes) | >= 0.60 | 0.086 (13/152) | 0.342 (64/187) | **fail**, 4x better |
| 2 Ball hit recall (10 px) | >= 0.35 | 0.212 (29/137) | 0.380 (52/137, CI 0.30-0.46) | point estimate passes; CI lower bound is below 0.35 |
| 3 Role accuracy | >= 0.928 | 0.9281 | 0.9361 (CI 0.927-0.944) | passes by a margin inside the CI |
| 4 Pooled HOTA | within 0.01 | 0.7372 | 0.6913 | **fail** (-0.046); DetA 0.790 -> 0.710, IDSW 127 -> 82, IDF1 0.738 -> 0.794 |

Ball gap filling on YOLOX detections (pre-registered criteria in `scripts/eval_ball_gapfill.py`): recall 0.380 -> 0.445
(+0.066, E1 needs +0.10: fail); filled points within 10 px 9/16 (E2 passes) but within 30 px 0.69 against 0.70 (fail,
n = 16); precision drop 0.040 (E3 passes); O3 (RTS no worse than linear) fails as before. Per window, ball hits
P0 -> P1: a1 34 -> 41 of 49, a2 0 of 51, a3 18 -> 20 of 37. In a2 none of the 26 raw ball detections is on the audited
ball, so a2 is a miss of the detector, not of the static-drop rule.

Caveats: the checkpoint was chosen by COCO AP on these same 153 frames, so every number above is optimistic. The
validation windows are one match. Person confidence floors (0.25) were not tuned for YOLOX, whose score is objectness x
class probability; whether that explains the DetA drop is not tested.

## Confidence-floor sweep (pre-declared 2026-10-20, before any sweep result)

Why: YOLOX scores are objectness x class probability, so the person floor 0.25 inherited from the Ultralytics config may
cut real players (DetA 0.790 -> 0.710). Grid, fixed now: `conf_person` in {0.10, 0.15, 0.25} x `conf_ball` in
{0.05, 0.10, 0.20}, nine settings, same checkpoint, tracker and static-drop. Detection runs once per window at
(0.10, 0.05); each setting thresholds those saved rows and re-runs shots, motion and track (thresholding after NMS is
equivalent to a lower floor because a low-scored box never suppresses a higher-scored one). Check: the (0.25, 0.10)
setting must reproduce the existing `inter_napoli_yx4` numbers exactly, or the shortcut is wrong.

Reported for every setting: pooled HOTA, DetA, IDF1, IDSW, ball hit recall and precision, referee role recall (matched
boxes), role accuracy, gap-fill P1 recall. No floor is adopted from this sweep: the windows are the checkpoint-selection
set, so any "best" cell is in-sample. A floor change would need a held-out window first.

### Sweep results (run 2026-10-20, all nine cells, same checkpoint, three audited windows)

The (0.25, 0.10) cell reproduces the earlier `inter_napoli_yx4` numbers exactly (HOTA 0.6913, 3024 boxes, ball hits 52,
referee 64/187, P1 61), so the thresholding shortcut is valid. The ball floor never changes a person metric.
Raw table: `artifacts/inter_napoli_yx4/sweep_conf_floors.csv`.

| person | ball | HOTA | DetA | IDF1 | IDSW | pred boxes | ball hits (recall) | ball precision | gap-fill P1 hits | referee role recall (matched) | role acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.05 | 0.6680 | 0.6879 | 0.7705 | 95 | 2938 | 53 (0.387) | 0.654 | 62 | 62/151 = 0.411 | 0.9433 |
| 0.10 | 0.10 | 0.6680 | 0.6879 | 0.7705 | 95 | 2938 | 52 (0.380) | 0.693 | 61 | 62/151 = 0.411 | 0.9433 |
| 0.10 | 0.20 | 0.6680 | 0.6879 | 0.7705 | 95 | 2938 | 52 (0.380) | 0.732 | 61 | 62/151 = 0.411 | 0.9433 |
| 0.15 | 0.05 | 0.6683 | 0.6888 | 0.7717 | 96 | 2941 | 53 (0.387) | 0.646 | 62 | 62/150 = 0.413 | 0.9440 |
| 0.15 | 0.10 | 0.6683 | 0.6888 | 0.7717 | 96 | 2941 | 52 (0.380) | 0.684 | 61 | 62/150 = 0.413 | 0.9440 |
| 0.15 | 0.20 | 0.6683 | 0.6888 | 0.7717 | 96 | 2941 | 52 (0.380) | 0.722 | 61 | 62/150 = 0.413 | 0.9440 |
| 0.25 | 0.05 | 0.6913 | 0.7097 | 0.7935 | 82 | 3024 | 53 (0.387) | 0.646 | 62 | 64/187 = 0.342 | 0.9361 |
| 0.25 | 0.10 | 0.6913 | 0.7097 | 0.7935 | 82 | 3024 | 52 (0.380) | 0.675 | 61 | 64/187 = 0.342 | 0.9361 |
| 0.25 | 0.20 | 0.6913 | 0.7097 | 0.7935 | 82 | 3024 | 52 (0.380) | 0.712 | 61 | 64/187 = 0.342 | 0.9361 |

What it shows, and what it does not:

- The hypothesis that the 0.25 person floor was too strict is **not supported**: lowering it to 0.15 or 0.10 made HOTA,
  DetA, IDF1 and IDSW worse (HOTA 0.668 against 0.691) and reduced the number of output boxes. Why was not investigated
  (the tracker's own score thresholds are the first place to look). The DetA gap to the Ultralytics model (0.790) is
  therefore not a confidence-floor effect inside this grid.
- Lower person floors raise referee role recall (0.34 to 0.41) and role accuracy (0.936 to 0.944), but over a different
  matched set (150-151 against 187 boxes), so the two are not like-for-like. Neither reaches the 0.60 target.
- The ball floor only trades precision (0.65 to 0.73) for one hit (53 against 52 of 137); the difference is one frame.
- No cell meets the HOTA criterion (within 0.01 of 0.7372) or the referee criterion (0.60). Higher person floors
  (above 0.25) were not in the pre-declared grid and were not run.
- Everything is in-sample (the windows chose the checkpoint). No floor change is recommended from this.

## Tracker-threshold and low ball-floor probes (pre-declared 2026-10-20, before any result)

Lead, untested: ByteTrack's `track_activation_threshold` is 0.30 (tuned for the Ultralytics scores). Boxes scoring below
it can only join the second association pass, never start a track, and YOLOX scores (objectness x class) may sit on a
different scale. Probe A: activation threshold {0.20, 0.30, 0.50} x person floor {0.15, 0.25}, six cells, ball floor
0.10, tracker stage only. The (0.30, 0.25) cell must reproduce the earlier (0.25, 0.10) cell exactly.
Probe B: ball floor {0.01, 0.02, 0.03} at person floor 0.25, activation 0.30, from a new detection pass at ball floor
0.01 (the earlier pass stopped at 0.05); compared with the 0.05 / 0.10 / 0.20 cells above. The minimum-consecutive-frames
setting is not varied. Same in-sample caveat as the first sweep: nothing is adopted from these windows alone, and every
cell is reported.

### Probe results (run 2026-10-20, all nine cells)

The (0.30, 0.25) cell of probe A reproduces the earlier (0.25, 0.10) cell exactly (HOTA 0.6913, DetA 0.7097, IDF1 0.7935,
IDSW 82, 3024 boxes, ball hits 52, referee 64/187, role accuracy 0.9361, P1 61), so the shortcut is valid. Raw table:
`artifacts/inter_napoli_yx4/sweep_tracker_ballfloor.csv`.

Probe A (ball floor 0.10):

| activation | person | HOTA | DetA | AssA | IDF1 | IDSW | pred boxes | pred ids | referee role recall (matched) | role acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 0.15 | 0.6679 | 0.6882 | 0.6491 | 0.7714 | 96 | 2938 | 142 | 62/150 = 0.413 | 0.9440 |
| 0.20 | 0.25 | 0.6910 | 0.7094 | 0.6741 | 0.7935 | 82 | 3021 | 132 | 63/186 = 0.339 | 0.9361 |
| 0.30 | 0.15 | 0.6683 | 0.6888 | 0.6494 | 0.7717 | 96 | 2941 | 141 | 62/150 = 0.413 | 0.9440 |
| 0.30 | 0.25 | 0.6913 | 0.7097 | 0.6744 | 0.7935 | 82 | 3024 | 131 | 64/187 = 0.342 | 0.9361 |
| 0.50 | 0.15 | 0.6677 | 0.6888 | 0.6481 | 0.7716 | 94 | 2924 | 140 | 57/145 = 0.393 | 0.9442 |
| 0.50 | 0.25 | 0.6907 | 0.7099 | 0.6730 | 0.7942 | 80 | 3003 | 130 | 59/182 = 0.324 | 0.9366 |

Probe B (person 0.25, activation 0.30; the person metrics are identical in all three cells):

| ball floor | ball hits (recall) | ball predicted frames | ball precision | gap-fill P1 hits | referee | role acc. |
|---|---|---|---|---|---|---|
| 0.01 | 57 (0.416) | 115 | 0.496 | 63 | 64/187 | 0.9361 |
| 0.02 | 57 (0.416) | 97 | 0.588 | 63 | 64/187 | 0.9361 |
| 0.03 | 56 (0.409) | 91 | 0.615 | 63 | 64/187 | 0.9361 |

Next to the first sweep at the same person floor: 0.05 gave 53 hits at 0.646 precision, 0.10 gave 52 at 0.675, 0.20
gave 52 at 0.712.

What it shows, and what it does not:

- The tracker activation threshold is **not** the cause of the DetA gap to the Ultralytics model: moving it from 0.20 to
  0.50 changes HOTA by at most 0.0006 at either person floor, so the lead is not supported. The cause remains unexplained.
  Possible places to look, none tested: the detector's own box quality and class confusion, the NMS threshold, and the
  training data distribution.
- Lowering the ball floor to 0.01-0.02 adds five hits over 0.10 (57 against 52 of 137, recall 0.416 against 0.380) and
  two gap-fill hits (63 against 61), but ball precision falls from 0.675 to 0.496-0.588 because the output has 77 ->
  97-115 ball frames. 0.03 sits between (56 hits, 0.615). Person metrics are unchanged, since the ball is not tracked.
  The ball-hit criterion (0.35) was already met at 0.10 (0.380), so a lower floor is not needed to pass it.
- Five hits over 137 ball frames is small; no uncertainty estimate was computed, so this is not shown to be better than
  the 0.10 setting. Whether extra low-confidence balls are real positives or lucky near-misses was not inspected.
- Everything is in-sample. Nothing is adopted; the default floors stay at 0.25 (person) and 0.10 (ball).

## Stage-2 fine-tune outcome (Colab run finished 2026-10-20; scored 2026-10-21)

Stage 2 (`exps/football4_stage2.py`, 20 epochs from the stage-1 `best_ckpt.pth`, seed 0, 5 no-augmentation epochs) did not
beat stage 1 on the Colab validation set or on the pipeline windows.

Checkpoints scored (SHA-256, so the files can be re-checked):

- stage-2 `best_ckpt.pth` (epoch 2): `d690b16802df2f7da773b0ff502447df40f7d291846e3e7742570cf81b0bca21`
- stage-2 `last_epoch_ckpt.pth` (epoch 20): `eaa3ef9a129cf3f74651baabdc234e627192fbee05769dab401ec13bb132c8e4`
- stage-1 `best_ckpt.pth` (the frozen scored checkpoint): `55ef734a54e0d13f1f0debfc1a0793b777d9728cc1d1cdf741afe3491c718eda`

Colab validation (20 images; referee n=298, ball n=137, so every per-class figure is noisy): AP50-95 moved between 0.375
and 0.400 over the run and never reached stage 1's 0.423. Standalone evals: `best_ckpt` 0.395 (ball 0.128, goalkeeper
0.435, player 0.701, referee 0.317); `last_epoch_ckpt` 0.380 (ball 0.115, referee 0.238). Referee AP swung between
0.24 and 0.32 and was best early; ball AP stayed at about 0.12-0.136. The no-augmentation epochs did not help.
`best_focus_ckpt.pth` has the same weights as `best_ckpt.pth` (epoch 2), so it was not scored separately.

Pipeline scoring (three audited windows, person floor 0.25, ball floor 0.10, tracker as in `configs/default.yaml`, same
procedure as the stage-1 cells; the stage-1 row is the (0.25, 0.10) cell above):

| checkpoint | HOTA | DetA | AssA | IDF1 | IDSW | pred boxes | ball hits (recall) | ball precision | referee role recall | role acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| stage 1 `best_ckpt` | 0.6913 | 0.7097 | 0.6744 | 0.7935 | 82 | 3024 | 52 (0.380) | 0.675 | 64/187 = 0.342 | 0.9361 |
| stage 2 `best_ckpt` (epoch 2) | 0.6885 | 0.7154 | 0.6632 | 0.7858 | 86 | 3014 | 53 (0.387) | 0.631 | 59/182 = 0.324 | 0.9336 |
| stage 2 `epoch_15_model` | 0.6426 | 0.6986 | 0.5916 | 0.7190 | 110 | 2970 | 56 (0.409) | 0.622 | 54/177 = 0.305 | 0.9335 |
| stage 2 `epoch_16_model` | 0.6515 | 0.6992 | 0.6076 | 0.7332 | 108 | 2981 | 57 (0.416) | 0.620 | 54/177 = 0.305 | 0.9337 |
| stage 2 `epoch_17_model` | 0.6530 | 0.6993 | 0.6102 | 0.7431 | 100 | 2968 | 54 (0.394) | 0.593 | 53/176 = 0.301 | 0.9324 |
| stage 2 `epoch_18_model` | 0.6595 | 0.6917 | 0.6292 | 0.7570 | 100 | 2939 | 55 (0.402) | 0.611 | 49/172 = 0.285 | 0.9317 |
| stage 2 `epoch_19_model` | 0.6688 | 0.7019 | 0.6379 | 0.7600 | 98 | 2964 | 54 (0.394) | 0.600 | 35/175 = 0.200 | 0.9266 |
| stage 2 `epoch_20_model` | 0.6681 | 0.6995 | 0.6387 | 0.7646 | 98 | 2956 | 55 (0.402) | 0.632 | 50/173 = 0.289 | 0.9323 |
| stage 2 `last_epoch_ckpt` (epoch 20) | 0.6681 | 0.6995 | 0.6387 | 0.7646 | 98 | 2956 | 55 (0.402) | 0.632 | 50/173 = 0.289 | 0.9323 |

`epoch_20_model` and `last_epoch_ckpt` give identical results, as expected for the same final weights. All eight
stage-2 files extracted from the Drive export match the `SHA256` file shipped with them. Per that file the stage-2
validation set is the three audited windows, so choosing epoch 2 as `best_ckpt` was itself selected on the windows used
for this scoring, which makes the comparison in-sample for stage 2 as well.

Against the pass criteria: no stage-2 checkpoint meets referee role recall >= 0.60 (range 0.200-0.324) or HOTA within
0.01 of 0.7372 (range 0.6426-0.6885). All meet ball hit recall >= 0.35 (0.387-0.416). Role accuracy >= 0.928 is met by
all except epoch 19 (0.9266). Stage 1 meets the same two and fails the same two. Epoch 2 is 0.003 HOTA below stage 1;
the later epochs are 0.023-0.049 below, and HOTA rises over epochs 15-20 while referee recall falls. Ball hits differ by
1-5 over 137 ball frames, which is not a measurable change. No uncertainty estimate was computed.

Stage 2 is not adopted; the default checkpoint stays the stage-1 `best_ckpt.pth`. Why stage 2 did not move referee
recall was not investigated.

## Drone hard negatives (dataset built 2026-10-21; model not trained)

**Problem.** In the 3-minute clip a hanging camera rig (a grey disc on a pole with a dark shadow below it) is detected
as the ball at confidence 0.7-0.8 and is sometimes chosen over the real ball. Size, brightness, saturation and the
shadow patch do not separate it from motion-blurred real balls, so a rule on those would also drop real balls. The
detector has to learn that it is background.

**What was built.**

- `scripts/mine_drone_negatives.py` cuts a 160 px patch around each false detection from a pipeline run. Sample ids were
  chosen by looking at a gallery of the ball candidates, not by a rule. A patch is skipped when any tracked person overlaps
  it, so no unlabelled player becomes "background". 15 patches survive from 34 samples the user confirmed as the rig (76-87, 230-242, 305-315, 394-401, 410, 411) (four hover episodes,
  samples 76-87, 230-242, 305-315 of `poc_seg1rb`). The 160 px size was chosen because 320 px left only 3 patches.
  Those 15 patches are highly correlated: four episodes, not 15 independent examples.
- `build_finetune_dataset.py --negatives DIR --negatives-repeat 5` adds them as box-less training images (75 images).
  `football_cv.detection.negatives` holds the tested helpers (`tests/test_negatives.py`).
- `data/finetune/v2_drone` is v1 (same three sources, same seed, same flags) plus the 75 negatives: 1442 train images,
  identical box counts to v1, 153 validation frames from the three audited windows. Validation windows start at 600 s,
  and the patches come from 615 s onward, so no patch overlaps them in time.

**Not done.** No model has been trained on it. The Colab run and its pre-declared criteria are in the next subsection. The drone-free windows cannot show a gain, only a regression, so a clean score on them
does not prove the drone is fixed. Continuity-based candidate selection (prefer candidates near the previous ball
position) is not implemented: without the retrained detector it risks locking onto the drone.

### Stage-3 run and pass criteria (declared 2026-10-21, before any stage-3 result exists)

Correction made before the run, after the user's review of the candidate gallery: samples 277 and 332, which I had counted
as rig, are not the rig. They were never mined into training, so the dataset and zip are unchanged. They were removed from the
D1 held-out set, which shrank from 11 to 9 samples, and the D1 limit was changed from 3 to 2 to keep the same proportion.

**Run.** `notebooks/yolox_stage3_drone_colab.ipynb` with `data/finetune/finetune_v2_drone.zip` (SHA-256
`14b65ced90ffe09c8904408183438856b9596b2e3a6612bbde21ba804d7c326c`, 253 MB, upload to `MyDrive/football_cv/`). Same recipe
as stage 1 from COCO weights: seed 0, 60 epochs, batch 8, 704x1280, no mixup, same YOLOX commit. The only change is the
dataset. One seed, so run-to-run noise is not separated from the effect of the negatives; the comparison is against the
single stage-1 checkpoint (`55ef734a...`). Checkpoint goes to `data/models/yolox_m_football4_drone/best_ckpt.pth`; the
stage-1 model stays the default until these are scored.

**Criteria (all must hold to adopt; a failed one is reported, not re-tuned):**

| # | Check | Pass | Stage 1 |
|---|---|---|---|
| D1 | Held-out rig samples (394, 396-401, 410, 411 of `poc_seg1rb`; no patch from this episode was in training): ball detections within 20 px of the rig at conf >= 0.10 | in at most 2 of 9, none at conf >= 0.30 | 9 of 9, max 0.82 (fails) |
| D2 | User-confirmed real balls (78, 86, 171, 341, 467, 849): a ball detection >= 0.10 within 10 px of the stage-1 tracked position | at least 5 of 6 | 6 of 6 |
| D3 | Pipeline scoring on the three audited windows, same tracker and floors as the stage-2 table: ball hits | >= 47 of 137 | 52 |
| D4 | Same: pooled HOTA | >= 0.681 | 0.6913 |
| D5 | Same: referee role recall (matched boxes) | >= 0.30 | 0.342 |
| D6 | Same: role accuracy | >= 0.928 | 0.9361 |

D1 and D2 are computed by `scripts/eval_drone_check.py` (run on stage 1 to produce the baseline column). D3-D6 use the
margins observed between stage-1 and stage-2 checkpoints (ball hits within 1-5, HOTA within 0.003, referee recall within
0.02), widened to roughly twice that, so a pass means "not measurably worse", not "better".

**Limits stated in advance.** D1 rests on nine frames from one rig episode, so it is weak evidence. The audited windows contain no rig, so D3-D6 can only
show no regression. The 15 mined episodes (`in_sample_rig` in the script output) are printed but never counted. Passing does
not show that the rig is gone in another stadium, camera angle or lighting.

## Rig as a person (dataset built 2026-10-22; model not trained)

**Problem.** The same hanging camera rig (grey dish on a pole, dark base) that the ball detector mistakes for the ball is
also detected as a **person** and tracked: tracks 83 and 87 (PoC samples 234-299), 156, 160 and 164 (396-419) in `poc_seg1rb`, and
tracks 22, 24 and 27 in the a2 window (17 samples). In a2 the role probe then calls it goalkeeper (`role_probe.md`). No position
rule separates it from a real keeper standing 21 m out, so the fix is in the detector. I checked each box by eye in a gallery
(`artifacts/rig_check/rig_person_gallery.jpg`); tracks 74 and 93 are corner flags, and tracks 2, 16, 133 are real players
that sit near a false ball, so none of those is used.

**Why the drone set did not cover it.** `drone_v2` skipped 21 of its 34 samples as "person in patch": the rig itself was the tracked
person. Its 15 patches teach "no ball here", and some happen to contain the rig, but nothing in it labels the rig's person box as
background.

**What was built.**

- `football_cv.detection.negatives.mine_person_box_patches` (4 tests): a 160 px patch centred on the largest rig box of a
  sample, skipped when a real person (a track that is not a rig track) overlaps it. `mine_drone_negatives.py --rig-tracks 83,87 --label rig`
  runs it. Samples 234-299 of `poc_seg1rb`, every third one (22 candidates): **13 patches**, 9 skipped (no rig box in the sample
  or another person in the patch). I looked at all 13; each shows only the rig and grass.
- `data/finetune/v3_rig` = `v2_drone` (drone patches, 75 images) plus the 13 rig patches x 5 = 65 images: 1507 train images,
  box counts identical to v1 and v2, the same 153 validation frames. Zip `data/finetune/finetune_v3_rig.zip`
  (SHA-256 `eca1867acb431214dc1aa9ffa74525a7596046d467f1732f8f6301204ff79967`, 254 MB). Because it contains the drone patches too, a
  single run answers both D1-D6 and R1-R3 below.
- `notebooks/yolox_stage3_rig_colab.ipynb`: the stage-3 notebook with the dataset, experiment and output names changed
  (`yolox_m_football4_rig`) and the negative-image count assert set to 140. Recipe, seed and hyper-parameters are unchanged.
  The checkpoint goes to `data/models/yolox_m_football4_rig/best_ckpt.pth`; stage 1 stays the default until it is scored.

**Held out.** The rig in a2 (2707 s, a different place and scale) and PoC samples 396-419 (679-684 s) contribute no training patch.
The training patches are one 13 s episode of one rig (234-299), so this is one object seen in one camera move.

### Criteria (declared 2026-10-22, before any stage-3b result exists; a failed one is reported, not re-tuned)

Detections are scored at the pipeline floor (person 0.25), by `scripts/eval_rig_person_check.py`, against the stage-1
tracked rig boxes (IoU >= 0.30, any person class).

| # | Check | Pass | Stage 1 (measured) |
|---|---|---|---|
| R1 | a2 rig samples (17): person detection on the rig | at most 3 of 17, none at conf >= 0.50 | 13 of 17, max 0.751 (fails) |
| R2 | PoC samples 396-419 (23): the same | at most 5 of 23, none at conf >= 0.50 | 11 of 23, max 0.807 (fails) |
| R3 | Real people kept: matched ground-truth person boxes on the three windows, same chain as stage-3b roles (`f5s` configs, checkpoint swapped), `eval_roles_detector.py` | at least 3,265 of 3,298 (99%) | 3,298 |
| D1-D6 | As declared above, unchanged | as above | as above |

R3 and D3-D6 need the pipeline run on the three windows with the new checkpoint. In-sample rig (PoC 234-299, 46 samples; 9 detected
with stage 1) is printed, never counted.

**Limits stated in advance.** One rig, one training episode, one seed, one match, one stadium. R1 and R2 are 17 and 23 frames
of the same two scenes, so they are weak evidence, not a rate. a2 is also one of the three validation windows that `best_ckpt` is
chosen on (COCO AP), so a2 is not untouched by checkpoint selection, although 17 rig boxes barely move that AP. R3 shows no loss
of real people, not a gain. Passing says nothing about other rigs, stadiums or camera angles.

## Not claimed

- No speed, memory or accuracy figure for YOLOX on this machine beyond the above.
- The pass thresholds are targets, not predictions.

## SAM 3 pseudo-labels (Colab notebook written, not run)

`notebooks/sam3_annotate_colab.ipynb` runs `facebook/sam3` (gated; needs your Hugging Face access and a Colab secret
`HF_TOKEN`) on `sam3_inputs_inter_napoli_v1.zip` (112 random wide-view frames from `scripts/sample_match_frames.py`,
seed 0, at least 60 s away from the evaluation windows). It uses text prompts per class and writes, to
`MyDrive/football_cv/sam3_out/inter_napoli_v1/`, one annotated JPEG and one JSON per frame plus `all_detections.json`.
Frames are saved one at a time and skipped on re-run.

- Ambiguous person boxes (goalkeeper/player/referee scores within 0.15) are flagged, not resolved.
- The ball is searched on the full frame and on 3x2 tiles; boxes over 40 px are dropped. Thresholds are starting points.
- The local 4 GB route was shelved; no VRAM or speed figure exists for it.
- Pseudo-labels carry SAM 3's errors. Every batch is audited in CVAT and its error rate recorded before it can enter
  training. Held-out windows stay out of training whatever the source.
- The frames are from the same match as the validation windows, so training on them inflates the held-out score.
- The JSON has a `condition` field (null) to fill in per frame (day/night, weather) for the multi-condition dataset.
- `build_finetune_dataset.py` does not read SAM 3 exports yet.
