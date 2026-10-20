# 4-class football detector

Optional detector backend: a fine-tuned model with the classes `ball`, `goalkeeper`, `player`, `referee`.
Select it with `configs/inter_napoli_4class.yaml` (it only changes `detection.model`). The default
pipeline is unchanged.

## The weights

| | |
|---|---|
| File | `models/football_4class/best.pt` (git-ignored, `*.pt`) |
| Source | the user's Google Drive file `1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK` |
| Size / SHA-256 | 172,628,985 bytes / `53fcd08a5e5283d7bba705f07fc04022a6ab73a2ba2a847d3bc8143c9ecad2eb` |
| Architecture | Ultralytics v8 `DetectionModel` (DFL head), 86 M parameters, class names `{0: ball, 1: goalkeeper, 2: player, 3: referee}` |
| Loader | the installed `ultralytics` loads it directly. It is **not** a legacy YOLOv5 checkpoint, so no `yolov5` package is needed |
| Licence | Ultralytics code is AGPL-3.0. The training data's licence and provenance are unknown to this repo; confirm them before publishing results or weights |

A `.pt` file is a pickle. Before loading, its imports were listed without executing it (only `torch.*` and
`ultralytics.*` classes). Do the same for any other checkpoint you drop in.

## How it is used

- `detector.class_roles` reads the model's class names. 4-class names map `player`/`goalkeeper`/`referee` to
  `object_class = person` with the class in a new `role` column; COCO models are unchanged (`role` empty).
- Class-agnostic NMS in 4-class mode, so a person gets one box, not one per role.
- `role` is carried through tracking; each track gets its **majority role**.
- Teams: only `player` (or role-less) tracks feed the kit-colour clustering, so referees and keepers no
  longer pull the two cluster centres. Referee tracks get `team_id = other` (no team). Goalkeeper tracks get
  `unknown`: a keeper's shirt does not name a team, and the planned rule (affiliation by position, which
  needs calibrated pitch coordinates) is not built yet.
- The CVAT pre-fill uses the detector role (referee also gets team `none`), which saves annotation clicks.
- Off-pitch balls and people are still removed by the pitch mask in the shots stage; the 4-class model does not
  replace it (see the single-frame check below).

## What has and has not been measured

One frame (`w2_corner_t2695`, sample 85, imgsz 1280, conf 0.10): 26 players, 1 goalkeeper, 1 referee, 3
balls. The goalkeeper box and two of the balls sit on the advertising boards, outside the pitch, so the
pitch mask is still needed. A yellow-kit person in the middle of the pitch is labelled `player` and may be
the referee. One frame says nothing about accuracy.

**No improvement is claimed.** The comparison with the YOLO11m baseline (HOTA, team, goalkeeper, referee,
role and ball metrics) is made on the three evaluation windows once their ground truth is annotated; see
`docs/annotation_guide.md`.

## Roboflow check (not used by the pipeline)

- The pasted workflow `banana-ripeness-image-classification/general-segmentation-api` is a SAM 3 text-prompt
  segmenter. It returned HTTP 402 (no credits in that workspace).
- The Universe model `football-players-detection-3zvbc/1` answered through `inference-sdk` on one frame
  (22 players, no other class on that frame). It uploads frames to a third party and is credit-metered, so it
  is not part of the pipeline.
- `inference-sdk` pulls numpy 2.x. Install it in its own venv, never in the project venv (numpy is pinned
  to 1.26.4).
- The key file `ROBOFLOW_API_KEY.txt` is git-ignored and was never printed. The same key was pasted into a
  chat message, so rotate it in Roboflow (Settings, API Keys).

## Later: YOLOX fine-tune

Fine-tune on Colab with the user's annotated 4-class dataset (plus more public sets), save the best
checkpoint to Drive, copy it to `models/`, and add a YOLOX backend behind the same `role` output. The
`football_training_yolo_v5.ipynb` notebook in the reference repo trains YOLOv5, so its recipe needs adapting
to YOLOX. Hold out whole matches or clips, not random frames, so the score is not inflated by near-duplicate
frames.
