# Annotation guide: tracking ground truth for the Inter–Napoli windows (phase 0)

Every tracking number in [implementation_plan_tactical.md](implementation_plan_tactical.md) is measured against
these annotations. Until they exist, no HOTA, IDF1, team-accuracy or ball figure is reported for this match.

## 1. What you get

`scripts/make_eval_window.py` runs the current tracker (ablation A) on each window in
[configs/inter_napoli.yaml](../configs/inter_napoli.yaml) and writes:

| file | tracked in git | what it is |
|---|---|---|
| `data/processed/inter_napoli_<window>/cvat_frames.zip` | no | the sampled frames, `000000.jpg` … at 5 fps; CVAT frame *i* is `sample_id` *i* |
| `data/annotations/inter_napoli_<window>/prefill_cvat.xml` | yes | tracker output as **CVAT for video 1.1**, to import and correct |
| `data/annotations/inter_napoli_<window>/labels.json` | yes | the label set, for CVAT's *Raw* label editor |
| `data/annotations/inter_napoli_<window>/window.json` | yes | window times, frame count, frame size, pipeline commit |

| window | video seconds | situation |
|---|---|---|
| `w1_open_t600` | 600–630 | first-half open play; overlaps the 60 s baseline clip |
| `w2_corner_t2695` | 2695–2725 | corner at 44:53, crowded box, referee in neon yellow next to the goalkeeper |
| `w3_open_t4800` | 4800–4830 | second-half open play, wide view, both assistants and spare balls visible |

You save the corrected export as `data/annotations/inter_napoli_<window>/annotations.xml`. No script
writes that file.

## 2. CVAT workflow, per window

Use either CVAT Online (cvat.ai) or a self-hosted CVAT. The frames are broadcast footage, so decide whether
uploading them to a third-party service is acceptable before you choose the online version.

1. **Create the task.**
   - Name it after the window.
   - Under *Labels*, switch to *Raw* and paste the contents of `labels.json`.
   - Under *Select files*, upload `cvat_frames.zip`.
   - In *Advanced configuration*, raise *Image quality* to 95 or more; at the default the ball is a smudge. Leave the sorting lexicographical.
   - Use one job for the whole window.
2. **Import the pre-fill.**
   - Task → *Actions* → *Upload annotations*, format **CVAT 1.1**, file `prefill_cvat.xml`.
   - Every pre-filled box is a keyframe, so editing one box changes only that frame.
   - Where the tracker lost someone, the track ends with an *outside* box.
3. **Correct it** using the rules in section 3. Expect roughly 25 people per frame, at 5 frames per second of window.
4. **Export.**
   - Task → *Export task dataset*, format **CVAT for video 1.1**, with images unticked.
   - Unzip the download and save its `annotations.xml` as `data/annotations/inter_napoli_<window>/annotations.xml`.
   - A *CVAT for images* export is rejected by the scorer, because it loses the track ids.
5. **Score it.**

   ```bash
   .venv/Scripts/python scripts/eval_tracking.py --config configs/inter_napoli.yaml --label A_baseline
   ```

   This writes `artifacts/inter_napoli/eval_tracking_A_baseline.json`. Windows that have no `annotations.xml`
   yet are listed under `missing` and left out of the pooled score.

## 3. Rules

### Who gets a box

- **`person`:** everyone involved in play — all 22 players, the referee, and both assistant referees, wherever they stand in view.
- **`ignore`:**
  - substitutes warming up, coaching staff, ball boys, photographers and stewards;
  - spare balls on the touchline.

  Draw one `ignore` box (or track) around such a group. The scorer discards predictions inside it, so they are neither false positives nor hits.

### Boxes

- Draw tight boxes around the whole body, head to feet. For a partly hidden person, draw the box where the whole body is (estimated) and tick *occluded*.
- When a person is completely hidden or out of view, mark the track *outside* from that frame. Do not guess a position.
- Pre-filled boxes come from the tracker. Check that each one is tight rather than accepting it as drawn. Accepting loose boxes inflates the localisation score (LocA, MOTP).

### Identity — the most important rule

- **One CVAT track is one real person for the whole window.** If someone leaves the view and comes back, switch *outside* off on the **same** track; do not start a new one. IDF1 and AssA depend on this.
- Where the tracker split one person into several tracks, merge them with *Merge* in the left toolbar.
- Where one tracker track jumps from one person to another, *Split* it at the jump and merge each piece into the right person.
- Delete false-positive tracks (shadows, crowd, advertising boards) that are not inside an `ignore` box.
- Add any person the tracker missed as a new track.

### Attributes on `person` (one value per track)

| `role` | `team` |
|---|---|
| `player` | `home` (Inter, dark navy) or `away` (Napoli, red) |
| `goalkeeper` | the side the goalkeeper plays for, **by position** (which goal they defend), not by kit colour; Inter's goalkeeper wears neon yellow like the officials |
| `referee` (referee and both assistants) | `none` |
| `unknown` | `unknown` — only when you really cannot tell; these boxes are left out of the label metrics |

The pre-fill sets `player`/`home` or `player`/`away` from the tracker's kit clustering, and `unknown` for
everything else. It is a guess: check every track.

### Ball

- Label only the match ball as `ball`, as one track. Box the visible ball; it is often only 5–10 px across.
- Mark it *outside* whenever it cannot be seen (in the air out of frame, hidden by a player, blurred beyond recognition). Do not guess where it is.
- Spare balls are `ignore`.

## 4. What the scorer reports

`football_cv.evaluation.mot` reports the following.

**People:**
- HOTA, DetA, AssA and LocA, computed with TrackEval's own code (vendored, MIT, commit `12c8791`);
- MOTA, MOTP, IDSW, Frag and MT/PT/ML;
- IDF1, IDP and IDR;
- predicted tracks per true person (fragmentation).

**Labels,** on boxes matched at IoU ≥ 0.5:
- outfield team accuracy, per box and per true track;
- goalkeeper affiliation;
- how often a referee was given a team (the target is 0).

Coverage, meaning the share of boxes that got a home/away call at all, is reported separately from the accuracy of those calls. Wilson 95 % intervals are included. The per-box intervals are optimistic because a track's boxes are not independent.

**Ball:**
- recall and precision with a 10 px centre radius;
- centre error;
- the longest run of frames where a visible ball was missed.

Windows are scored as separate sequences, then pooled with TrackEval's `combine_sequences`.

## 5. Limitations of this ground truth

- **Single annotator.** There is no inter-annotator agreement figure, so small score differences (about 1 HOTA point) between ablations are within annotation noise.
- **Anchoring.** Correcting a pre-fill biases the result towards the tracker that produced it: boxes that look acceptable stay as drawn. The self-check below only proves the plumbing, not the ground truth.
- **Small sample.** Three 30 s windows at 5 fps. Goalkeeper and referee results rest on a handful of tracks each; read them with their intervals.
- **Self-check.** `eval_tracking.py --gt-name prefill_cvat.xml --label self_check` scores the pre-fill against the tracks it came from and must give HOTA = MOTA = IDF1 = 1. It is not a result.
