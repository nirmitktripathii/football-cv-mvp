# Plan: a 60 s held-out ground-truth window

Status: **plan only, nothing annotated or run.** Written 2026-10-22. It needs two decisions from you (section 6).

## 1. Why

Every score we have comes from three 10 s windows (a1, a2, a3) that were also used to choose settings, and from a clip
(600-780 s) where hard negatives were mined. So (a) nothing is scored out of sample, and (b) 10 s is too short to show
what Re-ID is for: an identity that has to survive a pan, an occlusion or a re-entry. One 60 s window, annotated once and
**never used to tune anything**, answers both.

## 2. Window choice

- Not inside 600-780 s (PoC clip, drone and rig patches mined there), and not within 30 s of a1, a2 or a3.
- Must contain: open play, at least one set piece or stoppage, both goalkeepers or one in full view, the referee,
  at least one camera pan or zoom, and the ball visible for most of it.
- Selection method: a one-frame-per-second survey of 2-3 candidate minutes (cheap, no detector), then you pick one.
  I choose by looking at the survey images; I do not choose by pipeline results, so the window is not picked to look good.

## 3. What gets annotated

Same schema and labels as a1-a3 (person with role player / goalkeeper / referee, ball, ids), at **5 fps = 301 frames**
(six times a 10 s window; 10 fps ground truth would double it for no gain, the scorer matches by source frame).

## 4. Procedure (the existing one, `docs/skills/football-annotation.md`)

1. Freeze the pipeline: config, detector checkpoint hash and commit are written to `window.json` **before** it is run.
2. Run the pipeline on the window at the default 10 fps (about 6 min of detection on the GTX 1650, plus the one-off model
   load, plus calibration keyframes).
3. Pre-fill CVAT with the pipeline's own tracks, so the work is correcting, not drawing.
4. I correct boxes, roles and identities with a changelog; uncertain identities are marked, not guessed.
5. You audit at least 20 frames spread across the 60 s (about 7%); I report the audit error rate beside every metric.
6. Raw CVAT export is kept untouched next to the corrected file.

## 5. What it scores, and the criteria

To be **declared before step 2 runs**, at the same granularity as D1-D6 and R1-R3. Proposed (you confirm the numbers):

| | Measure | Proposed pass line |
|---|---|---|
| G1 | HOTA on the 60 s window | at least 0.70 (the three 10 s windows give 0.87; 60 s is harder) |
| G2 | IDF1 on the 60 s window | at least 0.70 |
| G3 | Role accuracy on matched on-pitch boxes (roles stage on) | at least 0.95 |
| G4 | Ball hit rate | report only; the detector is known to miss 40% |
| G5 | Identity breaks per player over 60 s, with and without the linker | report only: this is the first number for what Re-ID buys |

Whatever comes out is reported as it is. No setting is changed afterwards to fix it; a change would need a new window.

## 6. What I need from you

1. Approve the idea of selecting the window from a survey, and the proposed G1-G3 lines (or change them).
2. About 30-45 minutes later, the audit of 20 or more frames in CVAT. The clicking and merging are mine.

## 7. Time (estimate, not measured)

Survey and window choice: about an hour of my time. Pipeline run and calibration keyframes: about 30-45 minutes. Correction
of 301 frames: roughly six times the effort of one 10 s window, so the longest step, about half a working day of my time.
Your audit: 30-45 minutes. None of it needs a GPU beyond the local GTX 1650.
