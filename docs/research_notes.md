# Research notes

Decisions made while building the MVP, what was tried, and what did not work. Read alongside
`docs/evaluation.md` for the numbers.

## Footage

- One public broadcast clip: the official ISL full-replay upload of Bengaluru FC v Sporting Club
  Delhi (GW1, 2026-10-10), section 3120–3480 s, downloaded with `yt-dlp --download-sections`.
  It is copyrighted and stays in the git-ignored `data/raw/`.
- The section contains a goal (the 1-0), the replay block after it, the kick-off and long
  wide-angle passages: enough to stress camera cuts, replays and panning.

## Detection

- YOLO11m with COCO weights finds players well at imgsz 1280. The ball (≈ 8 px) is the weak point:
  49.5 % frame coverage, median confidence 0.22, frequent false positives. A football-specific
  detector (fine-tuned on SoccerNet or similar) is the obvious upgrade.
- FP16 on the GTX 1650 produced NaN boxes; FP32 is pinned.

## Calibration: what worked and what did not

- **Automatic line detection alone was not reliable** on this broadcast: ad boards, shadows and
  stand edges produce Hough lines parallel to real markings. Landmark naming stays a human step;
  the tool only proposes candidates.
- **Extrapolation is the main failure mode.** The first segment-13 keyframe (sample 1200) had all
  its landmarks in the upper right of the image. It fitted them to sub-metre accuracy but put the
  near touchline 29 m off. Replacing it with sample 1160 (a full box) plus two `on_line` points on
  the near touchline fixed it. Lesson: landmarks must span the image, and in-sample residuals
  cannot reveal this.
- **Exactly determined fits hide label errors.** A 4-point + 1-line keyframe (1250) validated with
  near-zero residuals but projected implausibly; it was removed.
- **Point-on-line constraints** (one DLT equation per point) are cheap and constrain the direction
  that box landmarks leave free, typically towards the near touchline.
- **Camera-motion propagation drifts** with pan length. Corrected: an early check on one keyframe pair gave
  0.4 m over 4 s, but hand-labelled held-out landmarks on 12 propagated frames (2–12 s from a keyframe)
  average about 4 m (docs/evaluation.md §10); 4–10 m (median) over ~30 s.
- **Line-chamfer refinement does not rescue large drift.** Aligning projected pitch lines to the
  image's white-line mask converges to a neighbouring line when the starting error is several
  metres. It is kept for keyframes (small corrections, checked against the anchors) and not used on
  propagated frames.
- The adopted rule is **reject rather than guess**: frames over 12 s from a keyframe are dropped
  (95 frames). Adding keyframes is the honest fix within this design.

## Tracking and identity

- ByteTrack works within a shot but has no appearance model, so ids break at occlusions and always
  at cuts. Per-track distance is a fragment. Re-identification (jersey numbers, appearance
  embeddings) is out of scope for the MVP.
- Team assignment by torso colour is adequate for blue v white kits; goalkeepers and referees land
  in "other"; 162 short or small tracks stay "unknown".

## Direction of play and sequences

- Direction was measured, not assumed: near each goal the deepest outfield player almost always
  belonged to the defending side (149 v 33 and 263 v 27 frames). Home (BFC) attacks left in this
  window.
- Candidate sequences are deliberately labelled candidates. Without possession, a final-third ball
  window says *where* the ball went, not *who* had it. The nearest-player share is shown as a proxy;
  it is below 0.4 for the direction-based attacking team in 3 of the 5 candidates (SEQ1, SEQ3, SEQ4),
  and all 5 reach 9–12 s from a keyframe, so all are rated low confidence. The proxy is weak in its
  own right: during an attack the nearest player is often a defender.

## xT and Atomic-VAEP

- The xT grid is socceraction's `ExpectedThreat` fitted on 1,890 StatsBomb open-data games. On the
  clip it is used as a *zone value* of the ball's position, since there are no actions to rate,
  and it is labelled exploratory.
- Atomic-VAEP trains and evaluates cleanly on public SPADL data (AUROC ≈ 0.87, Brier below the base
  rate). On the clip it is INCOMPLETE: deriving SPADL actions needs possession, action segmentation
  (pass, dribble, shot, tackle) and a ball track with far fewer gaps than this detector gives.
  Inventing that stream would make any VAEP number meaningless.

## Not done, by decision

- No PPDA, pressing or pass networks: they need events.
- v0.1: no interpolation to fill gaps in animations. Superseded: `reconstruction/` now fills short
  gaps with provenance and uncertainty, shown distinctly from observations (`docs/evaluation.md` §9).
- No claims about individual players.
