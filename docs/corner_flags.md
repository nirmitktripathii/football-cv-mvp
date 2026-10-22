# Corner flags tracked as people

Status: **criteria declared 2026-10-22, before the rule was run on anything.** Results are added below the line at the end.

## Problem

A corner flag (a pole with a flag at each pitch corner) is detected as a person when the camera shows a corner. On the
3-minute PoC clip (`poc_seg1rb`) four person tracks are flags, labelled by looking at crops (`artifacts/flag_check/`):
74 (26 rows), 93 (18), 84 (3) and 108 (7). They carry role `player`. A flag is a constant of the pitch: it stands on a
corner point and never moves.

## Rule (`calibration/offpitch.py: drop_corner_flags`)

A person track is a corner flag when it has at least `min_rows` (2) pitch-positioned rows, its role is neither referee
nor goalkeeper, the median distance of its projected foot to the nearest pitch corner is at most `radius_m` (2.5), and the
standard deviation of its x and of its y position are each at most `still_m` (0.6). The radius is wide on purpose: when the
detector box covers only the upper pole the foot projects up to 2 m beyond the real base (track 84: 2.2 m, 108: 1.95 m).

Config: `offpitch.corner_flags.{enabled, radius_m, still_m, min_rows}`; off by default, on in the working config
(`inter_napoli_4class.yaml`) only if the criteria below pass. It runs in `calibrate` beside the off-pitch rule and removes
rows from `tracks_projected.parquet`; `tracks.parquet` is untouched, like the existing off-pitch drop.

## Criteria (fixed now)

- **F1** all four labelled flag tracks (74, 93, 84, 108) are dropped on the PoC clip.
- **F2** every other track the rule drops on the PoC clip is looked at; a real player, referee or goalkeeper among them
  fails the rule. Photographers, a ball boxed as a person and other non-players are acceptable and listed.
- **F3** on the three audited windows (a1, a2, a3) no ground-truth person box is matched (IoU >= 0.30) by a dropped row.
- **F4** unit tests: flag dropped; a mover near a corner, a still track away from the corner, referee/keeper, single-row
  track and non-person rows all kept.

## Known limits (stated before the run)

- The radius and stillness limits were chosen after seeing the position statistics of these four flags and about twenty
  neighbouring tracks on the PoC clip, so F1 is in-sample. F3 is the held-out part, and a1-a3 contain few or no flags.
- Four flags in one clip is a small count. Another stadium, another camera or a flag the detector calls `referee` is untested.
- A player labelled `player` who stands still within 2.5 m of a corner for a whole track (for example a corner taker
  waiting) would be dropped. Whether that happens is what F2 and F3 look at.

---

## Result (2026-10-22): the rule FAILS F2 and F3 and stays off

Scored once with `scripts/eval_flag_rule.py`; no setting was changed after seeing the result.

| Criterion | Result |
|---|---|
| F1 labelled flags dropped (PoC) | **pass**: 74, 93, 84, 108 all dropped, none missed |
| F2 nothing real dropped (PoC) | **FAIL**: seven other tracks dropped (71, 72, 77, 81, 90, 99, 104). By eye (`artifacts/flag_check/poc_dropped_other.jpg`): 71 and 72 are the **referee** in neon yellow, detected as `player` (23 rows; 81, 2 rows, is probably the same referee); 77, 90 and 104 are photographers; 99 is the **ball** boxed as a person. |
| F3 no GT person box lost (a1-a3) | **FAIL**: 2 GT boxes. Window a2 track 30 (2 rows, pitch position 0.1 m, 67.1 m) is the away player waiting at the corner flag before the kick (GT `gt_id` 18, role player). a1 and a3 lose nothing and have no flag tracks. |
| F4 unit tests | pass (`tests/test_offpitch.py`) |

**Reading.** Position plus stillness identifies "a thing standing on a corner", and people stand still on corners: the
corner taker before the kick and the referee who is positioned at the corner. A flag cannot be told from them by pitch
position, stillness or box shape (box width/height: flags 0.37-0.62, the dropped referee 0.40, the taker 0.42, all players
0.29-0.53), so the rule trades real people for flags. The photographers and the ball it also removed are an accident, not a
design.

**Decision.** Rule left off everywhere (`offpitch.corner_flags.enabled: false`, not enabled in the working config). It is
kept as an option because it is tested and documented. What separates a flag from a person is appearance (a yellow flag
on a pole), so the fix belongs with the detector, as for the drone and the camera rig: hard-negative patches of flags.
The four flag tracks here, plus the corner tracks of other windows, give the patches. That needs a GPU run to test.

A different rule would have to be declared as a new rule with new criteria and a held-out check; I did not tune this one.
