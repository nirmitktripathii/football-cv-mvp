# Changelog: inter_napoli_fb4_a2_corner_t2707

Pre-fill: `prefill_cvat.xml` (4-class detector + ByteTrack). Output: `corrected_cvat.xml`. Source: det = raw detector box, interp = linear interpolation, manual = read by eye. Track ids below are the pre-fill's; the output renumbers them.

| # | op | target | frames | detail | reason | source |
|---|---|---|---|---|---|---|
| 1 | delete | track 46 | 2,11-50 | 41 boxes removed | false positive: static 'ball' on the advertising board at about (786,130), not the ball in play |  |
| 2 | add track | track 100 (ball None/None) | 0-50 | 51 boxes: 51 manual, 0 interpolated (-) | the live ball (dribbled, then placed on the corner arc); the spare ball on the board edge near (625,100) is not annotated | manual |
| 3 | delete | track 24 | 5,7-8 | 3 boxes removed | corner-taker fragment; replaced by the complete corner-taker track |  |
| 4 | delete | track 32 | 23 | 1 boxes removed | corner-taker fragment; replaced by the complete corner-taker track |  |
| 5 | delete | track 42 | 48-50 | 3 boxes removed | assistant-referee fragment with no role; replaced by the complete official track |  |
| 6 | add track | track 101 (person player/away) | 0-50 | 51 boxes: 40 det, 11 interpolated (11-12,14,16-21,34,39) | away player at the corner flag, absent from the pre-fill (the tracker's pitch mask drops the touchline) | det+interp |
| 7 | add track | track 102 (person referee/none) | 0-50 | 51 boxes: 33 det, 18 interpolated (1-2,8,17-21,23-24,26-27,29-31,33,42,45) | assistant referee in neon yellow at the corner flag, absent from the pre-fill | det+interp |
| 8 | relabel | track 0 | 1-50 | role/team player/unknown -> referee/none | neon-yellow match referee; pre-fill said player/unknown |  |
| 9 | delete | track 18 | 2 | 1 boxes removed | bottom-left graphics sliver at (45,701), not a person |  |
| 10 | delete | track 22 | 4-6 | 3 boxes removed | pitch-side camera dolly (white dome, no person) |  |
| 11 | delete | track 28 | 10-12 | 3 boxes removed | pitch-side camera dolly (white dome, no person) |  |
| 12 | merge | track 10 -> 21 | 1-2 | 2 boxes moved, 2 duplicates dropped (3-4) | same person (home goalkeeper) before the tracker re-assigned the id |  |
| 13 | relabel | track 21 | 1-30 | role/team -> goalkeeper/home | same person (home goalkeeper) before the tracker re-assigned the id |  |
| 14 | merge | track 36 -> 21 | 33-37,40-47 | 13 boxes moved, 0 duplicates dropped (-) | same person: track 21 ends at (296,279) f30 and 36 starts 5 px away at f33 |  |
| 15 | extend | track 21 (goalkeeper) | 48-50 | box held at its f47 position (goalkeeper is stationary at the post) | track was lost 3 frames early | manual |
| 16 | merge | track 9 -> 26 | 1-6 | 6 boxes moved, 0 duplicates dropped (-) | same away player: track 9 leaves view f6, track 26 starts 6 px away at f9 |  |
| 17 | merge | track 17 -> 27 | 1-8 | 8 boxes moved, 0 duplicates dropped (-) | same home player: 17 ends f8, 27 starts 17 px away at f9 |  |
| 18 | merge | track 40 -> 6 | 38-40 | 3 boxes moved, 0 duplicates dropped (-) | same away player: 6 ends f34, 40 starts 15 px away at f38, 41 continues at f42 |  |
| 19 | merge | track 41 -> 6 | 42-48,50 | 8 boxes moved, 0 duplicates dropped (-) | same away player: 6 ends f34, 40 starts 15 px away at f38, 41 continues at f42 |  |
| 20 | merge | track 30 -> 33 | 13-24 | 12 boxes moved, 1 duplicates dropped (25) | same home player at the near post: 30 hands over to 33 at f25, 34 is the same person under a second id |  |
| 21 | merge | track 34 -> 33 | 34-39,42-46 | 11 boxes moved, 7 duplicates dropped (29-33,40-41) | same home player at the near post: 30 hands over to 33 at f25, 34 is the same person under a second id |  |
| 22 | delete | track 35 | 33-34,38 | 3 boxes removed | duplicate box on track 12 (IoU up to 0.59) |  |
| 23 | delete | track 38 | 35 | 1 boxes removed | duplicate of track 3 (IoU 0.55) |  |
| 24 | delete | track 39 | 36-37 | 2 boxes removed | duplicate of tracks 37/14 (IoU 0.55-0.70) |  |
| 25 | delete | track 43 | 48-49 | 2 boxes removed | duplicate of track 29 (IoU 0.67) |  |
| 26 | delete | track 44 | 49-50 | 2 boxes removed | duplicate of track 3 (IoU 0.56) |  |
| 27 | delete | track 45 | 50 | 1 boxes removed | duplicate of track 37 (IoU 0.70) |  |
| 28 | merge | track 25 -> 23 | 7 | 1 boxes moved, 0 duplicates dropped (-) | same red-kit runner: 23 ends f4, 25 starts 22 px away at f7 |  |
| 29 | relabel | track 23 | 4,7 | role/team -> player/away | same red-kit runner: 23 ends f4, 25 starts 22 px away at f7 |  |
| 30 | move box | track 2 -> 19 | 1 | box at (501,332,521,359) moved to the runner's track | id switch: the box is on the away runner (id 19 from f3), not on id 2 | manual |
| 31 | add boxes | track 19 | 0,2 | raw detections [507.0, 325.0, 534.0, 370.0] and [489.0, 307.0, 503.0, 353.0] added | same runner, missed by the tracker at f0 and f2 | det |
| 32 | delete | track 20 | 3 | 1 boxes removed | single-frame fragment; folded into the new walker track |  |
| 33 | add track | track 103 (person player/home) | 0-12 | 13 boxes: 10 det, 3 interpolated (6-7,9) | dark-kit (home) player walking left at the edge of the cluster, missed by the tracker f0-f12; f3 box comes from the old id 20 | det+interp |
| 34 | add track | track 104 (person player/away) | 47-50 | 4 boxes: 4 det, 0 interpolated (-) | red-kit player entering at the top of the area, missed by the tracker | det |
| 35 | add track | track 105 (person player/home) | 49-50 | 2 boxes: 2 det, 0 interpolated (-) | dark-kit player entering at the top of the area, missed by the tracker | det |
| 36 | known gap | frames 0-2 | 0-2 | one away player hidden behind id 13's box (two overlapping red players in the pre-fill's single box) is not annotated | boxes too uncertain to read by eye |  |
| 37 | fill gap | track 6 | 35-37,41,49 | 5 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 38 | fill gap | track 7 | 14 | 1 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 39 | fill gap | track 11 | 37,39 | 2 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 40 | fill gap | track 13 | 8,13-14,42-43 | 5 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 41 | fill gap | track 14 | 47 | 1 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 42 | fill gap | track 21 | 31-32,38-39 | 4 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 43 | fill gap | track 23 | 5-6 | 2 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 44 | fill gap | track 26 | 7-8 | 2 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 45 | fill gap | track 29 | 14-15,18-19,41-43,49 | 8 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 46 | fill gap | track 37 | 40-42,46,48-49 | 6 boxes interpolated | gap of <= 3 sampled frames inside one track; the person stays in view | interp |
| 47 | add frame 0 | 17 tracks | 0 | matched raw sample-0 detections to sample-1 boxes (camera shift [6.3, 10.7] px) | the tracker emits nothing for sample 0, so frame 0 had no person boxes at all | det |
