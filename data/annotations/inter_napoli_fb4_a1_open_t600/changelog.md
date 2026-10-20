# Changelog: inter_napoli_fb4_a1_open_t600

Pre-fill: `prefill_cvat.xml` (4-class detector + ByteTrack). Output: `corrected_cvat.xml`. Source: det = raw detector box, interp = linear interpolation, manual = read by eye. Track ids below are this script's working ids; the output renumbers them.

| # | op | target | frames | detail | reason | source |
|---|---|---|---|---|---|---|
| 1 | seed | 47 tracks | 0-50 | offline relink of raw detections (conf >= 0.5, NMS 0.5, kit-colour gate) | the online tracker emits nothing for sample 0 and fragments tracks | det |
| 2 | delete | track 26 | 8,11 | 2 boxes removed | cameraman at the touchline, not a player or official |  |
| 3 | delete | track 39 | 36,40 | 2 boxes removed | staff member at the bottom touchline, outside the evaluation scope |  |
| 4 | delete | track 34 | 25 | 1 boxes removed | spectator or photographer in the stand |  |
| 5 | delete | track 37 | 35 | 1 boxes removed | camera dolly or debris on the touchline near the corner flag, detected as a person |  |
| 6 | delete | track 38 | 36 | 1 boxes removed | camera dolly or debris on the touchline near the corner flag, detected as a person |  |
| 7 | delete | track 40 | 37 | 1 boxes removed | camera dolly or debris on the touchline near the corner flag, detected as a person |  |
| 8 | delete | track 41 | 38 | 1 boxes removed | camera dolly or debris on the touchline near the corner flag, detected as a person |  |
| 9 | delete | track 42 | 39 | 1 boxes removed | camera dolly or debris on the touchline near the corner flag, detected as a person |  |
| 10 | delete | track 33 | 25 | 1 boxes removed | static artefact on the pitch (a white dash fixed in the image), not a person |  |
| 11 | relabel | track 22 | 0-2 | person -> ignore | person at the very edge of the frame, partly out of view; role cannot be told, so ignore rather than guess |  |
| 12 | relabel | track 6 | 0-50 | role/team player/unknown -> referee/none | neon-yellow kit, matches the match referee; the detector called him a player |  |
| 13 | relabel | track 13 | 0-50 | role/team player/unknown -> goalkeeper/away | yellow goalkeeper: the red (away) players defend his goal and clear the ball from it at frames 41-50, so he is the away keeper (the home keeper is the dark-grey one seen in window a2); the detector called him a player |  |
| 14 | drop boxes | track 5 | 1 | 1 boxes removed | frame 1 box covers only the legs; the other track's box on the same player is the full body |  |
| 15 | merge | track 5 -> 23 | 0 | 1 boxes moved, 0 duplicates dropped (-) | number 17, dark kit, one track split at the start |  |
| 16 | merge | track 24 -> 7 | 7,9-50 | 43 boxes moved, 2 duplicates dropped (6,8) | same red player, duplicate box at frames 6-8 |  |
| 17 | drop boxes | track 27 | 9 | 1 boxes removed | partial box on the same person; the kept track's box is used |  |
| 18 | merge | track 27 -> 3 | 10 | 1 boxes moved, 1 duplicates dropped (8) | same red player, tracker lost him at frame 10 |  |
| 19 | drop boxes | track 25 | 6 | 1 boxes removed | partial box on the same person; the kept track's box is used |  |
| 20 | merge | track 25 -> 12 | 7-24 | 18 boxes moved, 0 duplicates dropped (-) | same red player running to the goal mouth, split in three |  |
| 21 | merge | track 31 -> 12 | 25-31,33-41,45 | 17 boxes moved, 1 duplicates dropped (24) | same red player running to the goal mouth, split in three |  |
| 22 | drop boxes | track 28 | 17 | 1 boxes removed | partial box on the same person; the kept track's box is used |  |
| 23 | merge | track 28 -> 11 | 18-25,28 | 9 boxes moved, 0 duplicates dropped (-) | same dark player: running forward then jumping in the goal area, split in three |  |
| 24 | drop boxes | track 32 | 28 | 1 boxes removed | partial box on the same person; the kept track's box is used |  |
| 25 | merge | track 32 -> 11 | 26-27,30,33-36,39-43,47 | 13 boxes moved, 1 duplicates dropped (25) | same dark player: running forward then jumping in the goal area, split in three |  |
| 26 | drop boxes | track 29 | 18 | 1 boxes removed | partial box on the same person; the kept track's box is used |  |
| 27 | merge | track 29 -> 16 | 20-33,35-50 | 30 boxes moved, 1 duplicates dropped (19) | same red player, split at frame 18 |  |
| 28 | drop boxes | track 12 | 41,45 | 2 boxes removed | box sits on a dark player in the scrum, not on this red player |  |
| 29 | set box | track 11 | 41 | 1 boxes replaced or added | frame 41: the crouching dark player had no box (the existing box duplicated the dark player to his right); the box dropped from the red track fits him | manual |
| 30 | merge | track 43 -> 12 | 41-50 | 10 boxes moved, 0 duplicates dropped (-) | same red player, now carrying the ball: tracker restarted him at frame 41 |  |
| 31 | drop boxes | track 11 | 47 | 1 boxes removed | duplicate of the neighbouring dark player's box |  |
| 32 | merge | track 45 -> 11 | 44-50 | 7 boxes moved, 1 duplicates dropped (43) | same dark player: went to ground behind the red player at frame 43, tracker restarted him |  |
| 33 | drop boxes | track 36 | 29-30 | 2 boxes removed | partial box on the same person; the kept track's box is used |  |
| 34 | merge | track 36 -> 19 | 31-48 | 18 boxes moved, 0 duplicates dropped (-) | same dark player on the edge of the box, split in three |  |
| 35 | drop boxes | track 46 | 48 | 1 boxes removed | partial box on the same person; the kept track's box is used |  |
| 36 | merge | track 46 -> 19 | 49-50 | 2 boxes moved, 1 duplicates dropped (44) | same dark player on the edge of the box, split in three |  |
| 37 | delete | track 35 | 25 | 1 boxes removed | partial duplicate of a player inside the scrum |  |
| 38 | delete | track 30 | 24-25 | 2 boxes removed | partial duplicate of a player inside the scrum |  |
| 39 | delete | track 44 | 43 | 1 boxes removed | single-frame fragment |  |
| 40 | fill gap | track 0 | 35 | 1 boxes from low-confidence detections | short gap inside one person's track | det |
| 41 | fill gap | track 0 | 20,33-34 | 3 boxes interpolated | short gap inside one person's track | interp |
| 42 | fill gap | track 11 | 31-32,37 | 3 boxes from low-confidence detections | short gap inside one person's track | det |
| 43 | fill gap | track 11 | 29,38 | 2 boxes interpolated | short gap inside one person's track | interp |
| 44 | fill gap | track 12 | 32 | 1 boxes from low-confidence detections | short gap inside one person's track | det |
| 45 | fill gap | track 16 | 34 | 1 boxes from low-confidence detections | short gap inside one person's track | det |
| 46 | fill gap | track 20 | 30 | 1 boxes interpolated | short gap inside one person's track | interp |
| 47 | fill gap | track 23 | 26-27 | 2 boxes interpolated | short gap inside one person's track | interp |
| 48 | set box | track 12 | 32 | 1 boxes replaced or added | frame 32: gap-fill box duplicated the dark player's; replaced by the midpoint of this player's neighbouring boxes (hidden behind him) | manual |
| 49 | add track | track 100 (ball None/None) | 1-50 | 50 boxes: 43 manual, 7 interpolated (15,26-28,30,32,49) | ball in play, positions read by eye from zoomed crops (about +-4 px); frame 0 hidden at a player's feet, frames 26-28 hidden in the scrum, so left unannotated | manual+interp |
| 50 | drop boxes | track 100 | 26-28 | 3 boxes removed | ball not visible in the scrum; no position invented across a gap |  |
