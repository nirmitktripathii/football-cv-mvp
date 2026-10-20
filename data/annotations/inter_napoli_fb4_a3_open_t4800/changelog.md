# Changelog: inter_napoli_fb4_a3_open_t4800

Pre-fill: `prefill_cvat.xml` (4-class detector + ByteTrack). Output: `corrected_cvat.xml`. Source: det = raw detector box, interp = linear interpolation, manual = read by eye. Track ids below are this script's working ids; the output renumbers them.

| # | op | target | frames | detail | reason | source |
|---|---|---|---|---|---|---|
| 1 | seed | 41 tracks | 0-50 | offline relink of raw detections (conf >= 0.5, NMS 0.5, kit-colour gate) | the online tracker emits nothing for sample 0 and fragments tracks | det |
| 2 | delete | track 32 | 25-27 | 3 boxes removed | static dark figure at the top touchline (stand or staff), not a player or official |  |
| 3 | delete | track 33 | 36 | 1 boxes removed | member of staff in a suit on the bottom touchline, outside the evaluation scope |  |
| 4 | delete | track 34 | 36 | 1 boxes removed | single detection on empty grass |  |
| 5 | delete | track 22 | 0,3 | 2 boxes removed | duplicate box on the red player of track 20 (frames 0 and 3) |  |
| 6 | delete | track 29 | 16 | 1 boxes removed | duplicate box on the red player of track 20 (frame 16) |  |
| 7 | delete | track 40 | 46 | 1 boxes removed | partial box on the referee inside the scrum; the referee's own track has his full box |  |
| 8 | relabel | track 28 | 13-14,16-22,25 | person -> ignore | neon-yellow official walking behind the touchline (probably the fourth official), not one of the three on-pitch officials; ignored rather than guessed |  |
| 9 | relabel | track 17 | 0-17,19-44,46-47,49-50 | role/team player/unknown -> referee/none | neon-yellow kit, the match referee; the detector called him a player |  |
| 10 | drop boxes | track 17 | 44,46-47,49-50 | 5 boxes removed | from frame 44 these boxes sit on the red player and the dark player in the scrum, not on the referee |  |
| 11 | drop boxes | track 36 | 39-40 | 2 boxes removed | partial box on the same person; the kept track's box is used |  |
| 12 | merge | track 36 -> 17 | 44-50 | 7 boxes moved, 0 duplicates dropped (-) | same referee: the tracker restarted him at frame 39 and again at 44 |  |
| 13 | set box | track 17 | 46 | 1 boxes replaced or added | frame 46: the referee is bent over beside the scrum and the detector split him into an upper and a lower box; replaced by their union | manual |
| 14 | relabel | track 19 | 0,3-23 | role/team player/unknown -> goalkeeper/away | yellow goalkeeper at the left goal: the away keeper (the home keeper is dark grey); the detector called him a player |  |
| 15 | relabel | track 31 | 24-45,47-50 | role/team player/home -> goalkeeper/home | dark-grey goalkeeper at the right goal: the home keeper; the detector called him a player |  |
| 16 | merge | track 39 -> 31 | 46 | 1 boxes moved, 0 duplicates dropped (-) | same goalkeeper: a box at frame 46 where the track had a gap |  |
| 17 | relabel | track 21 | 0-2,4-6,8-25,28-33 | role/team player/unknown -> referee/none | neon-yellow assistant referee on the top touchline |  |
| 18 | merge | track 38 -> 21 | 40-46,50 | 8 boxes moved, 0 duplicates dropped (-) | same assistant referee: the tracker lost him at frames 34-39 (touchline mask), then restarted him |  |
| 19 | set box | track 21 | 34-39 | 6 boxes replaced or added | frames 34-39: assistant referee visible near the touchline but not detected; straight line between his frame-33 and frame-40 boxes | interp |
| 20 | relabel | track 27 | 12,15-16,19,21,23-28,30-31,33-34,36-37,40-42 | role/team player/unknown -> referee/none | neon-yellow assistant referee with the flag on the bottom touchline |  |
| 21 | set box | track 27 | 0-11 | 12 boxes replaced or added | frames 0-11: assistant referee on the touchline, undetected; boxes at frames 0, 4 and 8 read by eye from zoomed crops (about +-3 px), the others interpolated between them and his frame-12 detection | manual |
| 22 | merge | track 10 -> 24 | 0-3 | 4 boxes moved, 0 duplicates dropped (-) | same home player, split at frame 4 |  |
| 23 | merge | track 23 -> 18 | 4,7-11 | 6 boxes moved, 0 duplicates dropped (-) | same home player, split in three |  |
| 24 | merge | track 26 -> 18 | 12-32,34-50 | 38 boxes moved, 0 duplicates dropped (-) | same home player, split in three |  |
| 25 | merge | track 30 -> 11 | 21 | 1 boxes moved, 0 duplicates dropped (-) | same red player: a box at frame 21 where the track had a gap, and a duplicate restart at frame 39 |  |
| 26 | drop boxes | track 37 | 39 | 1 boxes removed | partial box on the same person; the kept track's box is used |  |
| 27 | merge | track 37 -> 11 | 40-50 | 11 boxes moved, 0 duplicates dropped (-) | same red player: a box at frame 21 where the track had a gap, and a duplicate restart at frame 39 |  |
| 28 | merge | track 35 -> 8 | 39-42 | 4 boxes moved, 0 duplicates dropped (-) | same home player at the left edge, tracker restarted him at frames 39-42 |  |
| 29 | fill gap | track 4 | 35-36 | 2 boxes interpolated | short gap inside one person's track | interp |
| 30 | fill gap | track 5 | 37 | 1 boxes from low-confidence detections | short gap inside one person's track | det |
| 31 | fill gap | track 6 | 35 | 1 boxes interpolated | short gap inside one person's track | interp |
| 32 | fill gap | track 17 | 18 | 1 boxes from low-confidence detections | short gap inside one person's track | det |
| 33 | fill gap | track 18 | 33 | 1 boxes interpolated | short gap inside one person's track | interp |
| 34 | fill gap | track 19 | 1 | 1 boxes from low-confidence detections | short gap inside one person's track | det |
| 35 | fill gap | track 19 | 2 | 1 boxes interpolated | short gap inside one person's track | interp |
| 36 | fill gap | track 21 | 47-49 | 3 boxes from low-confidence detections | short gap inside one person's track | det |
| 37 | fill gap | track 21 | 3,7,26-27 | 4 boxes interpolated | short gap inside one person's track | interp |
| 38 | fill gap | track 24 | 32 | 1 boxes interpolated | short gap inside one person's track | interp |
| 39 | fill gap | track 25 | 21-23,27 | 4 boxes interpolated | short gap inside one person's track | interp |
| 40 | fill gap | track 27 | 17-18,22,32,35 | 5 boxes from low-confidence detections | short gap inside one person's track | det |
| 41 | fill gap | track 27 | 13-14,20,29,38-39 | 6 boxes interpolated | short gap inside one person's track | interp |
| 42 | set box | track 25 | 21-23 | 3 boxes replaced or added | frames 21-23: the home player is hidden behind the red player of the tackle (only his head shows); the gap fill had put the box on empty grass. Box = the red player's box plus the offset between the two, interpolated from frames 20 and 24 | interp |
| 43 | set box | track 25 | 27 | 1 boxes replaced or added | frame 27: hidden behind the red player; the gap fill had duplicated the red player's box. Box = the red player's box plus the interpolated offset | interp |
| 44 | set box | track 4 | 35-36 | 2 boxes replaced or added | frames 35-36: red-kit player grappling with the dark-kit player, almost fully overlapped; the existing boxes had swapped sides. Read by eye (about +-3 px), flagged for audit | manual |
| 45 | set box | track 5 | 35-36 | 2 boxes replaced or added | frames 35-36: dark-kit player grappling with the red-kit player, almost fully overlapped; the existing boxes had swapped sides. Read by eye (about +-3 px), flagged for audit | manual |
| 46 | add track | track 100 (ball None/None) | 18-50 | 33 boxes: 20 manual, 13 interpolated (22-30,39,45-46,49) | ball in play, positions read by eye from zoomed crops (about +-4 px). Not annotated: frames 0-17 (not visible, probably held by the goalkeeper), 22-30 (a high ball above the top of the frame), 39 and 45-46 and 49 (hidden at a player's feet or in the scrum) | manual+interp |
| 47 | drop boxes | track 100 | 22-30,39,45-46,49 | 13 boxes removed | ball not visible in these frames; no position invented across a gap |  |
