"""Hand corrections for window a1 (inter_napoli_fb4_a1_open_t600): relink, review by eye, write corrected_cvat.xml + changelog.md.

    python data/annotations/inter_napoli_fb4_a1_open_t600/apply_corrections.py

Every manual edit is logged in changelog.md. Tracks are addressed by an anchor (frame, x, y) on the relinked
output, not by id, so the script survives a change in relink parameters (it stops if an anchor hits nothing).
Colours: home = dark/purple-blue kit (Inter); away = red kit (Napoli); referee and assistant referee are neon yellow.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import annotation_tools as T  # noqa: E402

win = T.Window("inter_napoli_fb4_a1_open_t600", ROOT)
c = T.Corrections(win)
c.load_relinked(T.relink(T.candidates(win)))

# anchors: (frame, x, y) of the box centre in the relinked output
A = {
    "cameraman": (8, 1262, 598), "touchline_staff": (36, 1007, 685), "spectator": (25, 917, 136),
    "dolly1": (35, 637, 111), "dolly2": (36, 665, 114), "dolly3": (37, 696, 118), "dolly4": (38, 725, 119), "dolly5": (39, 754, 121),
    "pitch_mark": (25, 800, 269), "partial_a": (25, 1033, 280), "partial_b": (24, 1030, 305), "fragment_c": (43, 684, 243),
    "white_edge": (0, 15, 195), "referee": (0, 594, 260), "keeper": (0, 1033, 241),
    "dark17_a": (0, 1054, 535), "dark17_b": (1, 1058, 523),
    "red_a": (0, 894, 488), "red_b": (6, 946, 490), "red_c": (0, 1070, 487), "red_d": (8, 1010, 474),
    "red_e": (0, 837, 363), "red_f": (6, 874, 354), "red_g": (24, 1035, 307),
    "dark_a": (0, 834, 287), "dark_b": (17, 883, 275), "dark_c": (25, 1027, 289),
    "red_h": (0, 871, 284), "red_i": (18, 905, 274),
    "dark_g": (43, 1068, 311), "red_j": (41, 1070, 308),
    "dark_d": (0, 801, 238), "dark_e": (29, 925, 262), "dark_f": (44, 911, 248),
}
I = {k: c.at(*v) for k, v in A.items()}

# -- not people on the pitch / outside scope
c.delete(I["cameraman"], "cameraman at the touchline, not a player or official")
c.delete(I["touchline_staff"], "staff member at the bottom touchline, outside the evaluation scope")
c.delete(I["spectator"], "spectator or photographer in the stand")
for k in ("dolly1", "dolly2", "dolly3", "dolly4", "dolly5"):
    c.delete(I[k], "camera dolly or debris on the touchline near the corner flag, detected as a person")
c.delete(I["pitch_mark"], "static artefact on the pitch (a white dash fixed in the image), not a person")
c.to_ignore(I["white_edge"], "person at the very edge of the frame, partly out of view; role cannot be told, so ignore rather than guess")

# -- roles
c.relabel(I["referee"], "referee", "none", "neon-yellow kit, matches the match referee; the detector called him a player")
c.relabel(I["keeper"], "goalkeeper", "away", "yellow goalkeeper: the red (away) players defend his goal and clear the ball from it at frames 41-50, so he is the away keeper "
          "(the home keeper is the dark-grey one seen in window a2); the detector called him a player")

# -- fragments of one person
def absorb(into, others, why):
    """merge, first dropping a fragment's box in frames where it overlaps the kept track by IoU <= 0.3 (a partial box of the same person)."""
    for o in others:
        shared = [f for f, b in c.tracks[o]["boxes"].items() if f in c.tracks[into]["boxes"] and T.iou(b, c.tracks[into]["boxes"][f]) <= 0.3]
        if shared:
            c.drop_frames(o, shared, "partial box on the same person; the kept track's box is used")
        c.merge(into, [o], why)


c.drop_frames(I["dark17_a"], [1], "frame 1 box covers only the legs; the other track's box on the same player is the full body")
c.merge(I["dark17_b"], [I["dark17_a"]], "number 17, dark kit, one track split at the start")
absorb(I["red_a"], [I["red_b"]], "same red player, duplicate box at frames 6-8")
absorb(I["red_c"], [I["red_d"]], "same red player, tracker lost him at frame 10")
absorb(I["red_e"], [I["red_f"], I["red_g"]], "same red player running to the goal mouth, split in three")
absorb(I["dark_a"], [I["dark_b"], I["dark_c"]], "same dark player: running forward then jumping in the goal area, split in three")
absorb(I["red_h"], [I["red_i"]], "same red player, split at frame 18")
# the scrum at frames 41-47: the red box on the dark crouching player (41, 45) and the boxes interpolated or duplicated on the
# dark player (44-47) were wrong; the red player with the ball (from 41) and the dark jumper (from 43) continue as new tracks
dark_at_41 = list(c.tracks[I["red_e"]]["boxes"][41])
c.drop_frames(I["red_e"], [41, 45], "box sits on a dark player in the scrum, not on this red player")
c.set_boxes(I["dark_a"], {41: dark_at_41}, "frame 41: the crouching dark player had no box (the existing box duplicated the dark player to his right); "
            "the box dropped from the red track fits him")
absorb(I["red_e"], [I["red_j"]], "same red player, now carrying the ball: tracker restarted him at frame 41")
c.drop_frames(I["dark_a"], [47], "duplicate of the neighbouring dark player's box")
absorb(I["dark_a"], [I["dark_g"]], "same dark player: went to ground behind the red player at frame 43, tracker restarted him")
absorb(I["dark_d"], [I["dark_e"], I["dark_f"]], "same dark player on the edge of the box, split in three")
c.delete(I["partial_a"], "partial duplicate of a player inside the scrum")
c.delete(I["partial_b"], "partial duplicate of a player inside the scrum")
c.delete(I["fragment_c"], "single-frame fragment")

red_behind = c.at(31, 1035, 283)

c.fill_gaps("short gap inside one person's track")

# frame 32: the red player behind the dark one is hidden, and the gap fill gave him the dark player's box (IoU 1.0 with it);
# use the straight line between his frame-31 and frame-33 boxes instead
b31, b33 = c.tracks[red_behind]["boxes"][31], c.tracks[red_behind]["boxes"][33]
c.set_boxes(red_behind, {32: [(p + q) / 2 for p, q in zip(b31, b33)]},
            "frame 32: gap-fill box duplicated the dark player's; replaced by the midpoint of this player's neighbouring boxes (hidden behind him)")

# -- ball: the real ball, read by eye. The detector's ball track is a static white dash at (812,269) and the
# spare balls behind the goal at (1077,255); neither is the ball in play.
box = lambda x, y: [x - 4, y - 4, x + 4, y + 4]  # noqa: E731
ball_xy = {1: (1049, 536), 2: (1054, 534), 3: (1051, 524), 4: (1050, 517), 5: (1036, 515), 6: (1017, 511), 7: (995, 508),
           8: (947, 491), 9: (961, 424), 10: (979, 368), 11: (993, 325), 12: (1007, 292), 13: (1019, 271), 14: (1029, 260),
           16: (1049, 268), 17: (1057, 285), 18: (1067, 308), 19: (1076, 338), 20: (1079, 340), 21: (1070, 308),
           22: (1061, 285), 23: (1052, 274), 24: (1044, 270), 25: (1034, 273), 29: (1027, 287), 31: (1036, 298),
           33: (1052, 292), 34: (1063, 300), 35: (1070, 296), 36: (1080, 297), 37: (1085, 304), 38: (1089, 308),
           39: (1090, 312), 40: (1078, 317), 41: (1069, 320), 42: (1062, 324), 43: (1058, 332), 44: (1054, 340),
           45: (1050, 347), 46: (1045, 353), 47: (1000, 363), 48: (933, 380), 50: (819, 404)}
ball = c.add_track("ball", None, None, {f: box(*p) for f, p in ball_xy.items()}, "manual",
                   "ball in play, positions read by eye from zoomed crops (about +-4 px); frame 0 hidden at a player's feet, "
                   "frames 26-28 hidden in the scrum, so left unannotated")
c.drop_frames(ball, [26, 27, 28], "ball not visible in the scrum; no position invented across a gap")

c.write()
