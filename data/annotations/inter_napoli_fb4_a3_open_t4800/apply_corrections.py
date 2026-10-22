"""Hand corrections for window a3 (inter_napoli_fb4_a3_open_t4800): relink, review by eye, write corrected_cvat.xml + changelog.md.

    python data/annotations/inter_napoli_fb4_a3_open_t4800/apply_corrections.py

Every manual edit is logged in changelog.md. Tracks are addressed by an anchor (frame, x, y) on the relinked
output, not by id, so the script survives a change in relink parameters (it stops if an anchor hits nothing).
Colours: home = dark/purple-blue kit (Inter); away = red kit (Napoli); referees are neon yellow.
Second half: the away (yellow) goalkeeper defends the left goal, the home (dark-grey) goalkeeper the right one.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import annotation_tools as T  # noqa: E402

win = T.Window("inter_napoli_fb4_a3_open_t4800", ROOT)
c = T.Corrections(win)
c.load_relinked(T.relink(T.candidates(win)))

# anchors: (frame, x, y) of the box centre in the relinked output
A = {
    "referee": (0, 480, 273), "referee_b": (44, 428, 293), "referee_frag": (46, 402, 290),
    "keeper_away": (0, 9, 293), "keeper_home": (24, 1262, 313), "keeper_home_b": (46, 1063, 225),
    "ar_top": (0, 434, 151), "ar_top_b": (40, 168, 155), "ar_bottom": (12, 1020, 649), "fourth": (13, 631, 648),
    "static_dark": (25, 228, 178), "staff": (36, 766, 652), "empty": (36, 808, 548),
    "dup_red_a": (0, 933, 393), "dup_red_b": (16, 993, 413), "red_20": (0, 937, 398),
    "dark_a": (4, 772, 221), "dark_a_b": (0, 760, 215),
    "dark_b": (0, 577, 220), "dark_b_b": (4, 613, 227), "dark_b_c": (12, 689, 232),
    "red_c": (0, 631, 278), "red_c_b": (21, 747, 305), "red_c_c": (50, 489, 241),
    "dark_d": (0, 334, 287), "dark_d_b": (39, 108, 330),
    "dark_hidden": (24, 902, 401), "away_4": (50, 131, 411), "home_5": (50, 155, 416),
}
I = {k: c.at(*v) for k, v in A.items()}

frag46 = list(c.tracks[I["referee_frag"]]["boxes"][46])  # the upper half of the referee, crouched in the scrum at frame 46

# -- not people on the pitch / outside scope
c.delete(I["static_dark"], "static dark figure at the top touchline (stand or staff), not a player or official")
c.delete(I["staff"], "member of staff in a suit on the bottom touchline, outside the evaluation scope")
c.delete(I["empty"], "single detection on empty grass")
c.delete(I["dup_red_a"], "duplicate box on the red player of track 20 (frames 0 and 3)")
c.delete(I["dup_red_b"], "duplicate box on the red player of track 20 (frame 16)")
c.delete(I["referee_frag"], "partial box on the referee inside the scrum; the referee's own track has his full box")
c.to_ignore(I["fourth"], "neon-yellow official walking behind the touchline (probably the fourth official), not one of the three on-pitch officials; ignored rather than guessed")


def absorb(into, others, why):
    """merge, first dropping a fragment's box in frames where it overlaps the kept track by IoU <= 0.3 (a partial box of the same person)."""
    for o in others:
        shared = [f for f, b in c.tracks[o]["boxes"].items() if f in c.tracks[into]["boxes"] and T.iou(b, c.tracks[into]["boxes"][f]) <= 0.3]
        if shared:
            c.drop_frames(o, shared, "partial box on the same person; the kept track's box is used")
        c.merge(into, [o], why)


# -- referee: his own track to frame 43, then boxes that slid onto the red player in the scrum; the second track is on him
c.relabel(I["referee"], "referee", "none", "neon-yellow kit, the match referee; the detector called him a player")
c.drop_frames(I["referee"], [f for f in c.tracks[I["referee"]]["boxes"] if f >= 44], "from frame 44 these boxes sit on the red player and the dark player in the scrum, not on the referee")
absorb(I["referee"], [I["referee_b"]], "same referee: the tracker restarted him at frame 39 and again at 44")
lower46 = c.tracks[I["referee"]]["boxes"][46]
c.set_boxes(I["referee"], {46: [min(frag46[0], lower46[0]), min(frag46[1], lower46[1]), max(frag46[2], lower46[2]), max(frag46[3], lower46[3])]},
            "frame 46: the referee is bent over beside the scrum and the detector split him into an upper and a lower box; replaced by their union")

# -- goalkeepers by kit (dark-grey = home, yellow = away); the second half has the sides swapped
c.relabel(I["keeper_away"], "goalkeeper", "away", "yellow goalkeeper at the left goal: the away keeper (the home keeper is dark grey); the detector called him a player")
c.relabel(I["keeper_home"], "goalkeeper", "home", "dark-grey goalkeeper at the right goal: the home keeper; the detector called him a player")
absorb(I["keeper_home"], [I["keeper_home_b"]], "same goalkeeper: a box at frame 46 where the track had a gap")

# -- assistant referees
c.relabel(I["ar_top"], "referee", "none", "neon-yellow assistant referee on the top touchline")
absorb(I["ar_top"], [I["ar_top_b"]], "same assistant referee: the tracker lost him at frames 34-39 (touchline mask), then restarted him")
keys = {f: c.tracks[I["ar_top"]]["boxes"][f] for f in (33, 40)}
c.set_boxes(I["ar_top"], {f: b for f, b in c.interpolate(keys)[0].items() if 33 < f < 40},
            "frames 34-39: assistant referee visible near the touchline but not detected; straight line between his frame-33 and frame-40 boxes", "interp")
c.relabel(I["ar_bottom"], "referee", "none", "neon-yellow assistant referee with the flag on the bottom touchline")
# his first 12 frames were dropped by the pitch mask (he stands on the touchline): three boxes read by eye, the rest interpolated
by_eye = {0: [977, 640, 998, 685], 4: [981, 628, 1000, 673], 8: [990, 626, 1012, 668], 12: c.tracks[I["ar_bottom"]]["boxes"][12]}
c.set_boxes(I["ar_bottom"], {f: b for f, b in c.interpolate(by_eye)[0].items() if f < 12},
            "frames 0-11: assistant referee on the touchline, undetected; boxes at frames 0, 4 and 8 read by eye from zoomed crops (about +-3 px), the others interpolated "
            "between them and his frame-12 detection")

# -- fragments of one player
absorb(I["dark_a"], [I["dark_a_b"]], "same home player, split at frame 4")
absorb(I["dark_b"], [I["dark_b_b"], I["dark_b_c"]], "same home player, split in three")
absorb(I["red_c"], [I["red_c_b"], I["red_c_c"]], "same red player: a box at frame 21 where the track had a gap, and a duplicate restart at frame 39")
absorb(I["dark_d"], [I["dark_d_b"]], "same home player at the left edge, tracker restarted him at frames 39-42")

c.fill_gaps("short gap inside one person's track")


def behind(hidden, front, frames, lo, hi, why):
    """Box of a player hidden behind another: the front player's box plus the offset between the two, interpolated from frames lo and hi."""
    fb, hb = c.tracks[front]["boxes"], c.tracks[hidden]["boxes"]
    off = {f: [h - r for h, r in zip(hb[f], fb[f])] for f in (lo, hi)}
    new = {}
    for f in frames:
        a = (f - lo) / (hi - lo)
        new[f] = [r + (1 - a) * p + a * q for r, p, q in zip(fb[f], off[lo], off[hi])]
    c.set_boxes(hidden, new, why, "interp")


behind(I["dark_hidden"], I["red_20"], [21, 22, 23], 20, 24,
       "frames 21-23: the home player is hidden behind the red player of the tackle (only his head shows); the gap fill had put the box on empty grass. "
       "Box = the red player's box plus the offset between the two, interpolated from frames 20 and 24")
behind(I["dark_hidden"], I["red_20"], [27], 26, 28,
       "frame 27: hidden behind the red player; the gap fill had duplicated the red player's box. Box = the red player's box plus the interpolated offset")

# the two players grappling at frames 35-36 are almost fully overlapped; the tracker's boxes were swapped onto the wrong player
c.set_boxes(I["away_4"], {35: [139, 406, 152, 445], 36: [128, 405, 140, 446]},
            "frames 35-36: red-kit player grappling with the dark-kit player, almost fully overlapped; the existing boxes had swapped sides. Read by eye (about +-3 px), flagged for audit")
c.set_boxes(I["home_5"], {35: [141, 410, 154, 447], 36: [132, 411, 146, 449]},
            "frames 35-36: dark-kit player grappling with the red-kit player, almost fully overlapped; the existing boxes had swapped sides. Read by eye (about +-3 px), flagged for audit")

# -- ball: the real ball, read by eye from zoomed crops. The detector's "ball" track is mostly the penalty spot (157-174,308), a spare
# ball behind the right goal drifting with the camera, and a white dash at (693,684); none of them is the ball in play.
box = lambda x, y: [x - 4, y - 4, x + 4, y + 4]  # noqa: E731
ball_xy = {18: (187, 263), 19: (249, 235), 20: (308, 208), 21: (348, 186), 31: (465, 189), 32: (490, 214), 33: (502, 244),
           34: (508, 278), 35: (525, 316), 36: (531, 327), 37: (516, 308), 38: (507, 298), 40: (491, 300), 41: (483, 311),
           42: (473, 319), 43: (464, 312), 44: (457, 304), 47: (369, 289), 48: (327, 277), 50: (253, 260)}
ball = c.add_track("ball", None, None, {f: box(*p) for f, p in ball_xy.items()}, "manual",
                   "ball in play, positions read by eye from zoomed crops (about +-4 px). Not annotated: frames 0-17 (not visible, probably held by "
                   "the goalkeeper), 22-30 (a high ball above the top of the frame), 39 and 45-46 and 49 (hidden at a player's feet or in the scrum)")
c.drop_frames(ball, list(range(22, 31)) + [39, 45, 46, 49], "ball not visible in these frames; no position invented across a gap")
c.write()
