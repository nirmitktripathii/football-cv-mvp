"""Apply the a2 hand-corrections to prefill_cvat.xml -> corrected_cvat.xml + changelog.md.

Every edit is a call to one of the ops below and lands in the changelog with its frames and reason.
Boxes marked `det` come from the 4-class detector's raw detections (detections.parquet), `interp` are
linear interpolations, `manual` are read by eye from zoomed frames. Run from the repo root:

    python data/annotations/inter_napoli_fb4_a2_corner_t2707/apply_corrections.py
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_cv.evaluation import cvat

CLIP = "inter_napoli_fb4_a2_corner_t2707"
ROOT = Path(__file__).resolve().parents[3]
ANN = ROOT / "data" / "annotations" / CLIP
RUN = ROOT / "data" / "processed" / CLIP
NUM_FRAMES, W, H = 51, 1280, 720
MAX_GAP = 3  # sampled frames (5 fps): gaps up to 0.6 s inside one track are interpolated

tracks: dict[int, dict] = {}   # id -> {"label", "role", "team", "boxes": {frame: [x1,y1,x2,y2]}}
log: list[dict] = []
dets = pd.read_parquet(RUN / "detections.parquet")


def note(op: str, target: str, frames: str, detail: str, why: str, source: str = "") -> None:
    log.append({"n": len(log) + 1, "op": op, "target": target, "frames": frames, "detail": detail, "why": why, "source": source})


def span(fs) -> str:
    fs = sorted(fs)
    if not fs:
        return "-"
    runs, a = [], fs[0]
    for p, q in zip(fs, fs[1:] + [None]):
        if q is None or q != p + 1:
            runs.append(f"{a}" if a == p else f"{a}-{p}")
            a = q
    return ",".join(runs)


def load() -> None:
    for t in ET.parse(ANN / "prefill_cvat.xml").getroot().findall("track"):
        boxes, attrs = {}, {}
        for b in t.findall("box"):
            if b.get("outside") == "1":
                continue
            boxes[int(b.get("frame"))] = [float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            attrs = {a.get("name"): a.text for a in b.findall("attribute")}
        tracks[int(t.get("id"))] = {"label": t.get("label"), "role": attrs.get("role"), "team": attrs.get("team"), "boxes": boxes}


def delete(tid: int, why: str) -> None:
    gone = sorted(tracks[tid]["boxes"])
    del tracks[tid]
    note("delete", f"track {tid}", span(gone), f"{len(gone)} boxes removed", why)


def relabel(tid: int, role: str, team: str, why: str) -> None:
    tr = tracks[tid]
    note("relabel", f"track {tid}", span(tr["boxes"]), f"role/team {tr['role']}/{tr['team']} -> {role}/{team}", why)
    tr["role"], tr["team"] = role, team


def iou(a, b) -> float:
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def merge(into: int, others: list[int], why: str, role=None, team=None) -> None:
    """Fold `others` into track `into`. Where both have a box in a frame, `into` wins if the boxes overlap (IoU>0.3); otherwise stop."""
    tr = tracks[into]
    for o in others:
        added, dropped = [], []
        for f, b in sorted(tracks[o]["boxes"].items()):
            if f in tr["boxes"]:
                if iou(tr["boxes"][f], b) <= 0.3:
                    raise SystemExit(f"merge {o}->{into}: frame {f} holds two different people (IoU<=0.3); not a duplicate")
                dropped.append(f)
            else:
                tr["boxes"][f] = b
                added.append(f)
        del tracks[o]
        note("merge", f"track {o} -> {into}", span(added), f"{len(added)} boxes moved, {len(dropped)} duplicates dropped ({span(dropped)})", why)
    if role or team:
        tr["role"], tr["team"] = role or tr["role"], team or tr["team"]
        note("relabel", f"track {into}", span(tr["boxes"]), f"role/team -> {tr['role']}/{tr['team']}", why)


def interpolate(keys: dict[int, list[float]]) -> tuple[dict[int, list[float]], set[int]]:
    """Linear box interpolation between keyframes; returns all boxes and the set of interpolated frames."""
    ks = sorted(keys)
    out, interp = {}, set()
    for f in range(ks[0], ks[-1] + 1):
        if f in keys:
            out[f] = list(keys[f])
            continue
        lo = max(k for k in ks if k < f)
        hi = min(k for k in ks if k > f)
        a = (f - lo) / (hi - lo)
        out[f] = [(1 - a) * p + a * q for p, q in zip(keys[lo], keys[hi])]
        interp.add(f)
    return out, interp


def add_track(label: str, role: str | None, team: str | None, keys: dict[int, list[float]], source: str, why: str) -> int:
    tid = max(max(tracks, default=-1) + 1, 100)
    boxes, interp = interpolate(keys)
    tracks[tid] = {"label": label, "role": role, "team": team, "boxes": boxes}
    note("add track", f"track {tid} ({label} {role}/{team})", span(boxes),
         f"{len(boxes)} boxes: {len(keys)} {source}, {len(interp)} interpolated ({span(interp)})", why, f"{source}+interp" if interp else source)
    return tid


def fill_gaps(why: str) -> None:
    for tid, tr in sorted(tracks.items()):
        if tr["label"] != "person":
            continue
        fs = sorted(tr["boxes"])
        filled = []
        for a, b in zip(fs, fs[1:]):
            if 1 < b - a <= MAX_GAP + 1:
                inner, ip = interpolate({a: tr["boxes"][a], b: tr["boxes"][b]})
                for f in ip:
                    tr["boxes"][f] = inner[f]
                    filled.append(f)
        if filled:
            note("fill gap", f"track {tid}", span(filled), f"{len(filled)} boxes interpolated", why, "interp")


def add_frame0(why: str) -> None:
    """The tracker emits no boxes for sample 0. Match the raw sample-0 person detections to the sample-1 boxes."""
    d0 = dets[(dets.sample_id == 0) & (dets.object_class == "person") & (dets.bbox_y1 < 690)]
    D = d0[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    ids = [t for t, tr in tracks.items() if tr["label"] == "person" and 1 in tr["boxes"] and 0 not in tr["boxes"]]
    B = np.array([tracks[t]["boxes"][1] for t in ids])
    cen = lambda a: np.c_[(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2]
    shift = np.zeros(2)
    for _ in range(5):
        cost = np.linalg.norm(cen(D)[:, None] - (cen(B)[None] + shift), axis=2)
        r, c = linear_sum_assignment(cost)
        shift = np.median(cen(D)[r] - cen(B)[c], axis=0)
    cost = np.linalg.norm(cen(D)[:, None] - (cen(B)[None] + shift), axis=2)
    r, c = linear_sum_assignment(cost)
    used, matched = set(), []
    for i, j in zip(r, c):
        hr = (D[i, 3] - D[i, 1]) / (B[j, 3] - B[j, 1])
        if cost[i, j] < 28 and 0.7 < hr < 1.4:
            tracks[ids[j]]["boxes"][0] = D[i].tolist()
            used.add(i)
            matched.append(ids[j])
    note("add frame 0", f"{len(matched)} tracks", "0", f"matched raw sample-0 detections to sample-1 boxes (camera shift {shift.round(1).tolist()} px)", why, "det")
    print("frame 0: unmatched detections", [D[i].round(0).tolist() for i in range(len(D)) if i not in used])
    print("frame 0: tracks without a frame-0 box", [t for t in ids if t not in matched])


def best_det(frame: int, x: tuple[float, float], y: tuple[float, float], min_w: float = 0.0, skip_role: str = ""):
    """Highest-confidence raw person detection whose centre lies in x-range, y-range."""
    d = dets[(dets.sample_id == frame) & (dets.object_class == "person") & ((dets.bbox_x2 - dets.bbox_x1) > min_w) & (dets.role != skip_role)]
    cx, cy = (d.bbox_x1 + d.bbox_x2) / 2, (d.bbox_y1 + d.bbox_y2) / 2
    d = d[(cx > x[0]) & (cx < x[1]) & (cy > y[0]) & (cy < y[1])].sort_values("confidence", ascending=False)
    return None if d.empty else d.iloc[0][["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float).tolist()


# ------------------------------------------------------------------------------------------------ corrections
def main() -> None:
    load()

    # Ball: the pre-fill's only ball track sits still on an advertising board; the live ball is not in it.
    delete(46, "false positive: static 'ball' on the advertising board at about (786,130), not the ball in play")
    ball = {0: (585.8, 148.3), 1: (583, 136), 2: (579, 128), 3: (581, 127), 4: (582, 126), 5: (587, 125), 6: (588, 124), 7: (588, 123),
            8: (590, 117), 9: (592, 115), 10: (593, 115), 11: (593, 114), 12: (594, 114), 13: (595, 112), 14: (594, 113), 15: (595, 111),
            16: (596, 111), 17: (597, 112)}
    ball.update({f: (600, 111) for f in range(18, 51)})
    add_track("ball", None, None, {f: [x - 4.5, y - 4.5, x + 4.5, y + 4.5] for f, (x, y) in ball.items()}, "manual",
              "the live ball (dribbled, then placed on the corner arc); the spare ball on the board edge near (625,100) is not annotated")

    # The corner taker and the assistant referee are in the raw detections but the tracker never emitted them.
    off = {f: best_det(f, (604, 650), (70, 150), min_w=10) for f in range(51) if f != 28}
    off[28] = best_det(28, (615, 640), (70, 120))
    off = {f: b for f, b in off.items() if b is not None}
    taker = {f: best_det(f, (555, 600), (80, 150), min_w=10, skip_role="goalkeeper") for f in range(51)}
    taker = {f: b for f, b in taker.items() if b is not None}
    print("official keyframes", span(off), "| taker keyframes", span(taker))

    for tid, why in ((24, "corner-taker fragment; replaced by the complete corner-taker track"),
                     (32, "corner-taker fragment; replaced by the complete corner-taker track"),
                     (42, "assistant-referee fragment with no role; replaced by the complete official track")):
        delete(tid, why)
    add_track("person", "player", "away", taker, "det",
              "away player at the corner flag, absent from the pre-fill (the tracker's pitch mask drops the touchline)")
    add_track("person", "referee", "none", off, "det", "assistant referee in neon yellow at the corner flag, absent from the pre-fill")

    # Roles and non-persons
    relabel(0, "referee", "none", "neon-yellow match referee; pre-fill said player/unknown")
    delete(18, "bottom-left graphics sliver at (45,701), not a person")
    delete(22, "pitch-side camera dolly (white dome, no person)")
    delete(28, "pitch-side camera dolly (white dome, no person)")

    # Goalkeeper: one person split across tracks 10, 21, 36 (home goalkeeper in the dark kit, at the near post)
    merge(21, [10], "same person (home goalkeeper) before the tracker re-assigned the id", role="goalkeeper", team="home")
    merge(21, [36], "same person: track 21 ends at (296,279) f30 and 36 starts 5 px away at f33")
    gk = tracks[21]["boxes"]
    for f in (48, 49, 50):
        gk[f] = list(gk[47])
    note("extend", "track 21 (goalkeeper)", "48-50", "box held at its f47 position (goalkeeper is stationary at the post)", "track was lost 3 frames early", "manual")

    # Same person split across tracks or duplicated by the tracker
    merge(26, [9], "same away player: track 9 leaves view f6, track 26 starts 6 px away at f9")
    merge(27, [17], "same home player: 17 ends f8, 27 starts 17 px away at f9")
    merge(6, [40, 41], "same away player: 6 ends f34, 40 starts 15 px away at f38, 41 continues at f42")
    merge(33, [30, 34], "same home player at the near post: 30 hands over to 33 at f25, 34 is the same person under a second id")
    for tid, why in ((35, "duplicate box on track 12 (IoU up to 0.59)"), (38, "duplicate of track 3 (IoU 0.55)"),
                     (39, "duplicate of tracks 37/14 (IoU 0.55-0.70)"), (43, "duplicate of track 29 (IoU 0.67)"),
                     (44, "duplicate of track 3 (IoU 0.56)"), (45, "duplicate of track 37 (IoU 0.70)")):
        delete(tid, why)

    # Teams of the remaining unknowns, by kit colour (red kit = away, dark/purple kit = home)
    merge(23, [25], "same red-kit runner: 23 ends f4, 25 starts 22 px away at f7", role="player", team="away")

    # Away runner in white shorts: the pre-fill's id 19 starts at f3, but the same person is boxed under id 2 at f1 (an id switch)
    # and unboxed at f0 and f2.
    tracks[19]["boxes"][1] = tracks[2]["boxes"].pop(1)
    note("move box", "track 2 -> 19", "1", "box at (501,332,521,359) moved to the runner's track", "id switch: the box is on the away runner (id 19 from f3), not on id 2", "manual")
    r0, r2 = best_det(0, (515, 530), (330, 370)), best_det(2, (490, 505), (320, 350))
    tracks[19]["boxes"][0], tracks[19]["boxes"][2] = r0, r2
    note("add boxes", "track 19", "0,2", f"raw detections {np.round(r0).tolist()} and {np.round(r2).tolist()} added", "same runner, missed by the tracker at f0 and f2", "det")

    # Home player walking left behind the red cluster, f0-f12: boxed once (id 20, f3) and otherwise missing.
    walker = {0: (453, 276, 466, 302), 1: (441, 263, 454, 295), 2: (432, 257, 444, 294), 3: tuple(tracks[20]["boxes"][3]), 4: (419, 253, 432, 294),
              5: (417, 268, 429, 300), 8: (408, 248, 422, 292), 10: (386, 243, 402, 276), 11: (382, 242, 396, 267), 12: (377, 242, 389, 273)}
    delete(20, "single-frame fragment; folded into the new walker track")
    add_track("person", "player", "home", {f: list(b) for f, b in walker.items()}, "det",
              "dark-kit (home) player walking left at the edge of the cluster, missed by the tracker f0-f12; f3 box comes from the old id 20")

    # Two players entering at the top of the penalty area, f47-f50 (no boxes in the pre-fill)
    add_track("person", "player", "away", {47: [404, 210, 416, 242], 48: [408, 205, 425, 240], 49: [399, 195, 414, 232], 50: [408, 188, 425, 225]}, "det",
              "red-kit player entering at the top of the area, missed by the tracker")
    add_track("person", "player", "home", {49: [418, 204, 430, 236], 50: [426, 195, 440, 233]}, "det",
              "dark-kit player entering at the top of the area, missed by the tracker")
    note("known gap", "frames 0-2", "0-2", "one away player hidden behind id 13's box (two overlapping red players in the pre-fill's single box) is not annotated", "boxes too uncertain to read by eye", "")

    fill_gaps(f"gap of <= {MAX_GAP} sampled frames inside one track; the person stays in view")
    add_frame0("the tracker emits nothing for sample 0, so frame 0 had no person boxes at all")
    write()


def write() -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "name").text = CLIP
    ET.SubElement(task, "size").text = str(NUM_FRAMES)
    ET.SubElement(task, "mode").text = "annotation"
    size = ET.SubElement(task, "original_size")
    ET.SubElement(size, "width").text = str(W)
    ET.SubElement(size, "height").text = str(H)
    order = sorted(tracks.items(), key=lambda kv: (kv[1]["label"] != "person", min(kv[1]["boxes"]), kv[0]))
    for new_id, (_, tr) in enumerate(order):
        rows = pd.DataFrame([{"sample_id": f, "bbox_x1": max(0, b[0]), "bbox_y1": max(0, b[1]), "bbox_x2": min(W, b[2]), "bbox_y2": min(H, b[3])}
                             for f, b in sorted(tr["boxes"].items())])
        cvat._add_track(root, new_id, tr["label"], rows, NUM_FRAMES, {} if tr["label"] == "ball" else {"role": tr["role"], "team": tr["team"]})
    ET.indent(root)
    (ANN / "corrected_cvat.xml").write_text(ET.tostring(root, encoding="unicode", xml_declaration=True), encoding="utf-8")
    lines = [f"# Changelog: {CLIP}", "",
             "Pre-fill: `prefill_cvat.xml` (4-class detector + ByteTrack). Output: `corrected_cvat.xml`. Source: det = raw detector box, "
             "interp = linear interpolation, manual = read by eye. Track ids below are the pre-fill's; the output renumbers them.", "",
             "| # | op | target | frames | detail | reason | source |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {e['n']} | {e['op']} | {e['target']} | {e['frames']} | {e['detail']} | {e['why']} | {e['source']} |" for e in log]
    (ANN / "changelog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    people = [t for t in tracks.values() if t["label"] == "person"]
    print(f"{len(people)} person tracks, {sum(len(t['boxes']) for t in people)} boxes; "
          f"ball boxes {sum(len(t['boxes']) for t in tracks.values() if t['label'] == 'ball')}; {len(log)} changelog entries")


if __name__ == "__main__":
    main()
