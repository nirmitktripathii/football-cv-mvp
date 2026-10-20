"""Helpers for hand-correcting an evaluation window (docs/annotation_guide.md), used by each window's
``data/annotations/<clip>/apply_corrections.py``.

``relink`` rebuilds person tracks offline from the raw detections (``detections.parquet``): it links
boxes across frames in camera-stabilised coordinates, using kit colour as a hard gate, and so fixes the
online tracker's fragmentation and its empty sample 0. Its output is a *starting point*: every track is
still checked by eye, and every hand edit is logged in that window's ``apply_corrections.py``.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_cv.calibration.motion import G_of, apply_h
from football_cv.evaluation import cvat

W, H, NUM_FRAMES = 1280, 720, 51
BOX = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


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


def iou(a, b) -> float:
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def kit_colour(img: np.ndarray, box) -> tuple[str, dict[str, float]]:
    """Kit class of the torso: away (red), home (dark), yellow (neon referee / yellow keeper) or unk."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = y1 + int(0.18 * h), y1 + int(0.55 * h)
    tx1, tx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    crop = img[max(ty1, 0):max(ty2, 1), max(tx1, 0):max(tx2, 1)]
    if crop.size == 0:
        return "unk", {}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    frac = {
        "away": float((((hue < 9) | (hue > 170)) & (sat > 110) & (val > 80)).mean()),
        "yellow": float(((hue > 18) & (hue < 45) & (sat > 100) & (val > 140)).mean()),
        "home": float(((val < 85) | ((hue > 100) & (hue < 165) & (sat > 45) & (val < 170))).mean()),
    }
    if frac["away"] > 0.25:
        return "away", frac
    if frac["yellow"] > 0.25:
        return "yellow", frac
    if frac["home"] > 0.3:
        return "home", frac
    return "unk", frac


@dataclass
class Window:
    clip: str
    root: Path
    dets: pd.DataFrame = field(init=False)
    motion: pd.DataFrame = field(init=False)
    frames: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.run = self.root / "data" / "processed" / self.clip
        self.ann = self.root / "data" / "annotations" / self.clip
        self.dets = pd.read_parquet(self.run / "detections.parquet")
        self.motion = pd.read_parquet(self.run / "motion.parquet").set_index("sample_id")
        self.frames = pd.read_parquet(self.run / "frames.parquet").set_index("sample_id")
        self._imgs: dict[int, np.ndarray] = {}

    def image(self, f: int) -> np.ndarray:
        if f not in self._imgs:
            self._imgs[f] = cv2.imread(str(self.run / self.frames.loc[f, "path"]))
        return self._imgs[f]

    def G(self, f: int) -> np.ndarray:
        return G_of(self.motion.loc[f])


def nms(d: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Per frame, drop a box overlapping a more confident one by IoU > thr (the detector's duplicate boxes in crowds)."""
    keep = []
    for _, g in d.sort_values("confidence", ascending=False).groupby("sample_id"):
        kept: list[list[float]] = []
        for idx, r in zip(g.index, g[BOX].to_numpy(float)):
            if all(iou(r, k) <= thr for k in kept):
                kept.append(list(r))
                keep.append(idx)
    return d.loc[keep]


def candidates(win: Window, min_conf: float = 0.5) -> pd.DataFrame:
    """Raw person detections worth linking, with kit colour and stabilised foot position."""
    d = win.dets[(win.dets.object_class == "person") & (win.dets.confidence >= min_conf)].copy()
    d = nms(d[(d.bbox_y1 < 690) & ((d.bbox_x2 - d.bbox_x1) > 6)])
    kit, sx, sy = [], [], []
    for r in d.itertuples():
        kit.append(kit_colour(win.image(r.sample_id), [r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2])[0])
    d["kit"] = kit
    for f, g in d.groupby("sample_id"):
        feet = np.c_[(g.bbox_x1 + g.bbox_x2) / 2, g.bbox_y2]
        s = apply_h(win.G(f), feet)
        sx.append(pd.Series(s[:, 0], index=g.index))
        sy.append(pd.Series(s[:, 1], index=g.index))
    d["sx"], d["sy"] = pd.concat(sx), pd.concat(sy)
    return d.sort_values(["sample_id", "confidence"], ascending=[True, False]).reset_index(drop=True)


def _compatible(a: str, b: str) -> bool:
    return a == b or "unk" in (a, b)


def relink(d: pd.DataFrame, max_gap: int = 4, gate: float = 0.9) -> pd.DataFrame:
    """Greedy-by-frame Hungarian linking; adds a ``tid`` column.

    Distance is the stabilised foot displacement in box heights, gated at ``gate`` per sampled frame of gap
    (plus 0.25 per extra frame). Teams may not mix: away/home/yellow are mutually exclusive, ``unk`` fits any.
    """
    d = d.copy()
    d["tid"] = -1
    live: dict[int, dict] = {}   # tid -> last row info
    nxt = 0
    for f in sorted(d.sample_id.unique()):
        rows = d[d.sample_id == f]
        ids = [t for t, s in live.items() if 0 < f - s["f"] <= max_gap + 1]
        if ids and len(rows):
            cost = np.full((len(rows), len(ids)), 1e6)
            for j, t in enumerate(ids):
                s = live[t]
                for i, r in enumerate(rows.itertuples()):
                    h = max(s["h"], r.bbox_y2 - r.bbox_y1)
                    dist = np.hypot(r.sx - s["sx"], r.sy - s["sy"]) / h
                    lim = gate + 0.25 * (f - s["f"] - 1)
                    if dist < lim and _compatible(s["kit"], r.kit):
                        cost[i, j] = dist
            ri, cj = linear_sum_assignment(cost)
            assigned = {}
            for i, j in zip(ri, cj):
                if cost[i, j] < 1e5:
                    assigned[rows.index[i]] = ids[j]
        else:
            assigned = {}
        for idx, r in zip(rows.index, rows.itertuples()):
            t = assigned.get(idx)
            if t is None:
                t, nxt = nxt, nxt + 1
            d.at[idx, "tid"] = t
            kit = r.kit if r.kit != "unk" else live.get(t, {}).get("kit", "unk")
            live[t] = {"f": f, "sx": r.sx, "sy": r.sy, "h": r.bbox_y2 - r.bbox_y1, "kit": kit}
    return d


def to_xml(tracks: dict[int, dict], clip: str, path: Path) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "name").text = clip
    ET.SubElement(task, "size").text = str(NUM_FRAMES)
    ET.SubElement(task, "mode").text = "annotation"
    size = ET.SubElement(task, "original_size")
    ET.SubElement(size, "width").text = str(W)
    ET.SubElement(size, "height").text = str(H)
    order = sorted(tracks.items(), key=lambda kv: (kv[1]["label"] != "person", min(kv[1]["boxes"]), kv[0]))
    for new_id, (_, tr) in enumerate(order):
        rows = pd.DataFrame([{"sample_id": f, "bbox_x1": max(0, b[0]), "bbox_y1": max(0, b[1]), "bbox_x2": min(W, b[2]), "bbox_y2": min(H, b[3])}
                             for f, b in sorted(tr["boxes"].items())])
        cvat._add_track(root, new_id, tr["label"], rows, NUM_FRAMES, {"role": tr["role"], "team": tr["team"]} if tr["label"] == "person" else {})
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True), encoding="utf-8")


class Corrections:
    """Track set under hand-correction, with a changelog of every edit (see data/annotations/<clip>/changelog.md).

    Sources in the log: ``det`` = raw detector box, ``interp`` = linear interpolation, ``manual`` = read by eye.
    """

    MAX_GAP = 3  # sampled frames (5 fps): gaps up to 0.6 s inside one track are interpolated

    def __init__(self, win: Window) -> None:
        self.win = win
        self.tracks: dict[int, dict] = {}
        self.log: list[dict] = []

    # -- bookkeeping
    def note(self, op: str, target: str, frames: str, detail: str, why: str, source: str = "") -> None:
        self.log.append({"n": len(self.log) + 1, "op": op, "target": target, "frames": frames, "detail": detail, "why": why, "source": source})

    def at(self, frame: int, x: float, y: float, tol: float = 12) -> int:
        """Id of the track whose box at ``frame`` contains (x, y) (nearest centre if several); stops if none."""
        hits = []
        for tid, tr in self.tracks.items():
            b = tr["boxes"].get(frame)
            if b and b[0] - tol <= x <= b[2] + tol and b[1] - tol <= y <= b[3] + tol:
                hits.append((np.hypot((b[0] + b[2]) / 2 - x, (b[1] + b[3]) / 2 - y), tid))
        if not hits:
            raise SystemExit(f"no track at frame {frame} near ({x},{y})")
        return min(hits)[1]

    # -- building
    def load_relinked(self, d: pd.DataFrame) -> None:
        """Seed from ``relink`` output: team from kit colour, role "player" (roles are set by hand below)."""
        for tid, g in d.groupby("tid"):
            known = g[g.kit != "unk"]
            kit = known.kit.mode().iloc[0] if len(known) else "unk"
            self.tracks[int(tid)] = {"label": "person", "role": "player", "team": {"away": "away", "home": "home"}.get(kit, "unknown"),
                                     "boxes": {int(r.sample_id): [float(r.bbox_x1), float(r.bbox_y1), float(r.bbox_x2), float(r.bbox_y2)] for r in g.itertuples()}}
        self.note("seed", f"{len(self.tracks)} tracks", "0-50", "offline relink of raw detections (conf >= 0.5, NMS 0.5, kit-colour gate)",
                  "the online tracker emits nothing for sample 0 and fragments tracks", "det")

    def delete(self, tid: int, why: str) -> None:
        gone = sorted(self.tracks[tid]["boxes"])
        del self.tracks[tid]
        self.note("delete", f"track {tid}", span(gone), f"{len(gone)} boxes removed", why)

    def to_ignore(self, tid: int, why: str) -> None:
        tr = self.tracks[tid]
        tr["label"], tr["role"], tr["team"] = "ignore", None, None
        self.note("relabel", f"track {tid}", span(tr["boxes"]), "person -> ignore", why)

    def relabel(self, tid: int, role: str, team: str, why: str) -> None:
        tr = self.tracks[tid]
        self.note("relabel", f"track {tid}", span(tr["boxes"]), f"role/team {tr['role']}/{tr['team']} -> {role}/{team}", why)
        tr["role"], tr["team"] = role, team

    def merge(self, into: int, others: list[int], why: str, role: str | None = None, team: str | None = None) -> None:
        """Fold ``others`` into ``into``. Where both have a box in a frame, ``into`` wins if the boxes overlap (IoU > 0.3), else stop."""
        tr = self.tracks[into]
        for o in others:
            added, dropped = [], []
            for f, b in sorted(self.tracks[o]["boxes"].items()):
                if f in tr["boxes"]:
                    if iou(tr["boxes"][f], b) <= 0.3:
                        raise SystemExit(f"merge {o}->{into}: frame {f} holds two different people (IoU<=0.3); not a duplicate")
                    dropped.append(f)
                else:
                    tr["boxes"][f] = b
                    added.append(f)
            del self.tracks[o]
            self.note("merge", f"track {o} -> {into}", span(added), f"{len(added)} boxes moved, {len(dropped)} duplicates dropped ({span(dropped)})", why)
        if role or team:
            tr["role"], tr["team"] = role or tr["role"], team or tr["team"]
            self.note("relabel", f"track {into}", span(tr["boxes"]), f"role/team -> {tr['role']}/{tr['team']}", why)

    def set_boxes(self, tid: int, boxes: dict[int, list[float]], why: str, source: str = "manual") -> None:
        self.tracks[tid]["boxes"].update({f: list(b) for f, b in boxes.items()})
        self.note("set box", f"track {tid}", span(boxes), f"{len(boxes)} boxes replaced or added", why, source)

    def drop_frames(self, tid: int, frames: list[int], why: str) -> None:
        for f in frames:
            self.tracks[tid]["boxes"].pop(f)
        self.note("drop boxes", f"track {tid}", span(frames), f"{len(frames)} boxes removed", why)

    @staticmethod
    def interpolate(keys: dict[int, list[float]]) -> tuple[dict[int, list[float]], set[int]]:
        ks = sorted(keys)
        out, interp = {}, set()
        for f in range(ks[0], ks[-1] + 1):
            if f in keys:
                out[f] = list(keys[f])
                continue
            lo, hi = max(k for k in ks if k < f), min(k for k in ks if k > f)
            a = (f - lo) / (hi - lo)
            out[f] = [(1 - a) * p + a * q for p, q in zip(keys[lo], keys[hi])]
            interp.add(f)
        return out, interp

    def add_track(self, label: str, role: str | None, team: str | None, keys: dict[int, list[float]], source: str, why: str) -> int:
        tid = max(max(self.tracks, default=-1) + 1, 100)
        boxes, interp = self.interpolate(keys)
        self.tracks[tid] = {"label": label, "role": role, "team": team, "boxes": boxes}
        self.note("add track", f"track {tid} ({label} {role}/{team})", span(boxes),
                  f"{len(boxes)} boxes: {len(keys)} {source}, {len(interp)} interpolated ({span(interp)})", why, f"{source}+interp" if interp else source)
        return tid

    def fill_gaps(self, why: str) -> None:
        """Gaps of <= MAX_GAP frames inside a person track: use a raw detection (confidence >= 0.25) that overlaps the
        interpolated box (IoU > 0.4) and is not already in a track, else the interpolated box."""
        used = {(f, tuple(np.round(b, 1))) for tr in self.tracks.values() for f, b in tr["boxes"].items()}
        low = self.win.dets[(self.win.dets.object_class == "person") & (self.win.dets.confidence >= 0.25)]
        for tid, tr in sorted(self.tracks.items()):
            if tr["label"] != "person":
                continue
            fs = sorted(tr["boxes"])
            from_det, from_int = [], []
            for a, b in zip(fs, fs[1:]):
                if not 1 < b - a <= self.MAX_GAP + 1:
                    continue
                inner, ip = self.interpolate({a: tr["boxes"][a], b: tr["boxes"][b]})
                for f in sorted(ip):
                    best = None
                    for r in low[low.sample_id == f].itertuples():
                        box = [r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2]
                        if (f, tuple(np.round(box, 1))) in used:
                            continue
                        v = iou(box, inner[f])
                        if v > 0.4 and (best is None or v > best[0]):
                            best = (v, box)
                    if best:
                        tr["boxes"][f] = [float(v) for v in best[1]]
                        from_det.append(f)
                    else:
                        tr["boxes"][f] = inner[f]
                        from_int.append(f)
            if from_det:
                self.note("fill gap", f"track {tid}", span(from_det), f"{len(from_det)} boxes from low-confidence detections", why, "det")
            if from_int:
                self.note("fill gap", f"track {tid}", span(from_int), f"{len(from_int)} boxes interpolated", why, "interp")

    # -- output
    def write(self) -> None:
        win = self.win
        to_xml(self.tracks, win.clip, win.ann / "corrected_cvat.xml")
        lines = [f"# Changelog: {win.clip}", "",
                 "Pre-fill: `prefill_cvat.xml` (4-class detector + ByteTrack). Output: `corrected_cvat.xml`. Source: det = raw detector box, "
                 "interp = linear interpolation, manual = read by eye. Track ids below are this script's working ids; the output renumbers them.", "",
                 "| # | op | target | frames | detail | reason | source |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {e['n']} | {e['op']} | {e['target']} | {e['frames']} | {e['detail']} | {e['why']} | {e['source']} |" for e in self.log]
        (win.ann / "changelog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        people = [t for t in self.tracks.values() if t["label"] == "person"]
        print(f"{len(people)} person tracks, {sum(len(t['boxes']) for t in people)} boxes; ignore tracks {sum(t['label'] == 'ignore' for t in self.tracks.values())}; "
              f"ball boxes {sum(len(t['boxes']) for t in self.tracks.values() if t['label'] == 'ball')}; {len(self.log)} changelog entries")
