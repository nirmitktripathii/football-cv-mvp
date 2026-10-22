"""Build a YOLOX (COCO-layout) fine-tuning dataset for the 4-class football detector.

    python scripts/build_finetune_dataset.py --out data/finetune/v1 \\
        --source data/external/football-players-detection-3zvbc \\
        --source data/external/another_export --prefer ball,referee --keep-other-frac 0.25

Training data: one or more Roboflow "YOLOv8" exports (``data.yaml`` + ``train|valid|test/{images,labels}``).
Class names are mapped to ball / goalkeeper / player / referee through ``CLASS_ALIASES``; any other class
is dropped and reported, never guessed.

Validation data: the audited evaluation windows (``data/annotations/inter_napoli_fb4_<w>``), converted from
CVAT. They are the project's held-out score, so no training image may look like them: training images whose
perceptual hash is within ``--leak-hamming`` bits of any window frame are removed and counted.

Output (the layout ``yolox.data.COCODataset`` reads: ``<data_dir>/annotations/<json>``, ``<data_dir>/<name>/``)::

    <out>/annotations/instances_train2017.json   instances_val2017.json
    <out>/train2017/*.jpg   <out>/val2017/*.jpg
    <out>/manifest.json     sources, licences, class counts, everything dropped and why

``--prefer`` keeps every image holding one of the listed classes and only ``--keep-other-frac`` of the rest
(seeded), which is how referee and ball examples are over-weighted without duplicating images.

``--negatives DIR`` adds hard-negative patches mined by ``scripts/mine_drone_negatives.py`` (the hovering drone,
the penalty spot): images with no boxes, so the detector learns that those objects are background. Each patch
is repeated ``--negatives-repeat`` times. They come from the PoC clip, whose time spans the miner already
excludes from the validation windows; the dHash leakage check does not apply to a crop, so it is not run on them.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_cv.detection.negatives import load_negatives  # noqa: E402
from football_cv.evaluation import cvat  # noqa: E402

log = logging.getLogger("build_finetune_dataset")

CLASSES = ("ball", "goalkeeper", "player", "referee")      # category ids 1..4 in the COCO json
CLASS_ALIASES = {
    "ball": "ball", "football": "ball", "soccer ball": "ball",
    "goalkeeper": "goalkeeper", "goal keeper": "goalkeeper", "keeper": "goalkeeper", "gk": "goalkeeper",
    "player": "player", "players": "player",
    "referee": "referee", "main referee": "referee", "main_referee": "referee",
}   # "side referee" / "assistant_referee" are deliberately absent: the pipeline scores them as off-pitch staff
WINDOWS = ("a1_open_t600", "a2_corner_t2707", "a3_open_t4800")
SPLIT_DIRS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}
MIN_BOX_PX = 2.0
FRAME_SIZE = (1280, 720)


def dhash(img: np.ndarray) -> int:
    """64-bit difference hash; robust to re-encoding and mild resizing, not to crops."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_yolo_labels(path: Path, w: int, h: int, names: list[str]) -> list[tuple[str, list[float]]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, (cx, cy, bw, bh) = int(parts[0]), (float(v) for v in parts[1:5])
        out.append((names[cls], [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]))
    return out


def load_source(src: Path):
    """(items, class names, metadata) for one Roboflow YOLO export; items are (image, label file, split)."""
    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][k] for k in sorted(cfg["names"])]
    rf = cfg.get("roboflow") or {}
    meta = {"path": str(src), "names": names, **{k: rf[k] for k in ("project", "license", "url", "version") if k in rf}}
    items = []
    for split in SPLIT_DIRS:
        img_dir = src / split / "images"
        if img_dir.is_dir():
            items += [(p, src / split / "labels" / (p.stem + ".txt"), split) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
    return items, names, meta


def window_frames() -> list[dict]:
    """Audited window frames with their boxes in the 4-class label space."""
    rows = []
    for w in WINDOWS:
        clip = f"inter_napoli_fb4_{w}"
        gt = cvat.parse_cvat_xml((ROOT / "data" / "annotations" / clip / "annotations.xml").read_text(encoding="utf-8"))
        run = ROOT / "data" / "processed" / clip
        for frame, grp in gt.groupby("frame"):
            boxes, skipped = [], Counter()
            for r in grp.itertuples():
                if r.label == "ball":
                    cls = "ball"
                elif r.label == "person" and r.role in ("player", "goalkeeper", "referee"):
                    cls = r.role
                else:
                    skipped[f"{r.label}/{r.role or '-'}"] += 1
                    continue
                boxes.append((cls, [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]))
            rows.append({"image": run / "frames" / f"{int(frame):06d}.jpg", "name": f"{w}_{int(frame):06d}.jpg",
                         "boxes": boxes, "skipped": skipped})
    return rows


class CocoBuilder:
    def __init__(self, out: Path, split_dir: str, dry_run: bool):
        self.dir, self.dry_run = out / split_dir, dry_run
        if not dry_run:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.images, self.annotations, self.counts = [], [], Counter()

    def add(self, src_img: Path, name: str, boxes: list[tuple[str, list[float]]], size: tuple[int, int]) -> None:
        if not self.dry_run:
            shutil.copy2(src_img, self.dir / name)
        iid = len(self.images) + 1
        self.images.append({"id": iid, "file_name": name, "width": size[0], "height": size[1]})
        for cls, (x, y, w, h) in boxes:
            self.annotations.append({"id": len(self.annotations) + 1, "image_id": iid, "category_id": CLASSES.index(cls) + 1,
                                     "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)], "area": round(w * h, 2), "iscrowd": 0})
            self.counts[cls] += 1

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        categories = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
        path.write_text(json.dumps({"images": self.images, "annotations": self.annotations, "categories": categories}), encoding="utf-8")


def clip_box(box: list[float], w: int, h: int) -> list[float] | None:
    x, y, bw, bh = box
    x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(w), x + bw), min(float(h), y + bh)
    return [x1, y1, x2 - x1, y2 - y1] if (x2 - x1) >= MIN_BOX_PX and (y2 - y1) >= MIN_BOX_PX else None


def _mapped_boxes(raw_boxes, w: int, h: int, source: str, manifest: dict) -> list[tuple[str, list[float]]]:
    boxes = []
    for raw, box in raw_boxes:
        cls = CLASS_ALIASES.get(raw.strip().lower())
        if cls is None:
            manifest["unmapped_classes"][f"{source}:{raw}"] += 1
            continue
        clipped = clip_box(box, w, h)
        if clipped is None:
            manifest["dropped"]["tiny_or_empty_box"] += 1
            continue
        boxes.append((cls, clipped))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", action="append", required=True, type=Path, help="Roboflow YOLOv8 export dir; repeatable")
    ap.add_argument("--prefer", default="ball,referee", help="classes whose images are always kept ('' = keep all)")
    ap.add_argument("--keep-other-frac", type=float, default=0.25)
    ap.add_argument("--leak-hamming", type=int, default=6, help="drop training images this close (64-bit dHash) to a validation frame")
    ap.add_argument("--dup-hamming", type=int, default=2, help="drop training images this close to an already kept one")
    ap.add_argument("--negatives", action="append", default=[], type=Path, help="mined hard-negative set (mine_drone_negatives.py); repeatable")
    ap.add_argument("--negatives-repeat", type=int, default=5, help="copies of each negative patch in the training set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = random.Random(args.seed)
    prefer = {c for c in args.prefer.split(",") if c}
    if prefer - set(CLASSES):
        raise SystemExit(f"--prefer has unknown classes {sorted(prefer - set(CLASSES))}")
    if args.out.exists() and any(args.out.iterdir()) and not args.dry_run:
        raise SystemExit(f"{args.out} is not empty; choose a new --out (datasets are versioned, not overwritten)")

    val_rows = window_frames()
    val_hashes = [dhash(cv2.imread(str(r["image"]))) for r in val_rows]
    log.info("validation: %d window frames", len(val_rows))

    train = CocoBuilder(args.out, "train2017", args.dry_run)
    kept_hashes: list[int] = []
    manifest = {"classes": list(CLASSES), "sources": [], "dropped": Counter(), "unmapped_classes": Counter(),
                "args": {"prefer": sorted(prefer), "keep_other_frac": args.keep_other_frac, "leak_hamming": args.leak_hamming,
                         "dup_hamming": args.dup_hamming, "seed": args.seed}}
    for si, src in enumerate(args.source):
        items, names, meta = load_source(src)
        n_before = len(train.images)
        for img_path, lab_path, _split in items:
            img = cv2.imread(str(img_path))
            if img is None:
                manifest["dropped"]["unreadable"] += 1
                continue
            h, w = img.shape[:2]
            boxes = _mapped_boxes(read_yolo_labels(lab_path, w, h, names), w, h, src.name, manifest)
            if not boxes:
                manifest["dropped"]["no_mapped_boxes"] += 1
                continue
            if prefer and not ({c for c, _ in boxes} & prefer) and rng.random() > args.keep_other_frac:
                manifest["dropped"]["not_preferred_subsampled"] += 1
                continue
            hsh = dhash(img)
            if any(hamming(hsh, v) <= args.leak_hamming for v in val_hashes):
                manifest["dropped"]["too_similar_to_validation_window"] += 1
                continue
            if any(hamming(hsh, v) <= args.dup_hamming for v in kept_hashes):
                manifest["dropped"]["near_duplicate_of_kept_image"] += 1
                continue
            kept_hashes.append(hsh)
            train.add(img_path, f"s{si}_{img_path.name}", boxes, (w, h))
        manifest["sources"].append({**meta, "images_kept": len(train.images) - n_before})
        log.info("%s: kept %d of %d images", src.name, len(train.images) - n_before, len(items))

    n_neg = 0
    for img_path, name in load_negatives(args.negatives, args.negatives_repeat):
        img = cv2.imread(str(img_path))
        train.add(img_path, name, [], (img.shape[1], img.shape[0]))
        n_neg += 1
    manifest["negatives"] = {"sets": [str(d) for d in args.negatives], "repeat": args.negatives_repeat, "images_added": n_neg}

    val, skipped = CocoBuilder(args.out, "val2017", args.dry_run), Counter()
    for r in val_rows:
        skipped.update(r["skipped"])
        val.add(r["image"], r["name"], r["boxes"], FRAME_SIZE)
    manifest["train"] = {"images": len(train.images), "boxes": dict(train.counts)}
    manifest["val"] = {"images": len(val.images), "boxes": dict(val.counts), "windows": list(WINDOWS),
                       "skipped_not_trainable": dict(skipped)}
    manifest["dropped"], manifest["unmapped_classes"] = dict(manifest["dropped"]), dict(manifest["unmapped_classes"])
    if not args.dry_run:
        train.write(args.out / "annotations" / "instances_train2017.json")
        val.write(args.out / "annotations" / "instances_val2017.json")
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
