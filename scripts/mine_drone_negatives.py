"""Cut hard-negative patches (the hovering drone, the penalty spot) out of a pipeline run.

    python scripts/mine_drone_negatives.py --run data/processed/poc_seg1rb \
        --samples 76,77,79 --samples-file ids.txt --out data/finetune/negatives/drone_v1 --exclude-range 600:615

The listed samples are ones a person has looked at and confirmed show a false "ball". Each gets a
``--size`` px patch centred on the false detection and holding no tracked person; the patches go into the
fine-tune set with no boxes (``build_finetune_dataset.py --negatives``). ``--exclude-range`` (video seconds,
repeatable) keeps patches out of the time spans of the audited validation windows.

``--rig-tracks 83,87`` mines the other way round: the listed tracks are the camera rig detected as a *person*, the patch is
centred on the rig box, and only real (non-rig) people block it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_cv.detection.negatives import mine_negative_patches, mine_person_box_patches  # noqa: E402


def parse_ids(samples: str, samples_file: Path | None) -> list[int]:
    text = samples + "," + (samples_file.read_text(encoding="utf-8").replace("\n", ",") if samples_file else "")
    return sorted({int(t) for t in text.replace(" ", "").split(",") if t})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--samples", default="")
    ap.add_argument("--samples-file", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--exclude-range", action="append", default=[], help="START:END video seconds to leave out; repeatable")
    ap.add_argument("--label", default="drone")
    ap.add_argument("--rig-tracks", default="", help="track ids confirmed as the rig detected as a person; mines person-box patches")
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; choose a new --out")

    ids = parse_ids(args.samples, args.samples_file)
    frames = pd.read_parquet(args.run / "frames.parquet").set_index("sample_id")
    spans = [tuple(float(v) for v in r.split(":")) for r in args.exclude_range]
    kept_ids, excluded = [], []
    for sid in ids:
        ts = float(frames.loc[sid, "timestamp_s"])
        (excluded if any(a <= ts <= b for a, b in spans) else kept_ids).append(sid)
    tracks = pd.read_parquet(args.run / "tracks_projected.parquet")
    first = cv2.imread(str(args.run / frames.loc[kept_ids[0], "path"]))
    frame_wh = (first.shape[1], first.shape[0])
    rig_tracks = [int(t) for t in args.rig_tracks.split(",") if t.strip()]
    if rig_tracks:
        patches, skipped = mine_person_box_patches(tracks, kept_ids, rig_tracks, args.size, frame_wh)
    else:
        patches, skipped = mine_negative_patches(tracks, kept_ids, args.size, frame_wh)

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True)
    entries = []
    for i, p in enumerate(patches):
        x1, y1, x2, y2 = p["window"]
        img = cv2.imread(str(args.run / frames.loc[p["sample_id"], "path"]))
        name = f"{args.label}_{args.run.name}_{p['sample_id']:04d}_{i:02d}.jpg"
        cv2.imwrite(str(img_dir / name), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        entry = {"file": name, "sample_id": p["sample_id"], "timestamp_s": float(frames.loc[p["sample_id"], "timestamp_s"]),
                 "window": list(p["window"])}
        if rig_tracks:
            entry["rig_box"] = [round(float(v), 1) for v in p["rig_box"]]
        else:
            entry["false_ball_box"] = [round(float(v), 1) for v in p["ball_box"]]
        entries.append(entry)
    manifest = {"run": str(args.run), "label": args.label, "size": args.size, "patches": entries,
                "skipped": {str(k): v for k, v in skipped.items()}, "excluded_by_time": excluded,
                "exclude_ranges_s": spans,
                "rig_tracks": rig_tracks,
                "confirmed_by": "visual inspection of the rig person-box gallery" if rig_tracks
                else "visual inspection of the ball candidate gallery"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(entries)} patches from {len(ids)} samples; skipped {len(skipped)}, excluded by time {len(excluded)}")


if __name__ == "__main__":
    main()
