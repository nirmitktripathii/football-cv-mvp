"""Inference with the fine-tuned YOLOX-m 4-class football detector.

The model was trained in ``notebooks/yolox_finetune_colab.ipynb`` with the YOLOX source at ``YOLOX_COMMIT``. This
module follows that commit's own inference path (``tools/demo.py``): ``ValTransform`` letterboxing, the model's built-in
decode in eval mode, then ``yolox.utils.postprocess`` (score = objectness x class probability, NMS). It does not
re-implement any of those. YOLOX is not a pip dependency: ``yolox_dir`` is a checkout of the pinned commit, put on
``sys.path`` when the predictor is built, and a checkout at any other commit is refused.

Boxes are returned in original-frame pixels as ``(xyxy, confidence, class index)`` arrays; ``detector.frame_rows`` turns
them into detection rows exactly as for the Ultralytics models.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
# output index -> class, the alphabetical order of the training set's categories (dataset manifest "classes")
CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
DEPTH, WIDTH = 0.67, 0.75   # YOLOX-m
NMS_IOU = 0.65              # the training experiment's nmsthre (YOLOX default)

Prediction = tuple[np.ndarray, np.ndarray, np.ndarray]


def letterbox_ratio(height: int, width: int, test_size: tuple[int, int]) -> float:
    """Scale ``ValTransform`` applies: the frame is resized by this factor and padded to ``test_size`` (h, w)."""
    return min(test_size[0] / height, test_size[1] / width)


def to_original_pixels(outputs, ratio: float, height: int, width: int) -> Prediction:
    """``postprocess`` rows ``(x1, y1, x2, y2, obj, cls_conf, cls)`` in network pixels -> original-frame arrays."""
    if outputs is None or len(outputs) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    out = np.array(outputs, dtype=float)
    xyxy = out[:, :4] / ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    return xyxy, out[:, 4] * out[:, 5], out[:, 6].astype(int)


def checkout_commit(yolox_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(yolox_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class YoloxPredictor:
    """Loads one checkpoint and predicts batches of image files."""

    def __init__(self, weights: Path, yolox_dir: Path, device: int | str, test_size: tuple[int, int],
                 conf_floor: float = 0.01, nms_iou: float = NMS_IOU):
        weights, yolox_dir = Path(weights), Path(yolox_dir)
        if not weights.exists():
            raise FileNotFoundError(f"YOLOX checkpoint not found: {weights}")
        if checkout_commit(yolox_dir) != YOLOX_COMMIT:
            raise RuntimeError(f"{yolox_dir} is not at the pinned YOLOX commit {YOLOX_COMMIT}")
        if str(yolox_dir) not in sys.path:
            sys.path.insert(0, str(yolox_dir))
        import torch
        import yolox
        from yolox.data.data_augment import ValTransform
        from yolox.exp import Exp
        from yolox.utils import postprocess

        if Path(yolox.__file__).resolve().parent.parent != yolox_dir.resolve():
            raise RuntimeError(f"imported yolox from {yolox.__file__}, not from {yolox_dir}")
        self._torch, self._postprocess, self._preproc = torch, postprocess, ValTransform(legacy=False)
        self.test_size, self.conf_floor, self.nms_iou = tuple(test_size), conf_floor, nms_iou
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        exp = Exp()
        exp.depth, exp.width, exp.num_classes = DEPTH, WIDTH, len(CLASS_NAMES)
        model = exp.get_model()
        # the checkpoint is this project's own training output (its SHA-256 is part of the provenance string), and it
        # holds more than tensors (optimiser state, numpy scalars), so the weights-only restriction is switched off
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])   # strict: a different architecture or class count fails here
        self.model = model.to(self.device).eval()
        self.provenance = f"yolox:{weights.name}:sha256-{checkpoint_sha256(weights)[:12]}:test{test_size[0]}x{test_size[1]}"

    def predict(self, paths: list[str]) -> list[Prediction]:
        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"cannot read image {p}")
            frames.append(img)
        blobs = np.stack([self._preproc(img, None, self.test_size)[0] for img in frames])
        batch = self._torch.from_numpy(blobs).float().to(self.device)
        with self._torch.no_grad():
            decoded = self.model(batch)   # eval mode: the head already decodes to (cx, cy, w, h, obj, class probs)
            kept = self._postprocess(decoded, len(CLASS_NAMES), self.conf_floor, self.nms_iou, class_agnostic=True)
        return [
            to_original_pixels(None if k is None else k.cpu().numpy(), letterbox_ratio(*img.shape[:2], self.test_size),
                               *img.shape[:2])
            for k, img in zip(kept, frames)
        ]
