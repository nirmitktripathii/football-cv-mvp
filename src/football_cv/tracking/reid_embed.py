"""Appearance embeddings from the SRITrack checkpoint (``models/sritrack/model.pth.tar-60``).

The checkpoint is a training snapshot: ``state_dict``, ``epoch`` (60), ``optimizer``, ``scheduler``. Its source
code is not in this repository, so the architecture is *inferred from the weight names and shapes*, and that
inference is checked by a strict load:

* ``model.*`` is a ViT-B/16 with a class token, 4 register tokens, per-block layer scale (``gamma_1/2``), a
  bias-free ``qkv`` and no position-embedding weights; the key names are those of timm's ``Eva`` class, which is
  built with rotary position embedding to match. Rotary embeddings hold no weights, so a strict load cannot
  confirm their settings (base, input size). Input size and normalisation are not stored in the checkpoint
  either; both are choices made here, and the embedding quality has to be measured, not assumed
  (``scripts/reid_embedding_test.py``).
* ``fc`` is ``Linear(768, 512)`` + ``BatchNorm1d(512)``, the usual ReID neck; ``classifier`` has 1278 identities
  from its training set and is not used.

Which of ``cls`` / ``mean`` / ``fc`` the training used for retrieval is also unknown, so all three are returned
and the test script scores each.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURES = ("cls", "mean", "fc")


class SriTrackEmbedder:
    def __init__(self, checkpoint: str | Path, input_hw: tuple[int, int] = (256, 128), device: str = "cpu"):
        from timm.models.eva import Eva  # imported here: timm is only needed when embeddings are used

        path = Path(checkpoint)
        # weights_only=True: the file is an untrusted pickle; this refuses anything but tensors and containers
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.input_hw = input_hw
        self.device = torch.device(device)
        self.backbone = Eva(
            img_size=input_hw, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, mlp_ratio=4.0,
            num_classes=0, use_abs_pos_emb=False, use_rot_pos_emb=True, num_reg_tokens=4, init_values=1e-5,
            global_pool="",
        )
        self.backbone.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")}, strict=True)
        self.neck = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512))
        self.neck.load_state_dict({k[3:]: v for k, v in state.items() if k.startswith("fc.")}, strict=True)
        self.backbone.eval().to(self.device)
        self.neck.eval().to(self.device)
        self.n_prefix = self.backbone.num_prefix_tokens
        log.info("SRITrack embedder loaded (sha256 %s..., input %s, %d prefix tokens)", self.sha256[:12], input_hw, self.n_prefix)

    def _prepare(self, crop_bgr: np.ndarray) -> torch.Tensor:
        h, w = self.input_hw
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1))

    @torch.no_grad()
    def embed(self, crops_bgr: list[np.ndarray], batch: int = 32) -> dict[str, np.ndarray]:
        """L2-normalised features per crop: ``cls`` (768), ``mean`` of patch tokens (768) and ``fc`` (512)."""
        out: dict[str, list[np.ndarray]] = {k: [] for k in FEATURES}
        for i in range(0, len(crops_bgr), batch):
            x = torch.stack([self._prepare(c) for c in crops_bgr[i : i + batch]]).to(self.device)
            tokens = self.backbone.forward_features(x)
            cls = tokens[:, 0]
            feats = {"cls": cls, "mean": tokens[:, self.n_prefix :].mean(1), "fc": self.neck(cls)}
            for k, v in feats.items():
                out[k].append(nn.functional.normalize(v, dim=1).cpu().numpy())
        return {k: np.concatenate(v) if v else np.empty((0, 0), np.float32) for k, v in out.items()}
