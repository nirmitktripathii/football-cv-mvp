"""Configuration loading and run-wide paths.

Every stage takes the same nested dict, loaded from YAML and optionally
overridden with ``key.subkey=value`` strings from the CLI.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config. A top-level ``extends: <file>`` (relative to this file) is loaded first and
    this file's values are deep-merged over it, so a per-match config only lists what differs."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("extends", None)
    if base is not None:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return apply_overrides(cfg, overrides or [])


def deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``top`` merged in; nested dicts merge, anything else replaces."""
    out = copy.deepcopy(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with ``a.b.c=value`` overrides applied (values are YAML-parsed)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def resolve(path: str | Path) -> Path:
    """Paths in the config are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def run_dir(cfg: dict[str, Any]) -> Path:
    """Processed outputs for one clip: data/processed/<clip_id>/."""
    d = resolve(cfg["paths"]["processed_dir"]) / cfg["clip"]["id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifacts_dir(cfg: dict[str, Any]) -> Path:
    d = resolve(cfg["paths"]["artifacts_dir"]) / cfg["clip"]["id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
