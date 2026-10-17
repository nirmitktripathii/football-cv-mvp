"""Loader for SkillCorner open broadcast-tracking data (MIT licence, github.com/SkillCorner/opendata).

Each line of ``<id>_tracking_extrapolated.jsonl`` is one frame at 10 fps with every player SkillCorner
placed, flagged ``is_detected`` (seen by the broadcast camera in that frame) or extrapolated. Only the
detected positions come from the picture; the extrapolated ones are SkillCorner's own estimate, so
they are *not* ground truth and this module keeps the flag. Coordinates are metres, origin at the
pitch centre, x along the long side.

The data is not redistributed here: ``data/external/`` is git-ignored. Credit SkillCorner when using it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FPS = 10.0
COLUMNS = ["frame", "t_s", "period", "player_id", "team", "x", "y", "detected"]


def _team_map(match: dict) -> dict[int, str]:
    home, away = match["home_team"]["id"], match["away_team"]["id"]
    out = {}
    for p in match["players"]:
        out[p["id"]] = "home" if p["team_id"] == home else "away" if p["team_id"] == away else "other"
    return out


def parse_tracking(lines, teams: dict[int, str]) -> pd.DataFrame:
    """Long table of player positions from the jsonl lines; frames without a timestamp are skipped."""
    rows = []
    for line in lines:
        d = json.loads(line)
        if d["timestamp"] is None:
            continue
        for p in d["player_data"]:
            rows.append((d["frame"], d["frame"] / FPS, d["period"], p["player_id"],
                         teams.get(p["player_id"], "other"), p["x"], p["y"], bool(p["is_detected"])))
    return pd.DataFrame(rows, columns=COLUMNS)


def load_match(match_dir: str | Path, match_id: int, refresh: bool = False) -> pd.DataFrame:
    """Tracking table for one match, cached as parquet next to the raw files."""
    match_dir = Path(match_dir)
    cache = match_dir / f"{match_id}_tracking.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    match = json.loads((match_dir / f"{match_id}_match.json").read_text())
    with open(match_dir / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        table = parse_tracking(f, _team_map(match))
    table.to_parquet(cache, index=False)
    log.info("parsed %d player-frames for match %s", len(table), match_id)
    return table


def detected_runs(table: pd.DataFrame, min_len_s: float, max_hole_frames: int = 1) -> pd.DataFrame:
    """Contiguous runs where a player is detected, one row per run (frame steps up to ``max_hole_frames`` stay in one run; 1 = consecutive frames only)."""
    runs = []
    for pid, g in table[table["detected"]].groupby("player_id"):
        f = g["frame"].to_numpy()
        breaks = np.flatnonzero(np.diff(f) > max_hole_frames) + 1
        for seg in np.split(np.arange(len(f)), breaks):
            if (f[seg[-1]] - f[seg[0]]) / FPS >= min_len_s:
                runs.append({"player_id": pid, "team": g["team"].iloc[0], "f0": f[seg[0]], "f1": f[seg[-1]]})
    return pd.DataFrame(runs, columns=["player_id", "team", "f0", "f1"])


VIEW_COLUMNS = ["frame", "x_tl", "y_tl", "x_bl", "y_bl", "x_br", "y_br", "x_tr", "y_tr"]


def load_views(match_dir: str | Path, match_id: int, refresh: bool = False) -> pd.DataFrame:
    """Camera-view polygon (pitch metres) per frame, corners ordered top-left, bottom-left, bottom-right, top-right."""
    match_dir = Path(match_dir)
    cache = match_dir / f"{match_id}_views.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    rows = []
    with open(match_dir / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            c = d.get("image_corners_projection")
            if d["timestamp"] is None or not c or c.get("x_top_left") is None:
                continue
            rows.append((d["frame"], *(c[f"{a}_{b}"] for b in ("top_left", "bottom_left", "bottom_right", "top_right") for a in ("x", "y"))))
    table = pd.DataFrame(rows, columns=VIEW_COLUMNS)
    table.to_parquet(cache, index=False)
    return table


def inside_margin_m(views: pd.DataFrame, frame: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed distance in metres from each point to the view polygon boundary (positive inside, NaN without a view)."""
    v = views.set_index("frame").reindex(frame)
    corners = np.stack([v[["x_tl", "y_tl"]].to_numpy(), v[["x_bl", "y_bl"]].to_numpy(),
                        v[["x_br", "y_br"]].to_numpy(), v[["x_tr", "y_tr"]].to_numpy()], axis=1)   # (n, 4, 2)
    nxt = np.roll(corners, -1, axis=1)
    edge = nxt - corners
    rel = np.stack([x, y], axis=1)[:, None, :] - corners
    cross = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]
    area = edge[..., 0][:, [0]] * edge[..., 1][:, [1]] - edge[..., 1][:, [0]] * edge[..., 0][:, [1]]
    sign = np.where(area >= 0, 1.0, -1.0)            # orientation of the polygon
    return (sign * cross / np.maximum(np.linalg.norm(edge, axis=2), 1e-9)).min(axis=1)
