"""Top-down pitch animation (matplotlib) driven only by observed, calibrated positions.

One animation frame per sampled video frame. A player is drawn only in frames
where it was actually detected, tracked and calibrated; trails connect only
consecutive observations of one run (no interpolation across gaps). Frames
without a calibrated wide view say so instead of showing stale positions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import animation  # noqa: E402
from mplsoccer import Pitch  # noqa: E402

from football_cv.trajectories.clean import usable  # noqa: E402

log = logging.getLogger(__name__)

TEAM_STYLE = {
    "home": {"color": "#1f5fbf", "edge": "white", "label": "home"},
    "away": {"color": "#f4f4f4", "edge": "black", "label": "away"},
    "other": {"color": "#f5c400", "edge": "black", "label": "referee / GK / other"},
    "unknown": {"color": "#9a9a9a", "edge": "black", "label": "unassigned"},
}
BALL_STYLE = {"color": "#ff7f0e", "edge": "black"}
TRAIL_S = 2.0
PITCH_KW = {"pitch_type": "custom", "pitch_length": 105, "pitch_width": 68, "pitch_color": "#2e7d32", "line_color": "white"}


def _colour(track_id: int, team: str) -> str:
    return BALL_STYLE["color"] if track_id < 0 else TEAM_STYLE.get(team, TEAM_STYLE["unknown"])["color"]


def _draw_trails(ax, use: pd.DataFrame, cur: pd.DataFrame, t: float, trail_s: float) -> list:
    lines = []
    visible = set(cur["track_id"])
    recent = use[(use["timestamp_s"] > t - trail_s) & (use["timestamp_s"] <= t)]
    for (tid, _), g in recent.groupby(["track_id", "run_id"]):
        if len(g) > 1 and tid in visible:
            (ln,) = ax.plot(g["x_s"], g["y_s"], color=_colour(tid, g["team_id"].iloc[0]), alpha=0.6, lw=1.4, zorder=4)
            lines.append(ln)
    return lines


def render_pitch_animation(
    traj: pd.DataFrame,
    frames: pd.DataFrame,
    calib: pd.DataFrame,
    out_path: Path,
    start_s: float | None = None,
    end_s: float | None = None,
    team_names: dict[str, str] | None = None,
    fps: float = 5.0,
    title: str = "",
) -> Path:
    """Write an MP4 (or GIF if the suffix is .gif) of the clip window [start_s, end_s]."""
    team_names = team_names or {}
    fr = frames
    if start_s is not None:
        fr = fr[fr["timestamp_s"] >= start_s]
    if end_s is not None:
        fr = fr[fr["timestamp_s"] <= end_s]
    use = usable(traj)
    by_sample = {k: g for k, g in use.groupby("sample_id")}
    status = calib.set_index("sample_id")["calibration_status"].to_dict()

    pitch = Pitch(linewidth=1.2, **PITCH_KW)
    fig, ax = pitch.draw(figsize=(10.5, 7.4))
    fig.subplots_adjust(top=0.9, bottom=0.1)
    scat = {
        k: ax.scatter([], [], s=90, c=v["color"], edgecolors=v["edge"], linewidths=1.0, zorder=5,
                      label=team_names.get(k, v["label"]))
        for k, v in TEAM_STYLE.items()
    }
    ball = ax.scatter([], [], s=45, c=BALL_STYLE["color"], edgecolors=BALL_STYLE["edge"], zorder=6, label="ball")
    trails: list = []
    header = fig.text(0.5, 0.95, "", ha="center", va="center", fontsize=11)
    note = fig.text(0.5, 0.025, "", ha="center", va="center", fontsize=9, color="#444444")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=8, frameon=False)
    rows = list(fr.itertuples())

    def update(i):
        for ln in trails:
            ln.remove()
        trails.clear()
        r = rows[i]
        cur = by_sample.get(r.sample_id)
        clock = f"   match clock {r.match_clock}" if getattr(r, "match_clock", None) else ""
        header.set_text(f"{title}   t = {r.timestamp_s:6.1f} s   frame {r.frame_id}{clock}")
        empty = np.empty((0, 2))
        for k, sc in scat.items():
            pts = empty if cur is None else cur[(cur["track_id"] >= 0) & (cur["team_id"] == k)][["x_s", "y_s"]].to_numpy()
            sc.set_offsets(pts)
        ball.set_offsets(empty if cur is None else cur[cur["track_id"] < 0][["x_s", "y_s"]].to_numpy())
        if cur is None:
            st = status.get(r.sample_id, "not_wide_view")
            note.set_text(f"no calibrated positions in this frame ({st.replace('_', ' ')})")
            return []
        note.set_text(
            f"{int((cur['track_id'] >= 0).sum())} tracked people, calibration: {status.get(r.sample_id)}"
            "   |   observed positions only, no interpolation"
        )
        trails.extend(_draw_trails(ax, use, cur, r.timestamp_s, TRAIL_S))
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(rows), interval=1000 / fps, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        anim.save(out_path, writer=writer)
    plt.close(fig)
    log.info("Pitch animation (%d frames) -> %s", len(rows), out_path)
    return out_path


def plot_snapshot(traj: pd.DataFrame, sample_id: int, ax=None, team_names: dict[str, str] | None = None,
                  trail_s: float = TRAIL_S):
    """Static pitch figure for one sampled frame (used by the dashboard and the report)."""
    team_names = team_names or {}
    use = usable(traj)
    pitch = Pitch(**PITCH_KW)
    if ax is None:
        fig, ax = pitch.draw(figsize=(8, 5.6))
    else:
        fig = ax.figure
        pitch.draw(ax=ax)
    cur = use[use["sample_id"] == sample_id]
    if len(cur):
        _draw_trails(ax, use, cur, cur["timestamp_s"].iloc[0], trail_s)
    for k, v in TEAM_STYLE.items():
        pts = cur[(cur["track_id"] >= 0) & (cur["team_id"] == k)]
        if len(pts):
            ax.scatter(pts["x_s"], pts["y_s"], s=90, c=v["color"], edgecolors=v["edge"], zorder=5,
                       label=team_names.get(k, v["label"]))
    b = cur[cur["track_id"] < 0]
    if len(b):
        ax.scatter(b["x_s"], b["y_s"], s=45, c=BALL_STYLE["color"], edgecolors="black", zorder=6, label="ball")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=8, frameon=False)
    return fig, ax
