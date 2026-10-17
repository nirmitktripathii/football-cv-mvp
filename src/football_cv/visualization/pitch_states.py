"""Top-down pitch animation that shows *how* each position is known.

Driven by ``player_states.parquet`` (``reconstruction/reconstruct.py``). The original observed-only
renderer (``pitch_animation.py``) is kept unchanged.

Encoding:

* solid marker, team colour            observed (detected, tracked, calibrated)
* translucent marker, dashed edge, ring  interpolated between two observations (ring = 2-sigma)
* faint marker, dotted edge, ring      predicted from history only (ring = 2-sigma)
* diamond with black edge              manually annotated position
* dashed ring with "?"                 location unknown: estimate withheld, ring = 2-sigma
* grey band below the near touchline   off-pitch region (annotated players only)
* red banner / border                  camera cut, or no calibrated view (nothing is drawn then)

Frames come from the sampled-frame table, so the pitch and the broadcast video share one clock
(``timestamp_s``; the header prints the same frame id as the annotated broadcast video).
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from mplsoccer import Pitch  # noqa: E402

from football_cv.visualization.pitch_animation import BALL_STYLE, PITCH_KW, TEAM_STYLE  # noqa: E402

log = logging.getLogger(__name__)

TRAIL_S = 2.0
OFF_PITCH_M = 8.0          # depth of the off-pitch band drawn below y = 0
MAX_RING_M = 12.0          # uncertainty rings larger than this are not drawn (too uncertain to show)
CUT_BANNER_FRAMES = 3
MARK = 90


def _team(row_team: str) -> dict:
    return TEAM_STYLE.get(row_team, TEAM_STYLE["unknown"])


def draw_players(ax, rows: pd.DataFrame) -> tuple[list, dict]:
    """Draw one sample's state rows; returns (artists to remove later, counts by encoding)."""
    artists: list = []
    counts = {"observed": 0, "interpolated": 0, "predicted": 0, "annotated": 0, "unknown": 0}
    for r in rows.itertuples():
        st = _team(r.team_id)
        if r.position_source == "OBSERVED":
            artists.append(ax.scatter(r.x_m, r.y_m, s=MARK, c=st["color"], edgecolors=st["edge"], linewidths=1.0, zorder=5))
            counts["observed"] += 1
        elif r.position_source == "INTERPOLATED":
            artists.append(ax.scatter(r.x_m, r.y_m, s=MARK, c=st["color"], alpha=0.6, edgecolors="black",
                                      linewidths=1.5, linestyles="--", zorder=4))
            artists += _ring(ax, r, "--")
            counts["interpolated"] += 1
        elif r.position_source == "PREDICTED":
            artists.append(ax.scatter(r.x_m, r.y_m, s=MARK, c=st["color"], alpha=0.4, edgecolors="black",
                                      linewidths=1.3, linestyles=":", zorder=3))
            artists += _ring(ax, r, ":")
            counts["predicted"] += 1
        elif r.position_source == "MANUALLY_ANNOTATED":
            artists.append(ax.scatter(r.x_m, r.y_m, s=MARK, marker="D", c=st["color"], edgecolors="black",
                                      linewidths=2.0, zorder=6))
            counts["annotated"] += 1
        elif np.isfinite(r.x_est_m) and np.isfinite(r.sigma_m) and 2 * r.sigma_m <= MAX_RING_M:
            artists += _ring(ax, r, "--", fill_alpha=0.12)
            artists.append(ax.text(r.x_est_m, r.y_est_m, "?", ha="center", va="center", fontsize=9, color=st["edge"],
                                   fontweight="bold", zorder=5))
            counts["unknown"] += 1
    return artists, counts


def _ring(ax, r, linestyle: str, fill_alpha: float = 0.0) -> list:
    """2-sigma uncertainty disc around the published position (or, if withheld, around the estimate)."""
    if not np.isfinite(r.sigma_m) or 2 * r.sigma_m > MAX_RING_M:
        return []
    x = r.x_m if np.isfinite(r.x_m) else r.x_est_m
    y = r.y_m if np.isfinite(r.y_m) else r.y_est_m
    st = _team(r.team_id)
    ring = Circle((x, y), 2 * r.sigma_m, ls=linestyle, lw=0.9, zorder=2, ec=(0, 0, 0, 0.55),
                  fc=(*matplotlib.colors.to_rgb(st["color"]), fill_alpha) if fill_alpha else "none")
    ax.add_patch(ring)
    return [ring]


def _trails(ax, observed: pd.DataFrame, t: float) -> list:
    """Trails of identities observed now, over their latest unbroken observed run within TRAIL_S."""
    out = []
    now = set(observed.loc[observed["timestamp_s"] == t, "identity_id"])
    recent = observed[(observed["timestamp_s"] > t - TRAIL_S) & (observed["timestamp_s"] <= t) & observed["identity_id"].isin(now)]
    for _, g in recent.groupby("identity_id"):
        g = g.sort_values("sample_id")
        run_no = (g["sample_id"].diff().fillna(1) != 1).cumsum()
        run = g[run_no == run_no.iloc[-1]]
        if len(run) > 1:
            (ln,) = ax.plot(run["x_m"], run["y_m"], color=_team(run["team_id"].iloc[0])["color"], alpha=0.5, lw=1.3, zorder=3)
            out.append(ln)
    return out


def _legend(ax, team_names: dict) -> None:
    grey = "#888888"
    rgb = matplotlib.colors.to_rgb(grey)
    handles = [
        Line2D([], [], marker="o", ls="", mfc=v["color"], mec=v["edge"], ms=8, label=team_names.get(k, v["label"]))
        for k, v in TEAM_STYLE.items()
    ]
    handles += [
        Line2D([], [], marker="o", ls="", mfc=BALL_STYLE["color"], mec="black", ms=6, label="ball"),
        Line2D([], [], marker="o", ls="", mfc=grey, mec="white", ms=8, label="observed"),
        Line2D([], [], marker="o", ls="--", mfc=(*rgb, 0.5), mec="white", ms=8, label="interpolated"),
        Line2D([], [], marker="o", ls=":", mfc=(*rgb, 0.25), mec="white", ms=8, label="predicted"),
        Line2D([], [], marker="D", ls="", mfc=grey, mec="black", ms=7, label="annotated"),
        Line2D([], [], marker="$?$", ls="", color="black", ms=8, label="location unknown"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=7.5, frameon=False)


def _pitch(figsize):
    pitch = Pitch(linewidth=1.2, pad_bottom=OFF_PITCH_M, **PITCH_KW)
    fig, ax = pitch.draw(figsize=figsize)
    ax.fill_between([-5, 110], -OFF_PITCH_M, 0, color="#555555", alpha=0.55, zorder=1)
    ax.text(52.5, -OFF_PITCH_M / 2, "off pitch (annotated players only)", ha="center", va="center", fontsize=7, color="white")
    return fig, ax


def plot_state_snapshot(states: pd.DataFrame, sample_id: int, team_names: dict | None = None, ax=None):
    """Static figure of one sample with provenance styling (used by the dashboard and tests)."""
    team_names = team_names or {}
    fig, ax = _pitch((8.5, 6.0)) if ax is None else (ax.figure, ax)
    cur = states[states["sample_id"] == sample_id]
    if len(cur):
        _trails(ax, states[states["position_source"] == "OBSERVED"], float(cur["timestamp_s"].iloc[0]))
    _, counts = draw_players(ax, cur)
    _legend(ax, team_names)
    return fig, ax, counts


def render_state_animation(
    states: pd.DataFrame, ball: pd.DataFrame, frames: pd.DataFrame, shots: pd.DataFrame, calib: pd.DataFrame,
    out_path: Path, start_s: float | None = None, end_s: float | None = None, team_names: dict | None = None,
    fps: float = 5.0, title: str = "",
) -> Path:
    """MP4/GIF of [start_s, end_s]; ``ball`` is the cleaned trajectory table (ball rows only are used)."""
    team_names = team_names or {}
    fr = frames.merge(shots[["sample_id", "shot_id"]], on="sample_id", how="left")
    if start_s is not None:
        fr = fr[fr["timestamp_s"] >= start_s]
    if end_s is not None:
        fr = fr[fr["timestamp_s"] <= end_s]
    rows = list(fr.itertuples())
    by_sample = {k: g for k, g in states.groupby("sample_id")}
    observed = states[states["position_source"] == "OBSERVED"]
    ball_by = {k: g for k, g in ball[(ball["track_id"] < 0) & ~ball["jump"] & ball["x_s"].notna()].groupby("sample_id")}
    status = calib.set_index("sample_id")["calibration_status"].to_dict()

    fig, ax = _pitch((10.5, 7.8))
    fig.subplots_adjust(top=0.9, bottom=0.17)
    _legend(ax, team_names)
    header = fig.text(0.5, 0.968, "", ha="center", va="center", fontsize=11)
    note = fig.text(0.5, 0.017, "", ha="center", va="center", fontsize=9, color="#444444")
    banner = fig.text(0.5, 0.5, "", ha="center", va="center", fontsize=18, color="#c62828", fontweight="bold",
                      bbox={"fc": "white", "ec": "#c62828", "alpha": 0.9, "pad": 6})
    border = fig.add_artist(Rectangle((0.003, 0.003), 0.994, 0.994, transform=fig.transFigure, fill=False,
                                      ec="#c62828", lw=6, visible=False))
    ball_sc = ax.scatter([], [], s=45, c=BALL_STYLE["color"], edgecolors="black", zorder=7)
    live: list = []
    cut_left = {"n": 0}
    prev_shot = {"id": None}

    def update(i):
        for a in live:
            a.remove()
        live.clear()
        r = rows[i]
        if prev_shot["id"] is not None and r.shot_id != prev_shot["id"]:
            cut_left["n"] = CUT_BANNER_FRAMES
        prev_shot["id"] = r.shot_id
        clock = f"   match clock {r.match_clock}" if getattr(r, "match_clock", None) else ""
        header.set_text(f"{title}   t = {r.timestamp_s:6.1f} s   frame {r.frame_id}{clock}")
        cur = by_sample.get(r.sample_id)
        st = status.get(r.sample_id, "not_wide_view")
        calibrated = st in ("keyframe", "propagated")
        b = ball_by.get(r.sample_id)
        ball_sc.set_offsets(np.empty((0, 2)) if b is None else b[["x_s", "y_s"]].to_numpy())
        banner.set_text("CAMERA CUT" if cut_left["n"] > 0 else "")
        cut_left["n"] = max(0, cut_left["n"] - 1)
        border.set_visible(not calibrated)
        if not calibrated:
            note.set_text(f"NO CALIBRATED VIEW ({st.replace('_', ' ')}): no coordinates, nothing drawn")
            return []
        if cur is not None:
            live.extend(_trails(ax, observed, r.timestamp_s))
            arts, c = draw_players(ax, cur)
            live.extend(arts)
            note.set_text(
                f"observed {c['observed']} | interpolated {c['interpolated']} | predicted {c['predicted']} | "
                f"annotated {c['annotated']} | location unknown {c['unknown']}   (calibration: {st})"
            )
        else:
            note.set_text(f"no tracked people in this frame (calibration: {st})")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(rows), interval=1000 / fps, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264",
                                                          extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    log.info("State animation (%d frames) -> %s", len(rows), out_path)
    return out_path
