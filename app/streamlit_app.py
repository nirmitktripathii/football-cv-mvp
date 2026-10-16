"""Streamlit dashboard over the saved pipeline outputs (read-only).

    streamlit run app/streamlit_app.py

Every number shown here is read from ``artifacts/`` or ``data/processed/``;
nothing is recomputed or invented in the UI. Run the pipeline stages first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from football_cv import config as C
from football_cv.visualization.pitch_animation import plot_snapshot

st.set_page_config(page_title="Football CV MVP", layout="wide")

cfg = C.load_config(C.DEFAULT_CONFIG, [])
ART, RUN = C.artifacts_dir(cfg), C.run_dir(cfg)
VAEP_DIR = C.resolve(cfg["paths"]["artifacts_dir"]) / "atomic_vaep"
NAMES = {"home": cfg["clip"]["home_team"], "away": cfg["clip"]["away_team"]}


@st.cache_data
def read_csv(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if Path(path).exists() else None


@st.cache_data
def read_parquet(path: str) -> pd.DataFrame | None:
    return pd.read_parquet(path) if Path(path).exists() else None


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def missing(what: str, stage: str) -> None:
    st.info(f"{what} not found. Run `python -m football_cv.pipeline {stage}` first.")


def video(path: Path, caption: str) -> None:
    if not path.exists():
        missing(path.name, "visualize")
        return
    if path.suffix == ".gif":
        st.image(str(path), caption=caption)
    else:
        st.video(str(path))
        st.caption(caption)


clip = cfg["clip"]
st.title(f"{clip['home_team']} v {clip['away_team']}: broadcast CV analytics MVP")
st.caption(
    f"Source: {clip['source_title']} ({clip['source_channel']}, {clip['source_upload_date']}), "
    f"section from replay second {clip['source_offset_s']}. Every output below is derived from this one "
    "broadcast clip; nothing has been checked against ground truth."
)

calib_report = read_json(ART / "calibration_report.json")
analytics = read_json(ART / "analytics" / "analytics_summary.json")
tabs = st.tabs(["Overview", "Videos", "Calibration", "Sequences", "Team shape & xT", "Players", "Atomic-VAEP"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    if calib_report:
        fr = calib_report["frames_by_status"]
        calibrated = sum(v for k, v in fr.items() if k in ("propagated", "keyframe"))
        c1.metric("Calibrated wide-view frames", calibrated, help=json.dumps(fr, indent=1))
        c2.metric("Trajectory rows", calib_report["trajectory_rows"])
        c3.metric("Manual keyframes", len(calib_report["keyframes"]))
        st.markdown("**Frames by calibration status**")
        st.json(fr, expanded=False)
    if analytics:
        c4.metric("Candidate sequences", len(analytics["sequences"]))
        st.markdown(f"**Direction of play:** {NAMES['home']} attack **{analytics['home_attacks']}** in this clip "
                    "(measured from which team's deepest player sits at each goal; see the config).")
        st.markdown(f"**xT here is** _{analytics['xt_label']}_.")
        st.markdown("**Not computed:**")
        for k, why in analytics["not_computed"].items():
            st.markdown(f"- `{k}`: {why}")
    shots = read_csv(str(ART / "shot_table.csv"))
    if shots is not None:
        st.subheader("Shots (camera cuts) and wide-view share")
        st.dataframe(shots, width="stretch", hide_index=True)

with tabs[1]:
    st.markdown("Boxes, track ids and team colours are drawn only where detected; the top-down animation "
                "shows only observed, calibrated positions (no interpolation).")
    video(ART / "videos" / "broadcast_full.mp4", "Annotated broadcast (5 fps samples)")
    video(ART / "videos" / "pitch_full.mp4", "Top-down pitch animation")
    traj = read_parquet(str(RUN / "trajectories.parquet"))
    if traj is not None and len(traj):
        sids = sorted(traj["sample_id"].unique())
        sid = st.select_slider("Snapshot of one sampled frame", options=sids, value=sids[len(sids) // 2])
        fig, _ = plot_snapshot(traj, sid, team_names=NAMES)
        st.pyplot(fig, width="content")

with tabs[2]:
    if not calib_report:
        missing("Calibration report", "calibrate")
    else:
        st.markdown("Manual landmark keyframes, propagated along the camera-motion chain of each segment. "
                    "`landmark_*` drift is measured at the next keyframe's own clicked landmarks.")
        st.dataframe(pd.DataFrame(calib_report["drift_checks"]), width="stretch", hide_index=True)
        kf = pd.DataFrame.from_dict(calib_report["keyframes"], orient="index")
        cols = [c for c in ["segment_id", "n_points", "n_line_points", "manual_rms_m", "refined", "valid", "validation"]
                if c in kf]
        st.dataframe(kf[cols], width="stretch")
        pick = st.selectbox("Keyframe overlay (projected pitch model)", list(calib_report["keyframes"]))
        overlay = ART / "calibration" / f"keyframe_{int(pick):06d}_overlay.jpg"
        if overlay.exists():
            st.image(str(overlay))
        st.markdown("**Track rows by calibration status**")
        st.json(calib_report["track_rows_by_status"], expanded=False)

with tabs[3]:
    seq = read_csv(str(ART / "analytics" / "sequences.csv"))
    if seq is None or not analytics:
        missing("Sequences", "analytics")
    elif seq.empty:
        st.warning("No candidate windows met the criteria.")
    else:
        st.markdown("Windows where the observed ball reaches a final third. **Candidates for review, not detected "
                    "events.** The attacking team follows from the direction of play; the nearest-player share "
                    "is a proxy check, not possession.")
        st.dataframe(seq, width="stretch", hide_index=True)
        detail = {d["seq_id"]: d for d in analytics["sequences"]}
        sid = st.selectbox("Sequence", seq["seq_id"].tolist())
        row = seq.set_index("seq_id").loc[sid]
        team = NAMES.get(row["probable_attacking_team"], row["probable_attacking_team"])
        st.markdown(f"**{sid}**: t {row['start_s']}-{row['end_s']} s (samples {row['first_sample']}-"
                    f"{row['last_sample']}), probable attack by **{team}** towards the **{row['attacking_direction']}**"
                    f" goal. Ball x {row['ball_x_start']} -> {row['ball_x_end']} m.")
        st.markdown(f"Confidence: {row['confidence']}")
        a, b = st.columns(2)
        with a:
            video(ART / "videos" / f"pitch_{sid}.gif", "Top-down, observed positions only")
        with b:
            video(ART / "videos" / f"broadcast_{sid}.mp4", "Broadcast with detections")
        st.json(detail.get(sid, {}), expanded=False)

with tabs[4]:
    shape = read_csv(str(ART / "analytics" / "team_shape.csv"))
    ball_xt = read_csv(str(ART / "analytics" / "ball_xt.csv"))
    if shape is None:
        missing("Team shape", "analytics")
    else:
        st.markdown("Shape of the **visible** players of each team (a broadcast shows part of the pitch), frames with "
                    "at least 4 visible players.")
        summary = read_csv(str(ART / "analytics" / "team_shape_summary.csv"))
        if summary is not None:
            st.dataframe(summary, hide_index=True)
        metric = st.selectbox("Metric", ["width_m", "depth_m", "hull_area_m2", "mean_nn_dist_m", "centroid_x_m"])
        fig = px.scatter(shape.assign(team=shape["team_id"].map(NAMES)), x="timestamp_s", y=metric, color="team",
                         opacity=0.6, height=360)
        st.plotly_chart(fig, width="stretch")
    if ball_xt is not None and len(ball_xt):
        st.markdown(f"**Ball-zone xT** ({analytics['xt_label'] if analytics else 'exploratory'})")
        long = ball_xt.melt(id_vars="timestamp_s", value_vars=["xt_home_attacking", "xt_away_attacking"],
                            var_name="perspective", value_name="zone_xt")
        long["perspective"] = long["perspective"].map(
            {"xt_home_attacking": f"{NAMES['home']} attacking", "xt_away_attacking": f"{NAMES['away']} attacking"})
        st.plotly_chart(px.scatter(long, x="timestamp_s", y="zone_xt", color="perspective", height=320),
                        width="stretch")

with tabs[5]:
    tracks = read_csv(str(ART / "track_summary.csv"))
    if tracks is None:
        missing("Track summary", "calibrate")
    else:
        st.markdown("Per **track** (not per player: identities break at occlusions and camera cuts). Distance and "
                    "speed use only consecutive observations of one run.")
        teams = st.multiselect("Teams", sorted(tracks["team_id"].dropna().unique()), default=["home", "away"])
        view = tracks[tracks["team_id"].isin(teams)].sort_values("distance_m", ascending=False)
        st.dataframe(view, width="stretch", hide_index=True)
        st.plotly_chart(px.histogram(view, x="p95_speed_mps", color="team_id", nbins=30, height=300),
                        width="stretch")

with tabs[6]:
    metrics = read_json(VAEP_DIR / "atomic_vaep_metrics.json")
    if not metrics:
        missing("Atomic-VAEP metrics", "atomic_vaep")
    else:
        st.error(f"On this clip: {metrics['cv_clip_status']}")
        st.markdown(f"Model: `socceraction.atomic.vaep.AtomicVAEP` ({metrics['learner']}, {metrics['n_features']} "
                    f"features), trained on {metrics['train_games']} games ({', '.join(metrics['train_stores'])}), "
                    f"evaluated on {metrics['test_games']} held-out games ({metrics['test_store']}).")
        rows = [{"label": k, **sc, "base_rate": metrics["test_base_rate"][k],
                 "brier_of_base_rate": metrics["test_brier_of_base_rate"][k]}
                for k, sc in metrics["test_scores"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        top = read_csv(str(VAEP_DIR / "atomic_vaep_integration_top_actions.csv"))
        if top is not None:
            st.markdown("Integration test: highest-valued atomic actions of one held-out game (rated end to end).")
            st.dataframe(top, width="stretch", hide_index=True)
