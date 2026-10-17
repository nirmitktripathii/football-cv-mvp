"""Command-line entry point: ``python -m football_cv.pipeline <stage> [--set key=value ...]``.

Stages run in order and each reads the previous stage's saved outputs from
``data/processed/<clip_id>/``, so any stage can be re-run on its own.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from football_cv import config as C
from football_cv.calibration import homography, motion, projection, shots
from football_cv.detection import detector
from football_cv.ingestion import video
from football_cv.tracking import teams, tracker
from football_cv.trajectories import clean

log = logging.getLogger("football_cv.pipeline")


def stage_ingest(cfg: dict) -> None:
    clip = cfg["clip"]
    src = C.resolve(clip["video_path"])
    meta = video.probe_video(src)
    log.info(
        "Video %s: %dx%d, %.3f fps (VFR=%s), %.2fs, %s frames",
        src.name, meta["width"], meta["height"], meta["fps"], meta["variable_frame_rate"],
        meta["duration_s"], meta["nb_frames"],
    )
    manifest = {
        "clip": clip,
        "video": meta,
        "sampling": cfg["sampling"],
        "clip_window_in_replay_s": [clip["source_offset_s"], clip["source_offset_s"] + meta["duration_s"]],
        "clip_window_match_clock": [
            video.match_clock(clip["source_offset_s"], clip["kickoff_replay_s"]),
            video.match_clock(clip["source_offset_s"] + meta["duration_s"], clip["kickoff_replay_s"]),
        ],
    }
    C.write_json(C.resolve(cfg["paths"]["manifests_dir"]) / f"{clip['id']}.json", manifest)

    out = C.run_dir(cfg)
    s = cfg["sampling"]
    idx = video.extract_samples(
        src, out, fps=float(s["fps"]), start_s=float(s["start_s"]), end_s=s["end_s"],
        replay_offset_s=float(clip["source_offset_s"]), kickoff_replay_s=clip.get("kickoff_replay_s"),
    )
    sheet = video.contact_sheet(out, idx, float(s["preview_every_s"]), C.artifacts_dir(cfg) / "preview_contact_sheet.jpg")
    log.info("Sampled %d frames; preview -> %s", len(idx), sheet)


def _frames(cfg: dict) -> pd.DataFrame:
    path = C.run_dir(cfg) / "frames.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the ingest stage first")
    return pd.read_parquet(path)


def stage_detect(cfg: dict) -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    d = cfg["detection"]
    out = C.run_dir(cfg)
    frames = _frames(cfg)
    weights = C.resolve(Path("models") / d["model"])
    weights.parent.mkdir(parents=True, exist_ok=True)
    attempt_download_asset(str(weights))  # official Ultralytics release asset; cached after first run
    dets = detector.detect_frames(
        frames, out, str(weights), imgsz=int(d["imgsz"]), conf_person=float(d["conf_person"]),
        conf_ball=float(d["conf_ball"]), device=d["device"], half=bool(d["half"]), batch=int(d["batch"]),
    )
    dets.to_parquet(out / "detections.parquet", index=False)
    summary = detector.detection_summary(dets, len(frames))
    C.write_json(out / "detection_summary.json", summary)
    log.info("Detections: %s", summary)


def _detections(cfg: dict) -> pd.DataFrame:
    path = C.run_dir(cfg) / "detections.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the detect stage first")
    return pd.read_parquet(path)


def stage_shots(cfg: dict) -> None:
    s = cfg["shots"]
    out = C.run_dir(cfg)
    frames = _frames(cfg)
    dets = _detections(cfg)
    shot_frames, dets = shots.classify_frames(
        frames, out, dets, min_green_ratio=float(s["min_green_ratio"]),
        max_median_person_height_px=float(s["max_median_person_height_px"]),
        min_persons=int(s["min_persons"]), cut_hist_threshold=float(s["cut_hist_threshold"]),
    )
    shot_frames.to_parquet(out / "shots.parquet", index=False)
    dets.to_parquet(out / "detections_on_pitch.parquet", index=False)
    table = shots.shot_table(shot_frames)
    table.to_csv(C.artifacts_dir(cfg) / "shot_table.csv", index=False)
    log.info(
        "Shots: %d shots, %d/%d frames wide view; %d/%d detections on pitch",
        len(table), int(shot_frames["wide_view"].sum()), len(shot_frames),
        int(dets["on_pitch"].sum()), len(dets),
    )


def _read(cfg: dict, name: str, stage: str) -> pd.DataFrame:
    path = C.run_dir(cfg) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the {stage} stage first")
    return pd.read_parquet(path)


def stage_motion(cfg: dict) -> None:
    c = cfg["calibration"]
    out = C.run_dir(cfg)
    dets = _read(cfg, "detections_on_pitch.parquet", "shots")
    persons = dets[(dets["object_class"] == "person") & dets["on_pitch"]]
    mot = motion.estimate_motion(
        _frames(cfg), out, _read(cfg, "shots.parquet", "shots"), persons,
        n_features=int(c["orb_features"]), min_inliers=int(c["min_inliers"]), max_chain_s=float(c["max_chain_s"]),
    )
    mot.to_parquet(out / "motion.parquet", index=False)
    segs = mot.groupby("segment_id").agg(
        shot_id=("shot_id", "first"), start_s=("timestamp_s", "min"), end_s=("timestamp_s", "max"),
        n_frames=("sample_id", "size"), median_inliers=("inliers", "median"),
    )
    segs.to_csv(C.artifacts_dir(cfg) / "motion_segments.csv")
    log.info("Motion: %d wide frames in %d segments", len(mot), len(segs))


def stage_track(cfg: dict) -> None:
    out = C.run_dir(cfg)
    frames = _frames(cfg)
    tr = tracker.track(
        _read(cfg, "detections_on_pitch.parquet", "shots"), _read(cfg, "motion.parquet", "motion"),
        cfg["tracking"], fps=float(cfg["sampling"]["fps"]),
    )
    t = cfg["teams"]
    colours = teams.sample_colours(tr, frames, out, min_box_h=float(t["min_box_height_px"]))
    table, diag = teams.assign_teams(
        colours, cfg["clip"]["home_kit"], cfg["clip"]["away_kit"], float(t["other_distance_factor"]),
    )
    tr = tr.merge(table[["track_id", "team_id"]], on="track_id", how="left")
    tr.loc[tr["object_class"] == "ball", "team_id"] = "ball"
    tr["team_id"] = tr["team_id"].fillna("unknown")
    tr.to_parquet(out / "tracks.parquet", index=False)
    table.to_csv(C.artifacts_dir(cfg) / "track_teams.csv", index=False)
    C.write_json(out / "team_diagnostics.json", diag)


def stage_calibrate(cfg: dict) -> None:
    import cv2

    c = cfg["calibration"]
    out = C.run_dir(cfg)
    art = C.artifacts_dir(cfg) / "calibration"
    art.mkdir(parents=True, exist_ok=True)
    frames = _frames(cfg)
    mot = _read(cfg, "motion.parquet", "motion")
    kfs = homography.load_keyframes(C.resolve(c["keyframes_file"]))
    kfs = projection.calibrate_keyframes(kfs, frames, out, mot, refine=bool(c.get("refine_with_lines", True)))
    paths = frames.set_index("sample_id")["path"]
    for kf in kfs:
        img = cv2.imread(str(out / paths[kf.sample_id]))
        for (u, v), name in zip(kf.image_pts, kf.names):
            cv2.circle(img, (int(u), int(v)), 6, (0, 255, 255), 2)
            cv2.putText(img, name, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if kf.H is not None:
            img = homography.draw_pitch_overlay(img, kf.H)
        cv2.imwrite(str(art / f"keyframe_{kf.sample_id:06d}_overlay.jpg"), img)
    drift = projection.drift_checks(kfs, mot, (720, 1280))
    calib = projection.frame_calibrations(kfs, mot, frames, out, float(c.get("max_keyframe_dt_s", float("inf"))))
    calib.to_parquet(out / "frame_calibration.parquet", index=False)

    tracks = _read(cfg, "tracks.parquet", "track")
    proj = projection.project_tracks(
        tracks, calib, float(c["pitch_margin_m"]), float(c["pitch_length_m"]), float(c["pitch_width_m"])
    )
    proj.to_parquet(out / "tracks_projected.parquet", index=False)
    t = cfg["trajectories"]
    traj = clean.clean_trajectories(proj, float(t["max_speed_mps"]), int(t["min_track_points"]))
    traj.to_parquet(out / "trajectories.parquet", index=False)
    clean.track_summary(traj).to_csv(C.artifacts_dir(cfg) / "track_summary.csv", index=False)

    report = {
        "keyframes": {str(k.sample_id): k.diagnostics for k in kfs},
        "drift_checks": drift,
        "frames_by_status": {str(k): int(v) for k, v in calib["calibration_status"].value_counts().items()},
        "track_rows_by_status": {str(k): int(v) for k, v in proj["calibration_status"].value_counts().items()},
        "trajectory_rows": len(traj), "jump_rows": int(traj["jump"].sum()),
    }
    C.write_json(out / "calibration_report.json", report)
    C.write_json(C.artifacts_dir(cfg) / "calibration_report.json", report)
    log.info("Calibration: %s", {k: v for k, v in report.items() if k != "keyframes"})


def stage_identity(cfg: dict) -> None:
    """Link track fragments of one person conservatively; writes the map and every decision (see identity.py)."""
    from football_cv.tracking import appearance, identity

    out, art = C.run_dir(cfg), C.artifacts_dir(cfg) / "identity"
    art.mkdir(parents=True, exist_ok=True)
    s = identity.settings(cfg)
    traj = _read(cfg, "trajectories.parquet", "calibrate")
    people = traj[(traj["track_id"] >= 0) & (traj["object_class"] == "person")].copy()
    status = pd.read_csv(C.artifacts_dir(cfg) / "track_teams.csv")[["track_id", "team_status"]]
    people = people.merge(status, on="track_id", how="left")
    people["team_status"] = people["team_status"].fillna("unknown")
    w, h = (cfg.get("reconstruction") or {}).get("image_size", [1280, 720])
    m = float(s.get("edge_margin_px", 20.0))
    people["edge"] = (people["bbox_x1"] < m) | (people["bbox_x2"] > w - m) | (people["bbox_y1"] < m) | (people["bbox_y2"] > h - m)
    desc = None
    if s.get("use_appearance", True):
        samples = appearance.sample_bands(_read(cfg, "tracks.parquet", "track"), _frames(cfg), out,
                                          min_box_h=float(cfg["teams"]["min_box_height_px"]))
        desc = appearance.describe_tracks(samples)
        desc.to_csv(art / "appearance.csv", index=False)
    ident, decisions, tracklets = identity.link_tracklets(people, cfg, desc)
    ident.to_parquet(out / "identity_map.parquet", index=False)
    decisions.to_csv(art / "link_decisions.csv", index=False)
    tracklets.to_csv(art / "tracklets.csv", index=False)
    chains = ident.groupby("identity_id").size()
    summary = {
        "tracklets": len(tracklets), "candidate_pairs": len(decisions),
        "by_status": {k: int(v) for k, v in decisions["status"].value_counts().items()} if len(decisions) else {},
        "identities": int(ident["identity_id"].nunique()), "tracks": len(ident),
        "merged_chains": int((chains > 1).sum()), "applied_to_reconstruction": bool(s.get("apply", False)),
        "appearance_descriptors": 0 if desc is None else len(desc), "settings": s,
    }
    C.write_json(art / "identity_summary.json", summary)
    log.info("Identity: %s", {k: v for k, v in summary.items() if k != "settings"})


def stage_reconstruct(cfg: dict) -> None:
    from football_cv.reconstruction import reconstruct
    from football_cv.tracking import identity

    out, art = C.run_dir(cfg), C.artifacts_dir(cfg) / "reconstruction"
    art.mkdir(parents=True, exist_ok=True)
    traj = _read(cfg, "trajectories.parquet", "calibrate")
    timeline = _read(cfg, "shots.parquet", "shots")[["sample_id", "timestamp_s", "shot_id"]]
    calib = _read(cfg, "frame_calibration.parquet", "calibrate")
    ann_file = (cfg.get("reconstruction") or {}).get("annotations_file")
    notes = reconstruct.load_annotations(C.resolve(ann_file) if ann_file else None)
    map_path = out / "identity_map.parquet"
    if identity.settings(cfg).get("apply") and map_path.exists():
        traj = traj.merge(pd.read_parquet(map_path), on="track_id", how="left")
        traj["identity_id"] = traj["identity_id"].fillna(traj["track_id"]).astype(int)
    states, gaps, warnings = reconstruct.reconstruct(traj, timeline, calib, cfg, notes)
    states.to_parquet(out / "player_states.parquet", index=False)
    gaps.to_parquet(out / "gaps.parquet", index=False)
    summary = {
        "identities": int(states["identity_id"].nunique()), "rows": len(states),
        "rows_by_source": {k or "none": int(v) for k, v in states["position_source"].value_counts().items()},
        "rows_by_state": {str(k): int(v) for k, v in states["state"].value_counts().items()},
        "gaps": len(gaps), "annotations": len(notes), "annotation_warnings": warnings,
        "settings": reconstruct.settings(cfg),
    }
    C.write_json(art / "reconstruction_summary.json", summary)
    log.info("Reconstruction: %s", {k: v for k, v in summary.items() if k != "settings"})


def stage_regions(cfg: dict) -> None:
    """Inferred probable regions for unobserved players, written apart from player_states."""
    from football_cv.reconstruction import regions

    out, art = C.run_dir(cfg), C.artifacts_dir(cfg) / "regions"
    art.mkdir(parents=True, exist_ok=True)
    states = _read(cfg, "player_states.parquet", "reconstruct")
    timeline = _read(cfg, "shots.parquet", "shots")[["sample_id", "timestamp_s", "shot_id"]]
    s = regions.settings(cfg)
    extended = regions.extend_states(states, timeline, float(s["horizon_s"]))
    reg = regions.mark_plausible_unseen(regions.infer_regions(extended, cfg), extended, cfg)
    reg.to_parquet(out / "inferred_regions.parquet", index=False)
    summary = {
        "regions": len(reg), "plausible_unseen": int(reg["plausible_unseen"].sum()),
        "identities": int(reg["identity_id"].nunique()),
        "by_components": {k: int(v) for k, v in reg["components"].value_counts().items()},
        "median_area_m2": float(reg["area_m2"].median()) if len(reg) else None, "settings": s,
    }
    C.write_json(art / "regions_summary.json", summary)
    log.info("Regions: %s", {k: v for k, v in summary.items() if k != "settings"})


def stage_reconstruction_eval(cfg: dict) -> None:
    from football_cv.reconstruction import evaluate

    traj = _read(cfg, "trajectories.parquet", "calibrate")
    good = traj[(traj["track_id"] >= 0) & ~traj["jump"] & traj["x_s"].notna() & ~traj["short_track"]]
    gaps = (cfg.get("reconstruction") or {}).get("eval_gap_lengths_s", [0.6, 1.0, 2.0, 4.0, 8.0])
    res = evaluate.masking_eval(good, tuple(gaps), 1.0 / float(cfg["sampling"]["fps"]))
    table = evaluate.summarise(res)
    art = C.artifacts_dir(cfg) / "reconstruction"
    art.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(art / "masking_eval.csv", index=False)
    log.info("Masking eval (%d scored samples):\n%s", len(res), table.round(2).to_string())


def _home_attacks(cfg: dict) -> str:
    side = cfg["clip"].get("home_attacks")
    if side not in ("left", "right"):
        raise ValueError("set clip.home_attacks to 'left' or 'right' (direction of play in this clip)")
    return side


def stage_analytics(cfg: dict) -> None:
    from football_cv.analytics import spatial, xt
    from football_cv.sequences import candidates

    out, art = C.run_dir(cfg), C.artifacts_dir(cfg) / "analytics"
    art.mkdir(parents=True, exist_ok=True)
    traj = _read(cfg, "trajectories.parquet", "calibrate")
    calib = _read(cfg, "frame_calibration.parquet", "calibrate")
    side = _home_attacks(cfg)
    grid = xt.load_grid(C.resolve(cfg["analytics"]["xt_model_path"]))

    ball_xt = xt.ball_threat(traj, grid, side)
    shape = spatial.team_shape(traj)
    prox = spatial.ball_proximity(traj)
    seq = candidates.find_candidates(ball_xt, prox, calib, side, int(cfg["analytics"].get("top_sequences", 5)))
    for name, df in (("ball_xt", ball_xt), ("team_shape", shape), ("ball_proximity", prox), ("sequences", seq)):
        df.to_parquet(out / f"{name}.parquet", index=False)
        df.to_csv(art / f"{name}.csv", index=False)
    spatial.shape_summary(shape).to_csv(art / "team_shape_summary.csv", index=False)

    details = []
    for r in seq.itertuples():
        in_win = shape[(shape["timestamp_s"] >= r.start_s) & (shape["timestamp_s"] <= r.end_s)]
        details.append({
            "seq_id": r.seq_id, "start_s": r.start_s, "end_s": r.end_s,
            "xt": xt.window_summary(ball_xt, r.start_s, r.end_s),
            "shape_median": spatial.shape_summary(in_win).to_dict(orient="records"),
            "confidence": r.confidence,
        })
    summary = {
        "home_attacks": side, "xt_label": xt.LABEL,
        "ball_observations_calibrated": len(ball_xt),
        "shape_frames": int(shape["sample_id"].nunique()), "proximity_rows": len(prox),
        "sequences": details,
        "not_computed": {
            "ppda": "requires defensive actions and passes (event data); not derivable from positions",
            "atomic_vaep_on_clip": "INCOMPLETE: no valid action stream; see artifacts/atomic_vaep for the public-data model",
        },
    }
    C.write_json(art / "analytics_summary.json", summary)
    log.info("Analytics: %d ball rows, %d shape rows, %d candidate sequences", len(ball_xt), len(shape), len(seq))


def stage_visualize(cfg: dict) -> None:
    from football_cv.visualization import broadcast, pitch_animation, pitch_states

    out, art = C.run_dir(cfg), C.artifacts_dir(cfg) / "videos"
    v = cfg.get("visualize", {})
    frames = _frames(cfg)
    calib = _read(cfg, "frame_calibration.parquet", "calibrate")
    traj = _read(cfg, "trajectories.parquet", "calibrate")
    tracks = _read(cfg, "tracks.parquet", "track")
    keys = ["sample_id", "track_id"]
    pos = traj[keys + ["x_s", "y_s", "jump", "short_track"]]
    traj_all = tracks.merge(pos, on=keys, how="left")
    hide = traj_all["jump"].fillna(False).astype(bool) | traj_all["short_track"].fillna(False).astype(bool)
    traj_all.loc[hide, ["x_s", "y_s"]] = float("nan")
    names = {"home": cfg["clip"]["home_team"], "away": cfg["clip"]["away_team"]}
    fps = float(cfg["sampling"]["fps"])

    windows = [("full", None, None)]
    seq_path = out / "sequences.parquet"
    if seq_path.exists():
        windows += [(r.seq_id, r.start_s, r.end_s) for r in pd.read_parquet(seq_path).itertuples()]
    states_path = out / "player_states.parquet"
    states = pd.read_parquet(states_path) if states_path.exists() else None
    shots_tbl = _read(cfg, "shots.parquet", "shots") if states is not None else None
    reg_path = out / "inferred_regions.parquet"
    regions_tbl = pd.read_parquet(reg_path) if reg_path.exists() else None
    for name, start, end in windows:
        if states is not None and (name != "full" or v.get("full_states", False)):
            pitch_states.render_state_animation(
                states, traj, frames, shots_tbl, calib, art / f"pitch_states_{name}.gif" if name != "full"
                else art / "pitch_states_full.mp4", start, end, names, fps,
                title=f"{names['home']} v {names['away']}" + ("" if name == "full" else f"  {name}"),
                regions=regions_tbl,
            )
        if name != "full" or v.get("full_broadcast", True):
            broadcast.render_broadcast_video(traj_all, frames, out, calib, art / f"broadcast_{name}.mp4", start, end, fps)
        suffix = ".mp4" if name == "full" else ".gif"
        pitch_animation.render_pitch_animation(
            traj, frames, calib, art / f"pitch_{name}{suffix}", start, end, names, fps,
            title=f"{names['home']} v {names['away']}" + ("" if name == "full" else f"  {name}"),
        )


def stage_atomic_vaep(cfg: dict) -> None:
    from football_cv.analytics import atomic_vaep

    a = cfg["analytics"]["atomic_vaep"]
    atomic_vaep.train_and_evaluate(
        C.resolve(a["spadl_dir"]), list(a["train_stores"]), a["test_store"],
        C.resolve(cfg["paths"]["artifacts_dir"]) / "atomic_vaep",
        max_games_per_store=a.get("max_games_per_store"), seed=int(a.get("seed", 0)),
    )


STAGES = {
    "ingest": stage_ingest,
    "detect": stage_detect,
    "shots": stage_shots,
    "motion": stage_motion,
    "track": stage_track,
    "calibrate": stage_calibrate,
    "identity": stage_identity,
    "reconstruct": stage_reconstruct,
    "regions": stage_regions,
    "analytics": stage_analytics,
    "visualize": stage_visualize,
}
# Not clip-specific and slow (~10 min); run explicitly, never as part of "all".
EXTRA_STAGES = {"atomic_vaep": stage_atomic_vaep, "reconstruction_eval": stage_reconstruction_eval}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, *EXTRA_STAGES, "all"])
    parser.add_argument("--config", default=str(C.DEFAULT_CONFIG))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = C.load_config(args.config, args.overrides)
    C.setup_logging(C.run_dir(cfg) / "logs" / "pipeline.log")
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        t0 = time.perf_counter()
        log.info("=== stage %s ===", name)
        {**STAGES, **EXTRA_STAGES}[name](cfg)
        log.info("=== stage %s done in %.1fs ===", name, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
