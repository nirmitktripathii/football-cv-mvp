"""Per-match configs that extend the default config."""
from football_cv.config import REPO_ROOT, deep_merge, load_config


def test_deep_merge_merges_nested_dicts_and_leaves_inputs_alone():
    base = {"a": {"x": 1, "y": 2}, "b": [1]}
    top = {"a": {"y": 3}, "b": [2], "c": 4}
    assert deep_merge(base, top) == {"a": {"x": 1, "y": 3}, "b": [2], "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1]}


def test_extends_is_resolved_relative_to_the_file_and_overrides_still_win(tmp_path):
    (tmp_path / "base.yaml").write_text("clip: {id: base, fps: 25}\nsampling: {fps: 5}\n", encoding="utf-8")
    (tmp_path / "match.yaml").write_text("extends: base.yaml\nclip: {id: match}\n", encoding="utf-8")
    cfg = load_config(tmp_path / "match.yaml", ["sampling.fps=10"])
    assert cfg == {"clip": {"id": "match", "fps": 25}, "sampling": {"fps": 10}}


def test_inter_napoli_config_loads_on_top_of_default():
    cfg = load_config(REPO_ROOT / "configs" / "inter_napoli.yaml")
    assert cfg["clip"]["home_team"] == "Inter" and "sampling" in cfg
    windows = set(cfg["evaluation"]["windows"])
    assert {"w1_open_t600", "w2_corner_t2695", "w3_open_t4800"} <= windows
    assert {"a1_open_t600", "a2_corner_t2707", "a3_open_t4800"} <= windows
