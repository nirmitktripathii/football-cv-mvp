"""assign_teams: kit clusters come from individual crops, so tracks that mix both kits cannot drag the centres."""
import numpy as np
import pandas as pd

from football_cv.tracking.teams import assign_teams

DARK, RED, BRIGHT = (89.0, 133.0, 122.0), (115.0, 155.0, 146.0), (220.0, 128.0, 135.0)


def _samples(tid, colours, rng):
    return [{"sample_id": i, "track_id": tid, "L": c[0] + rng.normal(0, 3), "a": c[1] + rng.normal(0, 2),
             "b": c[2] + rng.normal(0, 2)} for i, c in enumerate(colours)]


def _clip(n_mixed):
    rng = np.random.default_rng(0)
    rows = []
    for t in range(6):
        rows += _samples(t, [DARK] * 20, rng)                  # home
    for t in range(6, 12):
        rows += _samples(t, [RED] * 20, rng)                   # away
    for t in range(12, 12 + n_mixed):                          # identity switches between opponents
        rows += _samples(t, [DARK] * 10 + [RED] * 10, rng)
    for t in range(100, 103):                                  # a few bright crops (referee, glare)
        rows += _samples(t, [BRIGHT] * 4, rng)
    return pd.DataFrame(rows)


def test_pure_tracks_get_their_kit():
    table, _ = assign_teams(_clip(0), "black", "red", 2.2)
    team = table.set_index("track_id")["team_id"]
    assert (team.loc[0:5] == "home").all() and (team.loc[6:11] == "away").all()


def test_many_mixed_tracks_do_not_collapse_the_clusters():
    table, diag = assign_teams(_clip(20), "black", "red", 2.2)
    team = table.set_index("track_id")["team_id"]
    assert (team.loc[0:5] == "home").all() and (team.loc[6:11] == "away").all()
    assert np.allclose(diag["centroids_lab"]["home"], DARK, atol=6)
    assert np.allclose(diag["centroids_lab"]["away"], RED, atol=6)


def test_mixed_tracks_are_not_confident():
    table, _ = assign_teams(_clip(20), "black", "red", 2.2)
    mixed = table[table["track_id"].between(12, 31)]
    assert (mixed["team_status"] != "confident").all()
