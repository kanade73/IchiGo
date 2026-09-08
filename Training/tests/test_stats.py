import random

from ichigo_train.stats import (
    build_report,
    bootstrap_mean_ci,
    elo_from_score,
    game_score,
    incident_counts,
    outcome_for,
    pair_scores,
    tally,
    _percentile,
)


def make_game(pair_index, black, result, counted=True, incidents=None):
    white = "B" if black == "A" else "A"
    return {
        "pairIndex": pair_index, "colours": {"black": black, "white": white},
        "result": result, "countedResult": counted, "incidents": incidents or [],
    }


# -- hand-computed win/draw/loss tally and pair-score fixture (docs/spec/04-tasks.md T31) --------

def test_outcome_for_and_game_score():
    assert outcome_for("B", "B+5") == "win" and game_score("B", "B+5") == 1.0
    assert outcome_for("W", "B+5") == "loss" and game_score("W", "B+5") == 0.0
    assert outcome_for("B", "0") == "draw" and game_score("B", "0") == 0.5
    assert outcome_for("B", "truncated") is None and game_score("B", "truncated") is None
    assert outcome_for("B", None) is None


def test_hand_computed_tally_and_pair_scores():
    games = [
        make_game(0, "A", "B+5"),          # A black, black wins -> A win  (1.0)
        make_game(0, "B", "W+5"),          # B black (swap), white wins -> A is white -> A win (1.0)
        make_game(1, "A", "0"),            # draw -> 0.5
        make_game(1, "B", "0"),            # draw -> 0.5
        make_game(2, "A", "W+2"),          # A black, white wins -> A loss (0.0)
        make_game(2, "B", "B+2"),          # B black wins -> A white -> A loss (0.0)
        make_game(3, "A", None, counted=False),   # truncated: excluded entirely
        make_game(3, "B", "0"),                   # counted alone: no swapped partner -> excluded from pairing
    ]
    t = tally(games, "A")
    assert t["games"] == 7
    assert (t["wins"], t["draws"], t["losses"]) == (2, 3, 2)
    expected_mean = (1 + 1 + 0.5 + 0.5 + 0 + 0 + 0.5) / 7
    assert abs(t["meanScore"] - expected_mean) < 1e-12

    pairs = pair_scores(games, "A")
    assert pairs == [1.0, 0.5, 0.0]  # pair 3 has only one counted game -> dropped


def test_tally_empty_games_reports_none_mean():
    t = tally([], "A")
    assert t == {"games": 0, "wins": 0, "draws": 0, "losses": 0, "meanScore": None, "scores": []}
    assert pair_scores([], "A") == []


def test_incident_counts():
    games = [
        make_game(0, "A", "timeout", counted=False, incidents=[{"type": "timeout", "engine": "A"}]),
        make_game(0, "B", "illegal_move", counted=False, incidents=[{"type": "illegal_move", "engine": "B"}]),
        make_game(1, "A", "crash", counted=False, incidents=[{"type": "crash", "engine": "A"}]),
        make_game(1, "B", "truncated", counted=False),
        make_game(2, "A", "B+1"),
    ]
    counts = incident_counts(games)
    assert counts == {"timeouts": 1, "illegalMoves": 1, "crashes": 1, "truncations": 1}


# -- bootstrap CI reproducibility with a fixed seed -----------------------------------------------

def test_bootstrap_ci_reproducible_with_fixed_seed():
    # A finely-grained value set (not just {0, 0.5, 1.0}) so two different seeds' resampled-mean
    # distributions are very unlikely to land on the exact same quantised percentile by chance.
    values = [0.13, 0.41, 0.72, 0.05, 0.88, 0.33, 0.61, 0.24, 0.95, 0.09, 0.47, 0.66, 0.02, 0.79, 0.38]
    r1 = bootstrap_mean_ci(values, resamples=500, seed=42)
    r2 = bootstrap_mean_ci(values, resamples=500, seed=42)
    assert r1 == r2
    r3 = bootstrap_mean_ci(values, resamples=500, seed=43)
    assert r3["low"] != r1["low"] or r3["high"] != r1["high"]


def test_bootstrap_ci_matches_manual_rng_replay():
    values = [1.0, 0.5, 0.0, 1.0, 0.5]
    seed, resamples = 7, 300
    result = bootstrap_mean_ci(values, resamples=resamples, seed=seed)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    assert result["low"] == _percentile(means, 0.025)
    assert result["high"] == _percentile(means, 0.975)
    assert result["mean"] == sum(values) / n
    assert result["low"] <= result["mean"] <= result["high"]


def test_bootstrap_ci_empty_input():
    r = bootstrap_mean_ci([], resamples=100, seed=1)
    assert r["n"] == 0 and r["mean"] is None and r["low"] is None and r["high"] is None


# -- Elo p=0/1 guard --------------------------------------------------------------------------

def test_elo_guard_undefined_at_boundaries():
    assert elo_from_score(1.0) == "undefined"
    assert elo_from_score(0.0) == "undefined"
    assert elo_from_score(None) == "undefined"


def test_elo_point_estimate_is_finite_and_symmetric():
    e = elo_from_score(0.75)
    assert isinstance(e, float) and e > 0
    assert abs(elo_from_score(0.25) + e) < 1e-9
    assert elo_from_score(0.5) == 0.0


# -- report assembly ----------------------------------------------------------------------------

def test_build_report_contains_required_fields():
    games = [
        make_game(0, "A", "B+5"), make_game(0, "B", "W+5"),
        make_game(1, "A", "0"), make_game(1, "B", "0"),
    ]
    report = build_report(
        games, candidate="A", seed=99, engines={"A": {"kind": "gtp", "argv": ["x"]}, "B": {"kind": "uniform", "argv": None}},
        settings={"size": 9, "komi": 7.0}, openings_file_hash="none", resamples=200,
    )
    for key in ("games", "wins", "draws", "losses", "meanScore", "pairedCI95", "seed", "engines", "settings", "openingsFileHash", "incidents", "unpairedCI95", "eloEstimate", "countedGames"):
        assert key in report
    # pair 0 (A black B+5, B black W+5): A wins both colours -> 2 wins; pair 1 (both "0"): 2 draws.
    assert report["wins"] == 2 and report["draws"] == 2 and report["losses"] == 0
    assert report["meanScore"] == 0.75
    assert report["seed"] == 99
    assert report["pairedCI95"]["n"] == 2
