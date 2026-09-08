"""Match statistics (docs/spec/05-validation.md SS5, SS6; docs/spec/04-tasks.md T31).

Turns match.py's ``games.jsonl`` rows into the report.json fields: win/draw/loss tallies for the
candidate engine ("A"), paired and unpaired game-score bootstrap confidence intervals, a guarded
Elo point estimate, and incident counts. Only games with a truthy ``countedResult`` contribute to
any win/draw/loss/score statistic -- truncated games, aborted (illegal-move/timeout/crash) games
are excluded from win/draw/loss the same way docs/spec/05-validation.md SS6 excludes truncated
self-play games from being used as a win/loss target.

A game's ``result`` is either a GTP-style area-scoring string ("B+3.5", "W+0.5", "0" for a jigo)
recorded from the black engine's ``final_score`` (match.py's convention), or a non-counted marker
("truncated", "illegal_move", "timeout", "crash") that carries no score.
"""

from __future__ import annotations

import math
import random

SCORE_FOR_OUTCOME = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def outcome_for(candidate_colour: str, result: str | None) -> str | None:
    """``candidate_colour`` is "B" or "W" (which colour the candidate, engine "A", played this
    game). ``result`` is a score string or a falsy/non-counted marker. Returns "win"/"draw"/"loss"
    from the candidate's point of view, or None if the game does not count."""
    if not result or result[0] not in ("B", "W", "0"):
        return None
    if result == "0":
        return "draw"
    winner = result[0]
    return "win" if winner == candidate_colour else "loss"


def game_score(candidate_colour: str, result: str | None) -> float | None:
    outcome = outcome_for(candidate_colour, result)
    return None if outcome is None else SCORE_FOR_OUTCOME[outcome]


def _candidate_colour(game: dict, candidate: str) -> str:
    return "B" if game["colours"]["black"] == candidate else "W"


def tally(games: list[dict], candidate: str = "A") -> dict:
    """Win/draw/loss counts and the per-game score list for ``candidate`` ("A" or "B"), over
    counted games only."""
    wins = draws = losses = 0
    scores: list[float] = []
    for g in games:
        if not g.get("countedResult"):
            continue
        s = game_score(_candidate_colour(g, candidate), g.get("result"))
        if s is None:
            continue
        scores.append(s)
        if s == 1.0:
            wins += 1
        elif s == 0.5:
            draws += 1
        else:
            losses += 1
    n = len(scores)
    mean = sum(scores) / n if n else None
    return {"games": n, "wins": wins, "draws": draws, "losses": losses, "meanScore": mean, "scores": scores}


def pair_scores(games: list[dict], candidate: str = "A") -> list[float]:
    """One averaged score per colour-swapped opening pair (docs/spec/05-validation.md SS5
    "candidate expected scoreをペア単位で平均"). A pair contributes only when *both* of its games
    are counted -- a lone counted game with no swapped partner has no pair average to join, and
    mixing an unpaired single into the paired statistic would defeat the point of pairing (it
    exists to cancel opening-side bias by averaging the two colours of the same opening)."""
    by_pair: dict[int, list[float]] = {}
    for g in games:
        if not g.get("countedResult"):
            continue
        s = game_score(_candidate_colour(g, candidate), g.get("result"))
        if s is None:
            continue
        by_pair.setdefault(g["pairIndex"], []).append(s)
    return [sum(vals) / 2 for _, vals in sorted(by_pair.items()) if len(vals) == 2]


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile, same convention as numpy.percentile's default."""
    idx = q * (len(sorted_values) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_mean_ci(values: list[float], resamples: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """Percentile bootstrap CI of the mean of ``values``, reproducible for a fixed ``seed``
    (docs/spec/05-validation.md SS5/SS6 "paired bootstrap10000回、seed固定のpercentile95%CI")."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "low": None, "high": None, "resamples": resamples, "seed": seed, "ci": ci}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {
        "n": n, "mean": sum(values) / n, "low": _percentile(means, lo_q), "high": _percentile(means, hi_q),
        "resamples": resamples, "seed": seed, "ci": ci,
    }


def elo_from_score(mean_score: float | None) -> float | str:
    """Point-estimate Elo difference from an expected score in (0,1). Never returns +/-inf:
    p=0 or p=1 (or no counted games) reports the string "undefined" instead
    (docs/spec/05-validation.md SS5 "p=0/1時の無限Eloを出さず...Elo換算は参考")."""
    if mean_score is None or not (0.0 < mean_score < 1.0):
        return "undefined"
    return 400.0 * math.log10(mean_score / (1.0 - mean_score))


def incident_counts(games: list[dict]) -> dict:
    counts = {"timeouts": 0, "illegalMoves": 0, "crashes": 0, "truncations": 0}
    kind_key = {"timeout": "timeouts", "illegal_move": "illegalMoves", "crash": "crashes"}
    for g in games:
        for inc in g.get("incidents", []):
            key = kind_key.get(inc.get("type"))
            if key:
                counts[key] += 1
        if g.get("result") == "truncated":
            counts["truncations"] += 1
    return counts


def build_report(
    games: list[dict], *, candidate: str, seed: int, engines: dict, settings: dict,
    openings_file_hash: str, resamples: int = 10000,
) -> dict:
    """Assembles the report.json contents (docs/spec/04-tasks.md T31 acceptance): games,
    wins/draws/losses for the candidate, meanScore, pairedCI95, seed, engines argv,
    visits/time settings, openings file hash, and incident counts."""
    t = tally(games, candidate)
    paired_ci = bootstrap_mean_ci(pair_scores(games, candidate), resamples=resamples, seed=seed)
    unpaired_ci = bootstrap_mean_ci(t["scores"], resamples=resamples, seed=seed)
    return {
        "games": len(games),
        "countedGames": t["games"],
        "wins": t["wins"],
        "draws": t["draws"],
        "losses": t["losses"],
        "meanScore": t["meanScore"],
        "pairedCI95": paired_ci,
        "unpairedCI95": unpaired_ci,
        "eloEstimate": elo_from_score(t["meanScore"]),
        "seed": seed,
        "engines": engines,
        "settings": settings,
        "openingsFileHash": openings_file_hash,
        "incidents": incident_counts(games),
    }
