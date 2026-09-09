"""Real game-result targets from SGF ``RE`` tags (docs/spec/05-validation.md §5, T29).

``python -m ichigo_train sgf-results --sgf-dir DIR --positions data/positions-9.jsonl --out OUT``
parses each finished game's ``RE[...]`` tag into ``(result, kind)``::

    B+R / W+R                              -> resign
    B+<points> / W+<points>  (e.g. B+3.5)  -> score
    0 / Draw / Jigo                        -> draw

Only these forms are emitted; anything else (a time forfeit ``+T``, a disconnect/other forfeit
``+F``, unscored ``?``, ``Void``, a missing tag, or a malformed value) is skipped and counted in
the report rather than guessed at. Resignations legitimately settle win/loss
(docs/spec/05-validation.md §5 allows this), but this module never manufactures a score/ownership
target from one -- ``dataset.build_dataset``'s ``wdl_from_results`` only ever writes a win/draw/
loss WDL one-hot from this file's ``result`` field, so that restriction is structural rather than
needing extra plumbing here.

Games are matched to their SGF file through the positions file's ``sourceFile`` -> ``gameId`` map
(the same file the Swift ``ichigo features`` CLI wrote, docs/spec/02-training.md §2) rather than
by re-deriving ``PositionExport.canonicalHash`` from the SGF: RE/PB/PW/comments are not part of
that hash, and reading one tag does not need a byte-exact independent SGF replay.
"""

from __future__ import annotations

import json
import os
import re

RE_TAG = re.compile(r"RE\[([^\]]*)\]")
_SCORE = re.compile(r"^([BW])\+([0-9]+(?:\.[0-9]+)?)$", re.IGNORECASE)
_RESIGN = re.compile(r"^([BW])\+R(?:esign)?$", re.IGNORECASE)
_DRAW_VALUES = {"0", "draw", "jigo"}

RESULT_KINDS = ("resign", "score", "draw")


def parse_re(value: str | None) -> tuple[str, str] | None:
    """``value`` is the raw ``RE[...]`` payload (or ``None`` if the tag is absent). Returns
    ``(result, kind)`` with ``result`` in ``{"B","W","draw"}`` and ``kind`` in
    ``RESULT_KINDS``, or ``None`` if the tag is missing or not one of the recognised forms."""
    if value is None:
        return None
    v = value.strip()
    if v.lower() in _DRAW_VALUES:
        return "draw", "draw"
    m = _RESIGN.match(v)
    if m:
        return m.group(1).upper(), "resign"
    m = _SCORE.match(v)
    if m:
        return m.group(1).upper(), "score"
    return None


def extract_re(sgf_text: str) -> str | None:
    """First ``RE[...]`` tag's payload, or ``None``. SGF property values escape ``\\]``/``\\\\``;
    a real ``RE`` value (resign/score/draw/forfeit/void/"?") never contains ``]``, so a
    non-greedy bracket match is sufficient."""
    m = RE_TAG.search(sgf_text)
    return m.group(1) if m else None


def _source_file_map(positions_path: str) -> dict[str, str]:
    """``gameId -> sourceFile`` from the positions JSONL, first occurrence per game. Every row of
    one game carries the same ``(gameId, sourceFile)`` pair by construction (``PositionExport``
    writes one ``sourceFile`` per replayed SGF), so a mismatch means the positions file itself is
    inconsistent and is rejected rather than silently resolved by picking one."""
    source_of: dict[str, str] = {}
    with open(positions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gid, src = row["gameId"], row.get("sourceFile")
            if src is None:
                continue
            prior = source_of.get(gid)
            if prior is not None and prior != src:
                raise ValueError(f"gameId {gid} maps to two source files ({prior!r} and {src!r}) in {positions_path}")
            source_of[gid] = src
    return source_of


def build_results(sgf_dir: str, positions_path: str, out_path: str) -> dict:
    """Writes one JSONL row per resolvable game to ``out_path``: ``{"gameId","result","kind"}``,
    sorted by ``gameId`` (deterministic, independent of filesystem/positions-file row order).
    Returns a report of how many games were written vs. skipped and why."""
    source_of = _source_file_map(positions_path)
    report = {
        "positionsFile": positions_path, "sgfDir": sgf_dir, "out": out_path,
        "games": len(source_of), "gamesWritten": 0, "skippedUnparsed": 0, "missingFile": 0,
        "byKind": {k: 0 for k in RESULT_KINDS}, "byResult": {"B": 0, "W": 0, "draw": 0},
    }
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as out:
        for gid in sorted(source_of):
            src = source_of[gid]
            path = os.path.join(sgf_dir, src)
            if not os.path.isfile(path):
                report["missingFile"] += 1
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            parsed = parse_re(extract_re(text))
            if parsed is None:
                report["skippedUnparsed"] += 1
                continue
            result, kind = parsed
            out.write(json.dumps({"gameId": gid, "result": result, "kind": kind}) + "\n")
            report["gamesWritten"] += 1
            report["byKind"][kind] += 1
            report["byResult"][result] += 1
    return report
