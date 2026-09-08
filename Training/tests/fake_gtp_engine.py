"""Scripted fake GTP engine for match.py tests (Training/tests only, never shipped as a package).

Implements just enough GTP to drive ichigo_train.match.MatchRunner: protocol_version, name,
version, known_command, list_commands, boardsize, clear_board, komi, play, genmove,
time_settings, time_left, final_score, quit. Reads commands from stdin (an optional leading
numeric id, matching what GTPClient.send writes), replies with `=id ...`/`?id ...` terminated by
a blank line; nothing but GTP goes to stdout (diagnostics, e.g. a fake "model=" hash line like
the real engine logs, go to stderr).

This is a stub, not a rules engine: `play` accepts any vertex for any colour unless it matches
--reject-play, and `genmove` just pops the next vertex off --moves (ignoring which colour asked,
so the caller's script must already account for whose turn it is). Real Go legality is exercised
by the Swift engine's own tests, not by these Python match-runner tests (see match.py's module
docstring: the match runner trusts whatever a real GTP engine's `play`/`genmove` say).

Usage: python fake_gtp_engine.py [--moves V1,V2,...] [--reject-play V] [--sleep-genmove S]
                                  [--result R] [--model-hash H]
"""

import argparse
import sys
import time


def log(msg: str) -> None:
    sys.stderr.write(f"[fake-gtp] {msg}\n")
    sys.stderr.flush()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", default="", help="comma-separated GTP vertices returned by genmove, in order; 'pass' after that")
    ap.add_argument("--reject-play", default=None, help="a vertex for which `play` always fails (any colour)")
    ap.add_argument("--sleep-genmove", type=float, default=0.0, help="seconds to sleep before answering genmove (simulate a hang)")
    ap.add_argument("--result", default="0", help="final_score reply")
    ap.add_argument("--model-hash", default=None, help="if set, logs 'model=<hash>' to stderr on genmove, like the real engine")
    a = ap.parse_args(argv)
    moves = [m for m in a.moves.split(",") if m]
    move_i = 0

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].isdigit():
            cid, rest = parts[0], parts[1:]
        else:
            cid, rest = None, parts
        if not rest:
            continue
        cmd, args = rest[0].lower(), rest[1:]

        def reply(ok: bool, payload: str = "") -> None:
            marker = "=" if ok else "?"
            sys.stdout.write(f"{marker}{cid or ''} {payload}\n\n")
            sys.stdout.flush()

        if cmd == "protocol_version":
            reply(True, "2")
        elif cmd == "name":
            reply(True, "fake-gtp")
        elif cmd == "version":
            reply(True, "0")
        elif cmd == "known_command":
            reply(True, "true")
        elif cmd == "list_commands":
            reply(True, "protocol_version\nname\nversion\nboardsize\nclear_board\nkomi\nplay\ngenmove\ntime_settings\ntime_left\nfinal_score\nquit")
        elif cmd == "boardsize":
            reply(True)
        elif cmd == "clear_board":
            move_i = 0
            reply(True)
        elif cmd == "komi":
            reply(True)
        elif cmd == "play":
            vertex = args[1] if len(args) > 1 else None
            if a.reject_play is not None and vertex is not None and vertex.upper() == a.reject_play.upper():
                reply(False, "illegal move")
            else:
                reply(True)
        elif cmd == "genmove":
            if a.sleep_genmove:
                time.sleep(a.sleep_genmove)
            if move_i < len(moves):
                v = moves[move_i]
                move_i += 1
            else:
                v = "pass"
            if a.model_hash:
                log(f"genmove {v} visits=1 model={a.model_hash}")
            reply(True, v)
        elif cmd == "time_settings":
            reply(True)
        elif cmd == "time_left":
            reply(True)
        elif cmd == "final_score":
            reply(True, a.result)
        elif cmd == "quit":
            reply(True)
            break
        else:
            reply(False, "unknown command")


if __name__ == "__main__":
    main()
