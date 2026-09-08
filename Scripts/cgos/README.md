# IchiGo CGOS client

Implements docs/spec/04-tasks.md T30 and docs/spec/03-engine.md §9 ("勝率表示と CGOS"). Speaks the
public CGOS wire protocol (https://github.com/zakki/cgos, pinned to revision
`4dcff8754400fa43a71323ffc3d2c5eb4a19f7f1`) directly to an `ichigo gtp` subprocess. Python 3.12
stdlib only -- no dependencies to install.

- `ichigo_cgos_client.py` -- the client. See its module docstring for the full wire-protocol
  grammar and the exact CGOS analysis JSON format it produces, both cross-checked against the
  pinned revision's client *and* server source.
- `fake_cgos_server.py` -- a minimal local CGOS server for tests and local dry-runs. Pairs two
  logged-in clients per game and relays moves between them; it does **not** implement Go rules
  (both sides are real engines and already enforce legality themselves).

See `docs/provenance/cgos-import.json` for exactly what was read while writing these files (paths
+ sha256, at the pinned revision and locally) and what idea/grammar was taken from each.

## Quick start: play against yourself locally, no real server

This is the fastest way to see the whole thing work end to end. It needs a release build of
`ichigo` and a model (both already exist after `swift build -c release --product ichigo`, per the
repo's own build instructions).

```sh
# 1. From the repo root, start the fake server in one terminal:
python3 Scripts/cgos/fake_cgos_server.py --port 6867 \
  --account ichigo-a:pw-a --account ichigo-b:pw-b

# 2. In two other terminals, write a password file and a config for each side, then run the
#    client. (A real deployment keeps these under state/, gitignored -- this is just for the
#    local dry run.)
mkdir -p /tmp/ichigo-cgos-a /tmp/ichigo-cgos-b
echo pw-a > /tmp/ichigo-cgos-a/password.txt
echo pw-b > /tmp/ichigo-cgos-b/password.txt

cat > /tmp/ichigo-cgos-a/cgos.json <<'JSON'
{
  "host": "127.0.0.1", "port": 6867,
  "username": "ichigo-a", "password_file": "password.txt",
  "board_size": 9,
  "engine_argv": [".build/release/ichigo", "gtp", "--model-9", "models/p2-small-gl10.ichigo", "--visits", "200"],
  "analysis": true,
  "state_dir": "/tmp/ichigo-cgos-a/state", "log_dir": "/tmp/ichigo-cgos-a/logs"
}
JSON
# (same for b, swapping ichigo-a -> ichigo-b and the password file)

python3 Scripts/cgos/ichigo_cgos_client.py --config /tmp/ichigo-cgos-a/cgos.json --games 1
python3 Scripts/cgos/ichigo_cgos_client.py --config /tmp/ichigo-cgos-b/cgos.json --games 1
```

Both clients log to `<log_dir>/ichigo_cgos_client.log` (rotated at 20 MiB, 5 backups kept) and to
stderr. The fake server prints pairing/gameover events to stdout. Ctrl-C the server when done.

This is exactly the shape `Tests/cgos/test_cgos_integration.py` automates (with `--visits 4` and
a short per-game clock, so it finishes in well under a minute instead of needing a real think
time).

## Config reference (`configs/cgos.example.json`)

| field | meaning |
|---|---|
| `host`, `port` | CGOS server to connect to. Never a real server's address in a committed config -- see below. |
| `username` | CGOS login name. |
| `password_file` | **Relative path** to a file containing just the password. Must not be absolute (rejected at load time) and must never be committed with real contents -- keep it under a gitignored directory such as `state_dir` or a repo-root `secrets/` folder. |
| `board_size` | `9` or `19`. Must match the model given in `engine_argv`. |
| `engine_argv` | The exact argv to launch the engine, as a **JSON array** (never a shell string -- a string would need naive splitting, which breaks on any path containing a space; docs/spec/03-engine.md §9: "パス空白をnaive splitしない"). E.g. `[".build/release/ichigo", "gtp", "--model-9", "models/x.ichigo"]`. |
| `analysis` | `true` to request `kata-genmove_analyze` and send the CGOS analysis extension when the server offers `genmove_analyze`; `false` to always use plain `genmove`. |
| `state_dir` | Per-game move ledgers (`state_dir/games/<gid>.json`) live here. Must not be shared with any other CGOS client (this or RinGo's) running at the same time. |
| `log_dir` | Rotating log file lives here (20 MiB × 5, `logging.handlers.RotatingFileHandler`). Also must be exclusive to this client instance. |
| `max_games` | Optional. Stop (gracefully, between games) after this many completed games. `null`/omitted = unlimited. The CLI's `--games N` flag overrides this. |

## Setting up against the real public CGOS server

Nothing in this repository may contain a real host, account name, or password. Before connecting
for real:

1. Copy `configs/cgos.example.json` to a config **outside version control** (e.g.
   `configs/cgos.local.json`, or anywhere under a directory your `.gitignore` already excludes --
   `configs/` itself is gitignored in this repo, matching every other example config here) and
   fill in the real `host`/`port`/`username`.
2. Create the password file it points at (`password_file`, relative to the config file's own
   directory) with just the password in it, and make sure that path is not committed either.
3. Point `engine_argv` at the release binary and the model you intend to run
   (`swift build -c release --product ichigo` first if `.build/release/ichigo` doesn't exist).
   Confirm `board_size` matches that model.
4. Give it its own `state_dir`/`log_dir` (e.g. under a local, gitignored `runs/cgos/<name>/`) --
   distinct from any other CGOS client's directories, this client's own default suggestion in
   `cgos.example.json`, and from RinGo's `~/dev/univ/koubou/katago-mlx` CGOS state entirely
   (different repository, different process invocation: `python3 Scripts/cgos/
   ichigo_cgos_client.py ...` vs. RinGo's `launch-cgos.sh`/`cgosclient.py`, so `ps` output
   never confuses the two).
5. Run it: `python3 Scripts/cgos/ichigo_cgos_client.py --config <your-config>.json`. It runs
   until stopped (Ctrl-C / `SIGTERM`) or, if `max_games`/`--games` is set, until that many games
   complete.
6. To stop cleanly: send `SIGINT` (Ctrl-C) or `SIGTERM`. The client finishes whatever game is in
   progress, replies `ready`/`quit` to the server appropriately, shuts the engine subprocess down
   with `quit`, and exits 0. It never tears a game down mid-move.

Model and engine binary hashes are computed once, when the client starts (sha256 of the engine
binary plus a sorted-manifest sha256 of every file under the model directory named by
`--model-9`/`--model-19`/`--model` in `engine_argv`), and logged at the start of every game. The
engine subprocess is never restarted or replaced for the life of the client process --
docs/spec/03-engine.md §9: "対局途中model差替えなし".

## Operational behaviour

- **Reconnect**: on any socket/protocol error, the client reconnects with backoff
  1/2/4/8/16/30 seconds (capped), reset to 1s after a successful reconnect. On the server's
  `setup` for an in-progress game (a genuine resume, or catching up a game we'd missed some of),
  the client always does `boardsize → komi → clear_board → time_settings → replay every move via
  play → time_left` (both colours, restored from the last time value seen per colour in the
  replay). Because the engine's board is rebuilt from `clear_board` plus that authoritative
  replay every time, a move is never double-applied to the engine regardless of what happened
  before the reconnect; the per-game ledger under `state_dir/games/<gid>.json` mirrors this by
  being rewritten from scratch on every `setup`, rather than incrementally diffed.
- **Graceful stop**: `SIGTERM`/`SIGINT` set a flag checked exactly once, right after a `gameover`
  is fully processed -- never mid-game. If set (or the game-count limit was reached), the client
  replies `quit` instead of `ready` and exits.
- **Log rotation**: `logging.handlers.RotatingFileHandler`, 20 MiB × 5 backups, per
  docs/spec/03-engine.md §9.
- **State isolation**: every config gets its own `state_dir`/`log_dir`; nothing here reads or
  writes another CGOS client's directories (this codebase's or RinGo's).

## Testing

```sh
cd Training && uv run pytest -q ../Tests/cgos
```

- Fast, no-engine unit tests: backoff sequence, config validation (including the absolute
  `password_file` path and non-list `engine_argv` rejections), the `kata-genmove_analyze` → CGOS
  analysis-JSON parser (checked against the exact example line from docs/spec/03-engine.md §9),
  log-rotation configuration and actual rotation behaviour, model/engine hashing, and the move
  ledger.
- Fast, no-engine `fake_cgos_server.py` protocol tests (`test_fake_server_protocol.py`): pairing
  order (white named before black, black moves first), analysis-string recording, and a scripted
  mid-game disconnect + `setup` replay -- using plain-socket stub clients instead of real engines,
  as a low-cost regression net for the server's own logic.
- Real-engine end-to-end tests (`test_cgos_integration.py`, builds `.build/release/ichigo` if
  missing): two `ichigo_cgos_client.py` processes, each driving a real
  `ichigo gtp --model-9 models/p2-small-gl10.ichigo --visits 4` engine, playing through
  `fake_cgos_server.py`. Covers two complete 9×9 games reaching `gameover` with results and
  recorded analysis strings, a mid-game disconnect + `setup`-replay resume with no duplicated
  moves (cross-checked against the client's own ledger file), and a `SIGTERM` sent mid-game
  stopping the client only after that game completes (never starting a further game). These take
  a few minutes in total, since each is a real search against a real (if tiny) model.

## What this does not verify

Everything above is local: a fake server, on loopback, with test accounts. It does not verify
connecting to, authenticating with, or playing on the real public CGOS server -- that needs a
real account and connection details, which are supplied by whoever operates this client at
deployment time (T37; docs/spec/03-engine.md §9: "公開CGOSのアカウントと接続先はユーザーが実運用
時に設定する"). In particular this cannot verify by itself: the real server's actual `setup`/
`genmove`/`gameover` message content matching this document's grammar (it was cross-checked
against the pinned client *and* server source instead, not observed live), real network
conditions (latency, actual multi-second disconnects, the server's own move-limit/adjudication
policy if any), or whether the real server's UI actually displays the analysis strings this
client sends (docs/spec/05-validation.md §6 M2b: "クライアントによる解析値受理とサーバー側の保存
または表示を確認する" is a manual step at real-connection time, not something a local test can
confirm).
