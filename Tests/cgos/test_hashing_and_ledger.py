"""Model/binary hashing and the per-game move ledger (docs/spec/03-engine.md §9: "Model/binary
hashes are fixed at game start ... and logged" / "keep a move ledger in the state dir")."""

import hashlib
import json

from ichigo_cgos_client import MoveLedger, hash_engine_and_models, hash_file, hash_model_dir, model_dirs_from_argv


def test_hash_file_matches_sha256(tmp_path):
    f = tmp_path / "bin"
    f.write_bytes(b"hello world")
    assert hash_file(f) == hashlib.sha256(b"hello world").hexdigest()


def test_hash_model_dir_is_stable_and_order_independent(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "b.txt").write_bytes(b"bbb")
    (d / "a.txt").write_bytes(b"aaa")
    h1 = hash_model_dir(d)

    d2 = tmp_path / "model2"
    d2.mkdir()
    (d2 / "a.txt").write_bytes(b"aaa")
    (d2 / "b.txt").write_bytes(b"bbb")
    h2 = hash_model_dir(d2)
    assert h1 == h2


def test_hash_model_dir_changes_when_content_changes(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "a.txt").write_bytes(b"aaa")
    h1 = hash_model_dir(d)
    (d / "a.txt").write_bytes(b"aaZ")
    h2 = hash_model_dir(d)
    assert h1 != h2


def test_model_dirs_from_argv_finds_model_flags():
    argv = [".build/release/ichigo", "gtp", "--model-9", "models/x.ichigo", "--visits", "4"]
    assert model_dirs_from_argv(argv) == [__import__("pathlib").Path("models/x.ichigo")]


def test_hash_engine_and_models(tmp_path):
    engine = tmp_path / "ichigo"
    engine.write_bytes(b"binary-contents")
    model_dir = tmp_path / "models" / "x.ichigo"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text("{}", encoding="utf-8")

    argv = [str(engine), "gtp", "--model-9", str(model_dir)]
    hashes = hash_engine_and_models(argv)
    assert hashes["engine_binary"] == hashlib.sha256(b"binary-contents").hexdigest()
    assert hashes["model_manifest"] != ""
    # Stable and sensitive to the model directory's own manifest hash (it need not equal
    # hash_model_dir(model_dir) directly -- it also folds in the model path, to distinguish
    # --model-9/--model-19 pairs -- but it must change if the model contents change).
    assert hashes["model_manifest"] == hash_engine_and_models(argv)["model_manifest"]
    (model_dir / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
    assert hash_engine_and_models(argv)["model_manifest"] != hashes["model_manifest"]


def test_hash_engine_and_models_without_model_flag(tmp_path):
    engine = tmp_path / "ichigo"
    engine.write_bytes(b"x")
    hashes = hash_engine_and_models([str(engine)])
    assert hashes["model_manifest"] == ""


def test_move_ledger_reset_overwrites_stale_state(tmp_path):
    state_dir = tmp_path / "state"
    ledger = MoveLedger.reset(state_dir, gid="1", boardsize="9", komi="7.0")
    ledger.append("b", "d4", analysis=None, source="replay")
    ledger.append("w", "q4", analysis='{"a":1}', source="self")

    path = state_dir / "games" / "1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["gid"] == "1"
    assert len(payload["moves"]) == 2
    assert payload["result"] is None

    # A fresh setup for the same gid (e.g. after a reconnect) must rebuild the ledger from
    # scratch, not append onto the stale one -- this is what makes replay double-apply-proof at
    # the bookkeeping layer regardless of what happened before the reconnect.
    ledger2 = MoveLedger.reset(state_dir, gid="1", boardsize="9", komi="7.0")
    ledger2.append("b", "d4", analysis=None, source="replay")
    payload2 = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload2["moves"]) == 1


def test_move_ledger_mark_complete_persists_result(tmp_path):
    state_dir = tmp_path / "state"
    ledger = MoveLedger.reset(state_dir, gid="2", boardsize="9", komi="7.0")
    ledger.mark_complete("B+Resign")
    path = state_dir / "games" / "2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["result"] == "B+Resign"
