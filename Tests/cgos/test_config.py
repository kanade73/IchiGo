"""Config validation (docs/spec/04-tasks.md T30: "absolute password path rejected inside JSON,
argv must be a list")."""

import json
from pathlib import Path

import pytest

from ichigo_cgos_client import ConfigError, load_config

BASE_CONFIG = {
    "host": "cgos.example.invalid",
    "port": 6867,
    "username": "ichigo-test",
    "password_file": "secrets/password.txt",
    "board_size": 9,
    "engine_argv": [".build/release/ichigo", "gtp", "--model-9", "models/x.ichigo"],
    "analysis": True,
    "state_dir": "runs/cgos/ichigo-test/state",
    "log_dir": "runs/cgos/ichigo-test/logs",
}


def _write_config(tmp_path: Path, overrides: dict) -> Path:
    cfg = dict(BASE_CONFIG)
    cfg.update(overrides)
    path = tmp_path / "cgos.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path):
    path = _write_config(tmp_path, {})
    config = load_config(path)
    assert config.host == "cgos.example.invalid"
    assert config.port == 6867
    assert config.board_size == 9
    assert config.engine_argv == (".build/release/ichigo", "gtp", "--model-9", "models/x.ichigo")
    assert config.analysis_enabled is True
    # relative paths resolve against the config file's own directory
    assert config.password_file == (tmp_path / "secrets" / "password.txt").resolve()


def test_absolute_password_path_rejected(tmp_path):
    path = _write_config(tmp_path, {"password_file": "/etc/secrets/password.txt"})
    with pytest.raises(ConfigError, match="password_file"):
        load_config(path)


def test_absolute_windows_style_password_path_rejected(tmp_path):
    path = _write_config(tmp_path, {"password_file": "C:\\secrets\\password.txt"})
    with pytest.raises(ConfigError, match="password_file"):
        load_config(path)


def test_argv_must_be_a_list_not_a_string(tmp_path):
    path = _write_config(tmp_path, {"engine_argv": ".build/release/ichigo gtp --model-9 models/x.ichigo"})
    with pytest.raises(ConfigError, match="argv must be a list"):
        load_config(path)


def test_argv_must_be_non_empty(tmp_path):
    path = _write_config(tmp_path, {"engine_argv": []})
    with pytest.raises(ConfigError, match="argv must be a list"):
        load_config(path)


def test_argv_entries_must_be_strings(tmp_path):
    path = _write_config(tmp_path, {"engine_argv": [".build/release/ichigo", 9]})
    with pytest.raises(ConfigError):
        load_config(path)


def test_board_size_must_be_9_or_19(tmp_path):
    path = _write_config(tmp_path, {"board_size": 13})
    with pytest.raises(ConfigError, match="board_size"):
        load_config(path)


def test_missing_required_key_rejected(tmp_path):
    cfg = dict(BASE_CONFIG)
    del cfg["username"]
    path = tmp_path / "cgos.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="username"):
        load_config(path)


def test_config_must_be_a_json_object(tmp_path):
    path = tmp_path / "cgos.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(path)


def test_analysis_must_be_boolean(tmp_path):
    path = _write_config(tmp_path, {"analysis": "true"})
    with pytest.raises(ConfigError, match="analysis"):
        load_config(path)


def test_port_must_be_integer(tmp_path):
    path = _write_config(tmp_path, {"port": "6867"})
    with pytest.raises(ConfigError, match="port"):
        load_config(path)


def test_invalid_json_reports_config_error(tmp_path):
    path = tmp_path / "cgos.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_state_dir_and_log_dir_may_be_absolute(tmp_path):
    abs_state = tmp_path / "abs-state"
    abs_log = tmp_path / "abs-log"
    path = _write_config(tmp_path, {"state_dir": str(abs_state), "log_dir": str(abs_log)})
    config = load_config(path)
    assert config.state_dir == abs_state
    assert config.log_dir == abs_log
