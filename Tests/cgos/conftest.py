"""Shared fixtures for the CGOS client/fake-server tests (docs/spec/04-tasks.md T30).

Run with: ``cd Training && uv run pytest -q ../Tests/cgos``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CGOS_DIR = REPO_ROOT / "Scripts" / "cgos"
CLIENT_SCRIPT = CGOS_DIR / "ichigo_cgos_client.py"
FAKE_SERVER_MODULE = CGOS_DIR / "fake_cgos_server.py"
RELEASE_BINARY = REPO_ROOT / ".build" / "release" / "ichigo"
DEFAULT_MODEL = REPO_ROOT / "models" / "p2-small-gl10.ichigo"

for _p in (str(REPO_ROOT), str(CGOS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def release_binary() -> Path:
    """Builds `.build/release/ichigo` if it is not already present (docs/spec/04-tasks.md T30:
    "build the release binary if missing")."""
    if not RELEASE_BINARY.exists():
        subprocess.run(
            ["swift", "build", "-c", "release", "--product", "ichigo"],
            cwd=REPO_ROOT, check=True, timeout=1800,
        )
    assert RELEASE_BINARY.exists(), f"release binary missing after build: {RELEASE_BINARY}"
    return RELEASE_BINARY


@pytest.fixture(scope="session")
def model_path() -> Path:
    assert DEFAULT_MODEL.is_dir(), f"test fixture model not found: {DEFAULT_MODEL}"
    return DEFAULT_MODEL


def engine_argv(release_binary: Path, model_path: Path, visits: int = 4) -> List[str]:
    return [
        str(release_binary), "gtp", "--model-9", str(model_path), "--visits", str(visits),
    ]


def spawn_client(config_path: Path, *, games: int, extra_env: dict | None = None) -> subprocess.Popen:
    """Starts `ichigo_cgos_client.py --config ... --games N` as a real subprocess."""
    import os

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, str(CLIENT_SCRIPT), "--config", str(config_path), "--games", str(games)],
        cwd=REPO_ROOT,
        env=env,
    )
