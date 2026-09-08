"""Log rotation (docs/spec/03-engine.md §9: "ログrotateは20MiB×5")."""

import logging
import logging.handlers

from ichigo_cgos_client import LOG_ROTATE_BACKUP_COUNT, LOG_ROTATE_MAX_BYTES, build_logger


def test_constants_match_spec():
    assert LOG_ROTATE_MAX_BYTES == 20 * 1024 * 1024
    assert LOG_ROTATE_BACKUP_COUNT == 5


def test_build_logger_configures_rotating_file_handler(tmp_path):
    logger = build_logger(tmp_path, name="test-ichigo-cgos-client")
    try:
        rotating = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(rotating) == 1
        handler = rotating[0]
        assert handler.maxBytes == 20 * 1024 * 1024
        assert handler.backupCount == 5
        assert (tmp_path / "test-ichigo-cgos-client.log").exists()
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()


def test_build_logger_is_idempotent_for_the_same_name(tmp_path):
    # A second call (e.g. a reconnect helper re-fetching the logger) must not stack duplicate
    # handlers, or the 20MiB budget would silently multiply.
    logger1 = build_logger(tmp_path, name="test-idempotent")
    logger2 = build_logger(tmp_path, name="test-idempotent")
    try:
        assert logger1 is logger2
        assert len(logger1.handlers) == 2  # one rotating file handler + one stderr stream handler
    finally:
        for h in list(logger1.handlers):
            logger1.removeHandler(h)
            h.close()


def test_rotating_file_handler_rotates_and_prunes_backups(tmp_path):
    # Exercise the actual rotation behaviour with a small maxBytes so the test stays fast; the
    # 20MiB/5 *values* are covered by test_constants_match_spec and the config test above.
    log_path = tmp_path / "rotate-test.log"
    handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=500, backupCount=3, encoding="utf-8")
    logger = logging.getLogger("test-rotating-file-handler")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        for i in range(400):
            logger.info("padding line %04d %s", i, "x" * 40)
    finally:
        logger.removeHandler(handler)
        handler.close()

    assert log_path.exists()
    rotated = sorted(tmp_path.glob("rotate-test.log.*"))
    assert 1 <= len(rotated) <= 3
    assert log_path.stat().st_size <= 500 + 200  # last record may push slightly past maxBytes
