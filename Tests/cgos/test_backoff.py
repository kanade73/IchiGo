"""Backoff sequence (docs/spec/03-engine.md §9: "backoffは1/2/4/8/16/30秒上限")."""

from ichigo_cgos_client import Backoff


def test_backoff_sequence_caps_at_30():
    b = Backoff()
    delays = [b.next_delay() for _ in range(9)]
    assert delays == [1, 2, 4, 8, 16, 30, 30, 30, 30]


def test_backoff_reset_restarts_from_one():
    b = Backoff()
    for _ in range(4):
        b.next_delay()
    b.reset()
    assert b.next_delay() == 1
    assert b.next_delay() == 2
