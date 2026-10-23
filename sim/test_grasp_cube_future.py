"""Tests for waiting on ROS futures while the node spins in its callback thread."""
from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grasp_cube import wait_for_future  # noqa: E402


class FakeFuture:
    def __init__(self, value, complete=True):
        self.value = value
        self.complete = complete

    def add_done_callback(self, callback):
        if self.complete:
            threading.Thread(target=callback, args=(self,), daemon=True).start()

    def result(self):
        return self.value


def test_waits_for_a_future_completed_by_the_executor_thread():
    expected = object()
    assert wait_for_future(FakeFuture(expected), timeout=1.0) is expected


def test_timeout_does_not_spin_or_block_forever():
    assert wait_for_future(FakeFuture(None, complete=False), timeout=0.001) is None
