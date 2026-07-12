from array import array
import math
import threading
import time

from cm1.lib.locations import location_pose
from cm1.lib import voice
from cm1.lib.voice import parse_voice_command, pcm16_rms


def test_parse_voice_command():
    assert parse_voice_command("Aに行って") == "A"
    assert parse_voice_command("ビーへ移動してください") == "B"
    assert parse_voice_command("Cに向かって") == "C"


def test_parse_voice_command_rejects_non_movement_text():
    assert parse_voice_command("Aです") is None
    assert parse_voice_command("こんにちは") is None
    assert parse_voice_command("") is None


def test_location_pose_converts_degrees_to_radians():
    x, y, theta = location_pose("A")
    assert x == -0.1330670714378357
    assert y == -1.0817683935165405
    assert math.isclose(theta, math.radians(270))


def test_pcm16_rms():
    assert pcm16_rms(b"") == 0
    assert pcm16_rms(array("h", [1000, -1000]).tobytes()) == 1000


def test_blocking_work_can_be_cancelled():
    cancel_event = threading.Event()

    def worker():
        time.sleep(0.5)
        return "late result"

    threading.Timer(0.05, cancel_event.set).start()
    result = voice._run_cancellable(
        worker,
        cancel_event,
        poll_interval=0.01,
    )

    assert result is None
