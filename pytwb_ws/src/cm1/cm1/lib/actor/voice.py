"""ros_actor adapters for voice input and output."""

import threading

import py_trees
from ros_actor import SubNet, actor
from lib.voice import (
    DEVICE_INDEX,
    print_microphones,
    recognize_navigation_command,
    speak,
    test_microphone,
    transcribe_microphone,
)

_voice_cancel_event = threading.Event()


class VoiceNetwork(SubNet):
    # 音声入力を開始し、中断イベントを渡す
    def _listen(self, function):
        _voice_cancel_event.clear()
        return function(cancel_event=_voice_cancel_event)

    # マイクの確認・診断
    @actor
    def voice_list_mics(self):
        return print_microphones()

    @actor
    def voice_test_mic(self, seconds=3, device_index=DEVICE_INDEX):
        return test_microphone(seconds, device_index)

    # 音声認識とBlackboardへの反映
    @actor
    def voice_print(self):
        """Recognize and print one utterance without changing the blackboard."""
        return self._listen(transcribe_microphone) or False

    @actor
    def voice_command(self):
        """Recognize one command and store only its navigation pose."""
        pose = self._listen(recognize_navigation_command)
        if pose is None:
            return False
        py_trees.blackboard.Blackboard().set("target_pose", pose)
        return True

    # 実行中の録音・Whisperを中断
    @actor
    def voice_cancel(self):
        """Request cancellation of the active recording or transcription."""
        _voice_cancel_event.set()
        return True

    # 認識結果などの文字を読み上げ
    @actor
    def voice_speak(self, text):
        """Speak text offline via espeak-ng."""
        return speak(text)
