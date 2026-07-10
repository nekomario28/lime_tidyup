from lib.actor_bt import ActorBT
from pytwb.common import behavior


@behavior
class VoiceRecognition(ActorBT):
    desc = "listen to microphone and recognize speech"

    def __init__(self, name, node, language="ja", model="large-v3-turbo", device_index=-1):
        super().__init__(
            name, "voice_recognize", language, "recognized_text", device_index, model
        )


@behavior
class PrintVoiceText(ActorBT):
    desc = "print recognized speech text as plain text"

    def __init__(self, name, node):
        super().__init__(name, "voice_print_text")


@behavior
class ProcessVoiceCommand(ActorBT):
    desc = "parse recognized speech and set target_pose for A/B/C"

    def __init__(self, name, node):
        super().__init__(name, "voice_process")


@behavior
class VoiceSpeak(ActorBT):
    desc = "speak text via TTS"

    def __init__(self, name, node, text=""):
        super().__init__(name, "voice_speak", text)
