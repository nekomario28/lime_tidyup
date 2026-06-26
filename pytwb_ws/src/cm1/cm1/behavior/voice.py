from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class VoiceRecognition(ActorBT):
    desc = 'listen to microphone and recognize speech'

    def __init__(self, name, node, language='ja-JP'):
        super().__init__(name, 'voice_recognize', language)


@behavior
class ProcessVoiceCommand(ActorBT):
    desc = 'parse recognized speech and set target_pose for A/B/C'

    def __init__(self, name, node):
        super().__init__(name, 'voice_process')
