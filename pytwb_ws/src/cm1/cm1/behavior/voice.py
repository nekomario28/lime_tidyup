import py_trees

from lib.actor_bt import ActorBT
from pytwb.common import behavior
from ros_actor import run_actor


# 中断可能な音声入力Behaviorの共通処理
class VoiceInput(ActorBT):
    actor = None

    def __init__(self, name, node):
        super().__init__(name, self.actor)

    def terminate(self, new_status):
        if new_status == py_trees.common.Status.INVALID:
            run_actor("voice_cancel")
        return super().terminate(new_status)


# 音声命令、文字表示、読み上げをXMLから使えるようにする
@behavior
class VoiceCommand(VoiceInput):
    desc = "recognize a movement command and set target_pose"
    actor = "voice_command"


@behavior
class PrintVoiceText(VoiceInput):
    desc = "recognize and print speech without changing the blackboard"
    actor = "voice_print"


@behavior
class VoiceSpeak(ActorBT):
    desc = "speak text via TTS"

    def __init__(self, name, node, text=""):
        super().__init__(name, "voice_speak", text)
