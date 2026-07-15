from math import radians

from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class TurnLeft(ActorBT):
    desc = 'turn left by time and speed'

    def __init__(
        self,
        name,
        node,
        turn_seconds=0.50,
        turn_speed=0.20
    ):
        super(TurnLeft, self).__init__(
            name,
            'navigation'
        )

        self.turn_seconds = float(turn_seconds)
        self.turn_speed = abs(float(turn_speed))

    def initialise(self):
        super().prepare()

        # 回転角度 = 角速度 × 時間
        angle_rad = (
            self.turn_speed
            * self.turn_seconds
        )

        self.shared.set_callee([
            (
                'rotate_angle',
                (
                    angle_rad,
                    self.turn_speed
                )
            )
        ])

        self.run()