from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class TakeAim(ActorBT):
    desc = 'turn toward cube center'

    def __init__(
        self,
        name,
        node,
        threshold=0.50,
        turn_speed=0.20,
        center_tolerance=20.0
    ):
        super(TakeAim, self).__init__(
            name,
            'navigation'
        )

        self.threshold = float(threshold)
        self.turn_speed = float(turn_speed)
        self.center_tolerance = float(center_tolerance)

    def initialise(self):
        super().prepare()

        self.shared.set_callee([
            (
                'take_aim',
                (
                    self.threshold,
                    self.turn_speed,
                    self.center_tolerance
                )
            )
        ])

        self.run()