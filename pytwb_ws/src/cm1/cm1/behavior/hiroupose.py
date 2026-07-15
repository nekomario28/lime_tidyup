from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class Hirou_Pose(ActorBT):
    desc = 'move arm to hirou pose'

    def __init__(self, name, node):
        super(Hirou_Pose, self).__init__(
            name,
            'manipulator'
        )

    def initialise(self):
        super().prepare()

        self.shared.set_callee([
            (
                'hirou_pose',
                None
            )
        ])

        self.run()