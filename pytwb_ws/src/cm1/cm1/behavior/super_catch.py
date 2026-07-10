from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class PlacePose(ActorBT):
    desc = 'move arm to the placing pose without opening gripper'

    def __init__(self, name, node):
        super(PlacePose, self).__init__(
            name,
            'manipulator'
        )

    def initialise(self):
        super().prepare()

        self.shared.set_callee([
            ('place_pose', ())
        ])

        self.run()