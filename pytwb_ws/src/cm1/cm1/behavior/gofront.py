from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class GoFrontCube(ActorBT):
    desc = 'turn toward cube and approach it'

    def __init__(
        self,
        name,
        node,
        threshold=0.80,
        center_tolerance=25.0,
        stop_box_width=300.0,
        horizontal_fov=60.0
    ):
        super(GoFrontCube, self).__init__(
            name,
            'navigation'
        )

        self.threshold = float(threshold)
        self.center_tolerance = float(center_tolerance)
        self.stop_box_width = float(stop_box_width)
        self.horizontal_fov = float(horizontal_fov)

    def initialise(self):
        super().prepare()

        print(
            'GoFrontCube args:',
            self.threshold,
            self.center_tolerance,
            self.stop_box_width,
            self.horizontal_fov
        )

        self.shared.set_callee([
            (
                'go_front_cube',
                (
                    self.threshold,
                    self.center_tolerance,
                    self.stop_box_width,
                    self.horizontal_fov
                )
            )
        ])

        self.run()