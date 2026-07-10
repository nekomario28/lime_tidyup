from pytwb.common import behavior, ActorBT


@behavior
class GoFrontCube(ActorBT):
    desc = 'center the cube and approach it'

    def __init__(
        self,
        name,
        node,
        threshold=0.80,
        center_tolerance=40.0,
        stop_box_width=300.0
    ):
        # navigationサブシステムを指定
        super(GoFrontCube, self).__init__(
            name,
            'navigation'
        )

        self.threshold = float(threshold)
        self.center_tolerance = float(center_tolerance)
        self.stop_box_width = float(stop_box_width)

    def initialise(self):
        # actorを呼び出す準備
        super().prepare()

        # system.pyのgo_front_cube actorを指定
        self.shared.set_callee([
            (
                'go_front_cube',
                (
                    self.threshold,
                    self.center_tolerance,
                    self.stop_box_width
                )
            )
        ])

        # actorを実行
        self.run()