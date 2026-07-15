from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class SearchCube(ActorBT):
    desc = 'search cube until centered'

    def __init__(
        self,
        name,
        node,
        threshold=0.80,
        center_tolerance=120.0,
        image_width=848.0
    ):
        super(SearchCube, self).__init__(
            name,
            'navigation'
        )

        self.threshold = float(threshold)
        self.center_tolerance = float(center_tolerance)
        self.image_width = float(image_width)

    def initialise(self):
        super().prepare()

        self.shared.set_callee([
            (
                'search_cube',
                (
                    self.threshold,
                    self.center_tolerance,
                    self.image_width
                )
            )
        ])

        self.run()