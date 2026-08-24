from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class GoFrontCube(ActorBT):
    desc = 'turn toward cube and approach using accepted 3D pose or depth'

    def __init__(
        self,
        name,
        node,
        threshold=0.50,
        stop_distance=0.30,
        forward_speed=0.03,
        turn_speed=0.20,
        target_offset_px=40.0,
        pose_max_reprojection_error_px=4.0
    ):
        super(GoFrontCube, self).__init__(
            name,
            'navigation'
        )

        self.threshold = float(threshold)
        self.stop_distance = float(stop_distance)
        self.forward_speed = float(forward_speed)
        self.turn_speed = float(turn_speed)
        self.target_offset_px = float(target_offset_px)
        self.pose_max_reprojection_error_px = float(
            pose_max_reprojection_error_px
        )

    def initialise(self):
        super().prepare()

        print(
            'GoFrontCube args:',
            f'threshold={self.threshold}',
            f'stop_distance={self.stop_distance}',
            f'forward_speed={self.forward_speed}',
            f'turn_speed={self.turn_speed}',
            f'target_offset_px={self.target_offset_px}',
            'pose_max_reprojection_error_px='
            f'{self.pose_max_reprojection_error_px}'
        )

        self.shared.set_callee([
            (
                'go_front_cube',
                (
                    self.threshold,
                    self.stop_distance,
                    self.forward_speed,
                    self.turn_speed,
                    self.target_offset_px,
                    self.pose_max_reprojection_error_px
                )
            )
        ])

        self.run()
