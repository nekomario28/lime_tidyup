"""Behavior-tree adapter for bounded cube approach."""

from pytwb.common import behavior
from lib.actor_bt import ActorBT


@behavior
class GoFrontCube(ActorBT):
    """Turn and approach using configured pose or depth evidence."""

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
        pose_max_reprojection_error_px=4.0,
        use_pose_3d=1.0,
        depth_sample_count=1.0,
        depth_roi_fraction=0.0,
        depth_minimum_valid_ratio=0.0,
        depth_maximum_mad_m=10.0,
        depth_maximum_spread_m=10.0
    ):
        """Store approach and depth-quality parameters for the actor call."""
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
        self.use_pose_3d = float(use_pose_3d)
        self.depth_sample_count = float(depth_sample_count)
        self.depth_roi_fraction = float(depth_roi_fraction)
        self.depth_minimum_valid_ratio = float(
            depth_minimum_valid_ratio
        )
        self.depth_maximum_mad_m = float(depth_maximum_mad_m)
        self.depth_maximum_spread_m = float(depth_maximum_spread_m)

    def initialise(self):
        """Dispatch one configured cube-approach actor call."""
        super().prepare()

        print(
            'GoFrontCube args:',
            f'threshold={self.threshold}',
            f'stop_distance={self.stop_distance}',
            f'forward_speed={self.forward_speed}',
            f'turn_speed={self.turn_speed}',
            f'target_offset_px={self.target_offset_px}',
            'pose_max_reprojection_error_px='
            f'{self.pose_max_reprojection_error_px}',
            f'use_pose_3d={self.use_pose_3d}',
            f'depth_sample_count={self.depth_sample_count}',
            f'depth_roi_fraction={self.depth_roi_fraction}',
            'depth_minimum_valid_ratio='
            f'{self.depth_minimum_valid_ratio}',
            f'depth_maximum_mad_m={self.depth_maximum_mad_m}',
            f'depth_maximum_spread_m={self.depth_maximum_spread_m}'
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
                    self.pose_max_reprojection_error_px,
                    self.use_pose_3d,
                    self.depth_sample_count,
                    self.depth_roi_fraction,
                    self.depth_minimum_valid_ratio,
                    self.depth_maximum_mad_m,
                    self.depth_maximum_spread_m
                )
            )
        ])

        self.run()
