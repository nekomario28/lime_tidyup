import math

from cm1.lib.cube_pose import transform_pose


def test_transform_pose_applies_frame_rotation_and_translation():
    position, orientation = transform_pose(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 2.0, 3.0),
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
    )

    assert all(math.isclose(actual, expected, abs_tol=1e-9)
               for actual, expected in zip(position, (1.0, 3.0, 3.0)))
    assert all(math.isclose(actual, expected, abs_tol=1e-9)
               for actual, expected in zip(
                   orientation,
                   (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
               ))
