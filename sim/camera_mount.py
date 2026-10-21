"""Where the sim camera sits, in one place.

Gazebo's URDF->SDF conversion drops a ``<pose>`` written inside a ``<sensor>``
(gazebo reference="link"), so a sensor declared with an offset ends up at the
link origin with no rotation. Measured: the intended 0.155 m forward, 0.18 m up
and 26 deg down became (0, 0, 0.06) with identity rotation, which silently
undid every framing and geometry decision made on top of it.

The mount is therefore a real link with a fixed joint, whose origin the converter
does honour; the sensors hang on it with no pose of their own. The same numbers
drive the static TF (sim/camera_tf.py), so images, depth and TF agree by
construction.
"""
MOUNT_LINK = 'sim_camera_mount'
#: Where the RealSense sits on the real robot: between the two sandwich plates at
#: the front, low, about the cube's own height, and level. The upstream simulated
#: camera in practice_ws/worlds/*.world carries exactly this pose
#: (0.05915 0 0.0583 0 0 0), which is 0.059 m forward of base_link, at base_link's
#: height (base_link is 0.06 m above the floor, so -0.002 puts the lens just above
#: the cube) with no tilt. An earlier value here (0.155 forward, 0.18 up, 26 deg
#: down) was invented, and every measurement made on top of it described a camera
#: the robot does not have.
OFFSET_XYZ = (0.059, 0.0, -0.002)
OFFSET_RPY = (0.0, 0.0, 0.0)


def add_mount(urdf):
    """Attach the camera mount link to base_link in a URDF string."""
    if 'name="%s"' % MOUNT_LINK in urdf:
        return urdf
    xyz = ' '.join('%g' % value for value in OFFSET_XYZ)
    rpy = ' '.join('%g' % value for value in OFFSET_RPY)
    block = (
        '\n  <link name="%s"/>\n'
        '  <joint name="%s_joint" type="fixed">\n'
        '    <parent link="base_link"/>\n'
        '    <child link="%s"/>\n'
        '    <origin xyz="%s" rpy="%s"/>\n'
        '  </joint>\n' % (MOUNT_LINK, MOUNT_LINK, MOUNT_LINK, xyz, rpy)
    )
    if '</robot>' not in urdf:
        raise ValueError('unexpected description: no </robot>')
    return urdf.replace('</robot>', block + '</robot>', 1)
