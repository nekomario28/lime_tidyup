#!/usr/bin/env python3
"""Build the MuJoCo scene for the Lime grasp from the robot description.

The scene is what `mujoco_ros2_control` reads as physics (`mujoco_model` in the URDF's
ros2_control block); the URDF only supplies the ros2_control joint contract. Both come
from the same xacro, so the joint names agree by construction.

What the description needs before MuJoCo will take it (each verified against the real
files, see doc/research/2026-10-23-mujoco-lime-import.md):

1. **A floating root.** The URDF root is `base_footprint`, fixed to the world, so the
   robot could not be driven. A `world` link plus a `type="floating"` joint before it
   makes MuJoCo build a free body.
2. **Duplicate inertial.** `link6` carries two `<inertial>` elements -- a unit-sphere
   placeholder followed by the real one -- and MuJoCo refuses a repeated element.
3. **No `.dae`.** The one Collada mesh is the RealSense camera, which the arm does not
   use and MuJoCo cannot read.
4. **Meshes as files**, referenced from the output directory. The install space is a
   symlink install, so copies from there are dangling links -- take them from the source
   tree (`turtlebot3_ws/turtlebot3_lime/turtlebot3_lime_description/meshes`).
5. **The gripper is a mimic pair.** `gripper_right_joint` carries
   `<mimic joint="gripper_left_joint" multiplier="1"/>`; MuJoCo does not read that tag, so
   the scene reproduces it with a tendon plus an equality constraint, which is also what
   the hardware interface documents for parallel jaws. Only the left joint is controllable,
   matching `gripper_controller` (a `GripperActionController` on `gripper_left_joint`).

Actuators are position servos, because ros2_control declares position command interfaces
for the arm and the gripper. The wheels get none: `/cmd_vel` is served by the
`BaseVelocityPlugin`, which overrides the base's free-joint velocity directly (so a wheel
velocity command interface would have nothing to drive).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
GRIPPER_DRIVEN_JOINT = 'gripper_left_joint'
GRIPPER_MIMIC_JOINT = 'gripper_right_joint'
#: The gripper's travel, from the URDF limit ("lower -0.010 upper 0.019").
GRIPPER_RANGE = (-0.010, 0.019)
#: Torque the joints are allowed to apply. The URDF says `effort="1"` for every arm joint
#: -- a placeholder from the description, not a Dynamixel XM540's 10 N*m -- and MuJoCo
#: turns it into `actuatorfrcrange`, so a 1 N*m clamp leaves the wrist 0.3 rad short of its
#: command (measured). 10 N*m is the servo's own rating.
ARM_TORQUE = 10.0
#: Arm servo stiffness. The Gazebo arm is driven by a trajectory controller; here a
#: position servo is what the MuJoCo actuator provides, so kp has to be stiff enough to
#: hold the arm against gravity (measured: kp 60 left joint2 0.6 rad short of 1.0).
ARM_KP = 800.0
ARM_FORCE = 20.0
GRIPPER_KP = 400.0
GRIPPER_FORCE = 6.0
#: The cube, as sim/spawn_cube.py makes it.
CUBE_SIDE_M = 0.057
CUBE_MASS_KG = 0.08
#: base_footprint -> base_link, from the description's base_joint.
BASE_FOOTPRINT_Z = 0.060
#: The sticker grid the recognition expects, from the description's cube generator
#: (rubik_pose_cube: FACELET_SIZE_M / FACELET_GAP_M) and the face colours it labels by
#: (cube_pose_hypotheses.STANDARD_FACE_COLORS). A plain red box is enough for a physics
#: question but not for recognition: the face solve reads the sticker colours.
FACELET_SIZE_M = 0.0162
FACELET_GAP_M = 0.0012
FACE_COLORS_RGB = {'+x': (0.80, 0.05, 0.05), '-x': (0.90, 0.35, 0.03),
                   '+y': (0.05, 0.15, 0.80), '-y': (0.05, 0.55, 0.10),
                   '+z': (0.92, 0.92, 0.92), '-z': (0.90, 0.85, 0.10)}


def keep_single_inertial(text, link_name):
    """Keep the real inertial of a link that carries two (link6 has a placeholder)."""
    match = re.search(r'<link name="%s">.*?</link>' % link_name, text, re.S)
    if match is None:
        return text
    block = match.group(0)
    inertials = re.findall(r'<inertial>.*?</inertial>', block, re.S)
    if len(inertials) < 2:
        return text
    keep = max(inertials, key=len)
    fixed = block
    for candidate in inertials:
        fixed = fixed.replace(candidate, '', 1)
    fixed = re.sub(r'(<collision>.*?</collision>)', r'\1\n    ' + keep, fixed,
                   count=1, flags=re.S)
    return text.replace(block, fixed)


def patch_urdf(urdf_path, meshes_dir):
    """Apply the description fixes, in place."""
    path = Path(urdf_path)
    text = path.read_text()

    text = keep_single_inertial(text, 'link6')
    text = re.sub(r'<mesh filename="[^"]*\.dae"[^>]*/>', '', text)
    # absolute, because the scene is built and loaded in the same container while the
    # meshes stay where the description package keeps them. The URDF refers to
    # `package://.../meshes/...`, so the meshes directory replaces that prefix and the
    # package root replaces anything else under the package.
    meshes = Path(meshes_dir).resolve()
    text = text.replace('package://turtlebot3_lime_description/meshes/', str(meshes) + '/')
    text = text.replace('package://turtlebot3_lime_description/', str(meshes.parent) + '/')

    if 'name="mujoco_world"' not in text:
        text = text.replace(
            '  <link name="base_footprint"/>',
            '  <link name="mujoco_world"/>\n  <link name="base_footprint"/>', 1)
        text = text.replace(
            '  <joint name="base_joint" type="fixed">',
            '  <joint name="base_floating" type="floating">\n'
            '    <parent link="mujoco_world"/>\n'
            '    <child link="base_footprint"/>\n'
            '    <origin rpy="0 0 0" xyz="0 0 0"/>\n'
            '  </joint>\n'
            '  <joint name="base_joint" type="fixed">', 1)
    path.write_text(text)
    return meshes_dir


def build_mjcf(urdf_path, out_path):
    """Let MuJoCo convert, then keep the file it used."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(Path(urdf_path).resolve()))
    scratch = str(Path(out_path).with_suffix('.from_urdf.mjcf'))
    mujoco.mj_saveLastXML(scratch, model)
    return Path(scratch).read_text()


def strip_the_camera(mjcf_text):
    """Drop the RealSense camera's frames and geoms: unused by the arm, unreadable by us."""
    mjcf_text = re.sub(r'\s*<body name="camera[^"]*"[^>]*>.*?</body>', '', mjcf_text, flags=re.S)
    return mjcf_text


def relax_joint_torque(mjcf_text):
    """Give the arm joints a torque the servos actually have (see ARM_TORQUE)."""
    for name in ARM_JOINTS:
        mjcf_text = re.sub(r'(<joint name="%s"[^>]*?)actuatorfrcrange="[^"]*"'
                           % re.escape(name),
                           r'\1actuatorfrcrange="-%.1f %.1f"' % (ARM_TORQUE, ARM_TORQUE),
                           mjcf_text)
    return mjcf_text


def add_actuators(mjcf_text):
    """Position actuators, each named after the joint it drives.

    The name is the mapping: ros2_control joints are matched to MJCF actuators by name,
    and an actuator called `joint1_pos` driving `joint1` matches nothing -- measured, the
    hardware then reported every interface missing and failed to initialize while the
    scene itself loaded fine.
    """
    lines = ['    <position name="%s" joint="%s" kp="%.0f" dampratio="1" '
             'forcelimited="true" forcerange="-%.1f %.1f"/>'
             % (name, name, ARM_KP, ARM_FORCE, ARM_FORCE) for name in ARM_JOINTS]
    lines.append('    <position name="%s" tendon="gripper_split" kp="%.0f" '
                 'forcelimited="true" forcerange="-%.1f %.1f"/>'
                 % (GRIPPER_DRIVEN_JOINT, GRIPPER_KP, GRIPPER_FORCE, GRIPPER_FORCE))
    block = ('\n  <tendon>\n    <fixed name="gripper_split">\n'
             '      <joint joint="%s" coef="0.5"/>\n'
             '      <joint joint="%s" coef="-0.5"/>\n'
             '    </fixed>\n  </tendon>\n'
             '  <equality>\n    <joint joint1="%s" joint2="%s" polycoef="0 -1 0 0 0" '
             'solimp="0.95 0.99 0.001" solref="0.005 1"/>\n  </equality>\n'
             % (GRIPPER_DRIVEN_JOINT, GRIPPER_MIMIC_JOINT,
                GRIPPER_DRIVEN_JOINT, GRIPPER_MIMIC_JOINT))
    block += '\n  <actuator>\n' + '\n'.join(lines) + '\n  </actuator>\n'
    return mjcf_text.replace('</mujoco>', block + '</mujoco>')


def add_visual_settings(mjcf_text):
    """A lit environment, in neutral greys.

    Two measurements shaped this. MuJoCo's default render left the cube at saturation
    72/255 and value 101/255, below what the recognition's colour gate accepts (90 and 40),
    so the scene needs light. And the first sky I chose was blue (0.35 0.5 0.65), which is
    hue-close to the cube's blue face: the recognition's colour masks matched the whole
    background instead of the stickers, and the face quad came out 39 px wide where the
    sticker field is 33.9 px -- a range error of about 0.11 m. The Gazebo world this is
    compared against has a dark surround, so the sky here is neutral grey for the same
    reason.
    """
    visual = ('  <visual>\n'
              '    <headlight diffuse="0.8 0.8 0.8" ambient="0.35 0.35 0.35" '
              'specular="0.1 0.1 0.1"/>\n'
              '    <rgba haze="0.15 0.25 0.35 1"/>\n'
              '    <global azimuth="120" elevation="-20"/>\n'
              '  </visual>\n'
              '  <asset>\n'
              '    <texture type="skybox" builtin="gradient" rgb1="0.28 0.28 0.30" '
              'rgb2="0.10 0.10 0.11" width="512" height="3072"/>\n'
              '  </asset>\n')
    return mjcf_text.replace('<worldbody>', visual + '  <worldbody>', 1)


def add_floor_and_lighting(mjcf_text):
    """A lit, frictional floor: the cube's grasp depends on it, so this one is real."""
    scene = '''
    <light name="key" pos="1.5 0 3.0" dir="-0.3 0 -1" directional="true"
           diffuse="1 1 1" specular="0.2 0.2 0.2"/>
    <light name="fill" pos="0.5 -1.0 2.0" dir="0 0.3 -1" directional="true"
           diffuse="0.6 0.6 0.6" specular="0 0 0"/>
    <geom name="floor" type="plane" size="4 4 0.05" rgba="0.30 0.30 0.34 1"
          friction="1 0.005 0.0001"/>
'''
    return mjcf_text.replace('  <worldbody>\n', '  <worldbody>\n' + scene, 1)


def _sticker_geoms(centre_x):
    """54 thin plates (9 per face) carrying the face colours, visual-only.

    contype/conaffinity 0 keeps them out of the contact solver: the sticker field is what
    the camera reads, and 54 extra collision geoms on a 57 mm cube would put contact noise
    into the physics the same run is measuring.
    """
    pitch = FACELET_SIZE_M + FACELET_GAP_M
    half = CUBE_SIDE_M / 2.0
    offset = 0.0004
    lines = []
    for face, rgb in FACE_COLORS_RGB.items():
        axis = {'x': 0, 'y': 1, 'z': 2}[face[1]]
        sign = 1.0 if face[0] == '+' else -1.0
        for row in (-1, 0, 1):
            for column in (-1, 0, 1):
                position = [0.0, 0.0, 0.0]
                position[axis] = sign * (half + offset)
                others = [index for index in range(3) if index != axis]
                position[others[0]] = column * pitch
                position[others[1]] = row * pitch
                size = [FACELET_SIZE_M / 2.0, FACELET_SIZE_M / 2.0, FACELET_SIZE_M / 2.0]
                size[axis] = offset
                lines.append('      <geom type="box" pos="%.4f %.4f %.4f" '
                             'size="%.4f %.4f %.4f" rgba="%.2f %.2f %.2f 1" '
                             'contype="0" conaffinity="0"/>'
                             % (position[0], position[1], position[2],
                                size[0], size[1], size[2], *rgb))
    return lines


def add_cube(mjcf_text, cube_x, yaw_deg=0.0, with_stickers=True):
    import math

    half = CUBE_SIDE_M / 2.0
    half_yaw = math.radians(yaw_deg) / 2.0
    stickers = '\n'.join(_sticker_geoms(cube_x)) if with_stickers else ''
    scene = '''    <body name="cube" pos="%.4f 0 %.4f" quat="%.6f 0 0 %.6f">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="%.4f %.4f %.4f" rgba="0.03 0.03 0.03 1"
            mass="%.3f" friction="0.9 0.005 0.0001"/>
%s
    </body>
''' % (cube_x, half + 0.001, math.sin(half_yaw), math.cos(half_yaw),
       half, half, half, CUBE_MASS_KG, stickers)
    return mjcf_text.replace('  <worldbody>\n', '  <worldbody>\n' + scene, 1)


def add_camera(mjcf_text, offset_xyz, offset_rpy, name='sim_camera',
               resolution=(640, 480), fovy_deg=42.0):
    """A camera on the base at the mount sim/camera_mount.py defines.

    Two frames have to line up here, and that is the whole difficulty. Our optical frame
    looks along +z with y down (what the perception assumes and what the static TF
    describes), while MuJoCo's camera looks along -z with y up, so the camera's orientation
    is the optical one with its y and z axes flipped.

    It is attached **inside the base body**, not to the world: the mount offset is a
    base_link offset, and a camera in `worldbody` sits at world coordinates -- measured,
    that put it 6 cm below the base and produced a range error proportional to distance
    (0.8 cm at 0.4 m to 2.6 cm at 1.0 m) that looked exactly like a modelling error in the
    cube. The base_link offset is added to base_footprint's own 0.060 m.

    The CameraPlugin names its parameters after the MJCF camera, and takes the frame it
    advertises from its config, so the name here has to match sim/mj/plugins_mujoco.yaml.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import camera_tf

    mount = camera_tf.rpy_matrix(*offset_rpy)
    optical = camera_tf.matrix_product(mount, camera_tf.OPTICAL_FROM_MOUNT)
    # MuJoCo camera: -z forward, +y up; our optical frame: +z forward, +y down
    flip = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    rotation = camera_tf.matrix_product(optical, flip)
    quat = quaternion_from_matrix(rotation)
    x, y, z = offset_xyz
    scene = ('      <camera name="%s" mode="fixed" pos="%.4f %.4f %.4f" '
             'quat="%.6f %.6f %.6f %.6f" fovy="%.1f" resolution="%d %d"/>\n'
             % (name, x, y, z, *quat, fovy_deg, resolution[0], resolution[1]))
    return mjcf_text.replace('    <body name="base_footprint">',
                             '    <body name="base_footprint">\n' + scene, 1)


def quaternion_from_matrix(m):
    """xyzw quaternion of a rotation matrix (the largest-diagonal branch)."""
    import math

    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((m[2][1] - m[1][2]) / scale, (m[0][2] - m[2][0]) / scale,
                (m[1][0] - m[0][1]) / scale, 0.25 * scale)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (0.25 * scale, (m[0][1] + m[1][0]) / scale,
                (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale)
    if m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][1] + m[1][0]) / scale, 0.25 * scale,
                (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale)
    scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale,
            0.25 * scale, (m[1][0] - m[0][1]) / scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True, help='plain URDF to fix in place')
    parser.add_argument('--meshes', required=True, help='meshes directory to copy in')
    parser.add_argument('--out', required=True, help='scene MJCF to write')
    parser.add_argument('--cube-x', type=float, default=None,
                        help='place the cube this far ahead (m)')
    parser.add_argument('--cube-yaw-deg', type=float, default=0.0)
    parser.add_argument('--no-stickers', action='store_true',
                        help='plain cube (physics only) instead of the sticker grid')
    parser.add_argument('--camera', action='store_true',
                        help='add the base camera (needs camera_mount.OFFSET_*)')
    parser.add_argument('--camera-resolution', default='640x480',
                        help='camera resolution WxH; the Gazebo camera is 848x480 and the '
                             'face solve scales with the quad it is given')
    arguments = parser.parse_args()

    out = Path(arguments.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    patch_urdf(arguments.urdf, arguments.meshes)
    text = build_mjcf(arguments.urdf, out)
    text = strip_the_camera(text)
    text = add_visual_settings(text)
    text = add_floor_and_lighting(text)
    if arguments.cube_x is not None:
        text = add_cube(text, arguments.cube_x, arguments.cube_yaw_deg,
                        with_stickers=not arguments.no_stickers)
    if arguments.camera:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import camera_mount
        width, _, height = arguments.camera_resolution.partition('x')
        # the camera hangs on base_link, which the description puts 0.060 m above
        # base_footprint; the scene's camera lives inside that body
        x, y, z = camera_mount.OFFSET_XYZ
        body_offset = (x, y, z + BASE_FOOTPRINT_Z)
        text = add_camera(text, body_offset, camera_mount.OFFSET_RPY,
                          resolution=(int(width), int(height)))
        print('camera %s at %s rpy %s (optical frame sim_camera_color_optical_frame)'
              % (camera_mount.MOUNT_LINK, camera_mount.OFFSET_XYZ,
                 camera_mount.OFFSET_RPY))
    text = relax_joint_torque(text)
    text = add_actuators(text)
    out.write_text(text)
    print('wrote %s (%d chars)' % (out, len(text)))

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out))
    print('MuJoCo accepts it: nq=%d nv=%d nu=%d nbody=%d' % (model.nq, model.nv,
                                                             model.nu, model.nbody))
    print('bodies: %s' % [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                          for i in range(model.nbody)])


if __name__ == '__main__':
    main()
