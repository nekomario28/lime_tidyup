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


def add_floor_and_lighting(mjcf_text):
    """A lit, frictional floor: the cube's grasp depends on it, so this one is real."""
    scene = '''
    <light name="key" pos="1.5 0 2.5" dir="-0.4 0 -1" directional="true"
           diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="4 4 0.05" rgba="0.45 0.45 0.5 1"
          friction="1 0.005 0.0001"/>
'''
    return mjcf_text.replace('  <worldbody>\n', '  <worldbody>\n' + scene, 1)


def add_cube(mjcf_text, cube_x, yaw_deg=0.0):
    import math

    half = CUBE_SIDE_M / 2.0
    half_yaw = math.radians(yaw_deg) / 2.0
    scene = '''    <body name="cube" pos="%.4f 0 %.4f" quat="%.6f 0 0 %.6f">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="%.4f %.4f %.4f" rgba="0.85 0.12 0.12 1"
            mass="%.3f" friction="0.9 0.005 0.0001"/>
    </body>
''' % (cube_x, half + 0.001, math.sin(half_yaw), math.cos(half_yaw),
       half, half, half, CUBE_MASS_KG)
    return mjcf_text.replace('  <worldbody>\n', '  <worldbody>\n' + scene, 1)


def add_camera(mjcf_text, offset_xyz, offset_rpy):
    """A camera on the base at the mount sim/camera_mount.py defines.

    The CameraPlugin publishes whatever camera the MJCF names, so this has to match the
    mount the static TF describes or the perception would be given a pose for one lens
    and images from another -- the failure the mount link exists to prevent.
    """
    import math

    roll, pitch, yaw = offset_rpy
    x, y, z = offset_xyz
    # the optical frame looks along +z with y down; the camera body here is the mount
    # frame turned into it, so the same rotation as sim/camera_tf.py is applied
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    quat = (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)
    scene = ('    <camera name="sim_camera" pos="%.4f %.4f %.4f" quat="%.6f %.6f %.6f '
             '%.6f" fovy="42"/>\n' % (x, y, z, *quat))
    return mjcf_text.replace('  </worldbody>', scene + '  </worldbody>', 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True, help='plain URDF to fix in place')
    parser.add_argument('--meshes', required=True, help='meshes directory to copy in')
    parser.add_argument('--out', required=True, help='scene MJCF to write')
    parser.add_argument('--cube-x', type=float, default=None,
                        help='place the cube this far ahead (m)')
    parser.add_argument('--cube-yaw-deg', type=float, default=0.0)
    parser.add_argument('--camera', action='store_true',
                        help='add the base camera (needs camera_mount.OFFSET_*)')
    arguments = parser.parse_args()

    out = Path(arguments.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    patch_urdf(arguments.urdf, arguments.meshes)
    text = build_mjcf(arguments.urdf, out)
    text = strip_the_camera(text)
    text = add_floor_and_lighting(text)
    if arguments.cube_x is not None:
        text = add_cube(text, arguments.cube_x, arguments.cube_yaw_deg)
    if arguments.camera:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import camera_mount
        text = add_camera(text, camera_mount.OFFSET_XYZ, camera_mount.OFFSET_RPY)
        print('camera: %s at %s rpy %s' % (camera_mount.MOUNT_LINK,
                                           camera_mount.OFFSET_XYZ,
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
