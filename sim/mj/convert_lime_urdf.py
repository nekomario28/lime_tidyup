#!/usr/bin/env python3
"""Turn the Lime description into a MuJoCo scene.

The Gazebo stack cannot answer the question this exists for -- whether the fingers hold
the cube -- because its contact solver diverges at the grasp (measured: with the
grasp plugin off, the cube was shoved 1.1 m along the floor and the robot itself was
launched to x = -35 m). A different engine is the only way to measure that.

What the conversion needs, each verified against the real description:

1. ``xacro use_sim:=false`` gives a clean URDF (no gazebo/ros2_control tags), with the
   meshes as ``package://`` references and a ``scale="0.001"`` (the STLs are in mm).
2. **link6 carries two ``<inertial>`` elements** -- a placeholder with a unit-sphere
   inertia followed by the real one -- and MuJoCo refuses a repeated element. The longer
   (real) block is kept.
3. The meshes must be real files: the install space is a symlink install, so copying it
   gives dangling links. Take them from the source tree instead.
4. The one ``.dae`` mesh is the RealSense camera, which the arm does not use and MuJoCo
   cannot read; it is dropped.

Run (meshes are inside the container, from the source tree rather than the install space):

    docker cp <lime_container>:/root/turtlebot3_ws/turtlebot3_lime/\\
turtlebot3_lime_description/meshes /tmp/lime_mj/meshes
    python3 sim/mj/convert_lime_urdf.py --urdf lime.urdf --meshes /tmp/lime_mj/meshes \\
        --out /tmp/lime_mj/scene.mjcf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

#: Position actuators for the arm and the gripper. The arm is a servo here (kp/kv), not
#: a trajectory controller: what the grasp needs is a joint that lags when it is blocked,
#: because that lag is how the touch sweep decides the jaws have met the cube.
ARM_GAINS = {'kp': 200.0, 'kv': 20.0, 'force': 20.0}
GRIPPER_GAINS = {'kp': 60.0, 'kv': 2.0, 'force': 6.0}
ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
GRIPPER_JOINTS = ('gripper_left_joint', 'gripper_right_joint')

#: The cube as sim/spawn_cube.py makes it: 57 mm, 0.08 kg.
CUBE_SIDE_M = 0.057
CUBE_MASS_KG = 0.08


def read_urdf(path, meshes_dir):
    text = Path(path).read_text()
    text = text.replace('package://turtlebot3_lime_description/', '')
    text = re.sub(r'<mesh filename="[^"]*\.dae"/>', '', text)
    text = _keep_single_inertial(text, 'link6')
    Path(path).write_text(text)
    return meshes_dir


def _keep_single_inertial(text, link_name):
    """Drop the duplicate inertial (the placeholder) from one link."""
    link = re.search(r'<link name="%s">.*?</link>' % link_name, text, re.S)
    if link is None:
        return text
    block = link.group(0)
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


def add_actuators(mjcf_text):
    lines = []
    for name in ARM_JOINTS:
        lines.append('    <position name="%s_pos" joint="%s" kp="%.0f" kv="%.0f" '
                     'forcelimited="true" forcerange="-%.1f %.1f"/>'
                     % (name, name, ARM_GAINS['kp'], ARM_GAINS['kv'],
                        ARM_GAINS['force'], ARM_GAINS['force']))
    for name in GRIPPER_JOINTS:
        lines.append('    <position name="%s_pos" joint="%s" kp="%.0f" kv="%.0f" '
                     'forcelimited="true" forcerange="-%.1f %.1f"/>'
                     % (name, name, GRIPPER_GAINS['kp'], GRIPPER_GAINS['kv'],
                        GRIPPER_GAINS['force'], GRIPPER_GAINS['force']))
    block = '\n  <actuator>\n' + '\n'.join(lines) + '\n  </actuator>\n'
    return mjcf_text.replace('</mujoco>', block + '</mujoco>')


def add_floor_and_cube(mjcf_text, cube_x):
    half = CUBE_SIDE_M / 2.0
    scene = '''
    <geom name="floor" type="plane" size="4 4 0.05" rgba="0.5 0.5 0.5 1"
          friction="1 0.005 0.0001"/>
    <body name="cube" pos="%.3f 0 %.4f">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="%.4f %.4f %.4f" rgba="0.8 0.1 0.1 1"
            mass="%.3f" friction="0.9 0.005 0.0001"/>
    </body>
''' % (cube_x, half, half, half, half, CUBE_MASS_KG)
    return mjcf_text.replace('  <worldbody>\n', '  <worldbody>\n' + scene, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True,
                        help='plain URDF (xacro use_sim:=false) to fix in place')
    parser.add_argument('--meshes', required=True,
                        help='directory the mesh files are referenced from')
    parser.add_argument('--out', required=True, help='MJCF scene to write')
    parser.add_argument('--cube-x', type=float, default=0.24,
                        help='where the cube sits on the floor (m, base frame)')
    arguments = parser.parse_args()

    meshes = Path(arguments.meshes).resolve()
    read_urdf(arguments.urdf, meshes)

    import mujoco  # imported here so --help works without mujoco installed

    model = mujoco.MjModel.from_xml_path(str(Path(arguments.urdf).resolve()))
    scratch = str(Path(arguments.out).with_suffix('.converted.mjcf'))
    mujoco.mj_saveLastXML(scratch, model)

    # MuJoCo resolves mesh files relative to the MJCF, so the scene is written next to
    # the meshes directory and references them by name.
    text = Path(scratch).read_text()
    text = text.replace(str(meshes) + '/', '')
    text = add_floor_and_cube(text, arguments.cube_x)
    text = add_actuators(text)
    Path(arguments.out).write_text(text)
    print('wrote %s (%d chars); meshes are referenced relative to %s'
          % (arguments.out, len(text), Path(arguments.out).parent))
    print('note: run it from the directory that holds the meshes directory, or make '
          'the mesh paths absolute')


if __name__ == '__main__':
    main()
