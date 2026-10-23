#!/usr/bin/env python3
"""Step the converted Lime scene and report whether the physics is sane.

A smoke test, not a controller: it settles the scene, commands the arm forward, and
prints what it can measure. It exists because the question the port is for -- do the
fingers hold the cube? -- needs a simulator that does not diverge, and "it loaded" is not
the same as "it steps".

Verified with this (MuJoCo 3.17, CPU): the scene settles with the cube at z = 0.0284 (its
half side), max |qvel| = 0 with no motion commanded, the six arm joints track their
commands under gravity, and the fingertip bodies can be read out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
GRIPPER_JOINTS = ('gripper_left_joint', 'gripper_right_joint')
PALM_BODIES = ('gripper_left_link', 'gripper_right_link')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--seconds', type=float, default=1.5)
    arguments = parser.parse_args()

    scene = Path(arguments.scene).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    print('scene: nq=%d nv=%d nu=%d nbody=%d' % (model.nq, model.nv, model.nu,
                                                 model.nbody))

    for _ in range(500):
        mujoco.mj_step(model, data)
    print('settled: cube z=%.4f  max|qvel|=%.4f'
          % (data.body('cube').xpos[2], np.abs(data.qvel).max()))

    for _ in range(int(arguments.seconds / model.opt.timestep)):
        for index, value in enumerate([0.0, 1.0, -1.2, 0.8, 0.0, 0.0]):
            data.ctrl[index] = value
        for offset, name in enumerate(GRIPPER_JOINTS):
            data.ctrl[len(ARM_JOINTS) + offset] = 0.015
        mujoco.mj_step(model, data)

    print('arm joints: %s' % [round(float(data.joint(name).qpos[0]), 3)
                              for name in ARM_JOINTS])
    print('end effector: %s' % np.round(data.body('link7').xpos, 4))
    for name in PALM_BODIES:
        print('%-20s %s' % (name, np.round(data.body(name).xpos, 4)))
    print('gripper joints: %s' % [round(float(data.joint(name).qpos[0]), 4)
                                  for name in GRIPPER_JOINTS])
    print('cube: %s   max|qvel|=%.3f' % (np.round(data.body('cube').xpos, 4),
                                         np.abs(data.qvel).max()))


if __name__ == '__main__':
    main()
