"""Position-only IK (the analogue of MoveIt's position_only_ik) and its orientation cost."""
import math
import random
import sys

import numpy as np
import PyKDL
import urdf_parser_py.urdf


def build(robot, base, tip):
    joints = []
    current = tip
    while current != base:
        parent = next(j for j in robot.joints if j.child == current)
        joints.append(parent)
        current = parent.parent
    joints.reverse()
    chain = PyKDL.Chain()
    limits = []
    for joint in joints:
        if joint.type == 'revolute':
            axis = joint.axis
            kind = PyKDL.Joint(joint.name, PyKDL.Vector(0, 0, 0),
                               PyKDL.Vector(*axis), PyKDL.Joint.RotAxis)
            limits.append((joint.limit.lower, joint.limit.upper))
        else:
            kind = PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)
        origin = joint.origin
        rot = origin.rpy if origin is not None else [0, 0, 0]
        xyz = origin.xyz if origin is not None else [0, 0, 0]
        chain.addSegment(PyKDL.Segment(
            joint.child, kind,
            PyKDL.Frame(PyKDL.Rotation.RPY(*rot), PyKDL.Vector(*xyz))))
    return chain, limits


def main():
    robot = urdf_parser_py.urdf.URDF.from_xml_file(sys.argv[1])
    chain, limits = build(robot, sys.argv[2], sys.argv[3])
    dof = chain.getNrOfJoints()
    lower = np.array([low for low, _ in limits])
    upper = np.array([high for _, high in limits])
    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    jac = PyKDL.Jacobian(dof)

    def tip(q):
        frame = PyKDL.Frame()
        fk.JntToCart(q, frame)
        return frame

    def solve_position(target_p, seed, iterations=120, damping=0.05):
        q = PyKDL.JntArray(dof)
        for i in range(dof):
            q[i] = seed[i]
        for _ in range(iterations):
            frame = tip(q)
            error = np.array([
                target_p[0] - frame.p.x(),
                target_p[1] - frame.p.y(),
                target_p[2] - frame.p.z(),
            ])
            if float(np.linalg.norm(error)) < 1e-4:
                return q
            chain.JntToJac(q, jac)
            matrix = np.array([[jac[r, c] for c in range(dof)]
                               for r in range(3)])
            # damped least squares on the translational Jacobian only
            jjt = matrix @ matrix.T + damping * np.eye(3)
            step = matrix.T @ np.linalg.solve(jjt, error)
            values = np.array([q[i] for i in range(dof)]) + step
            values = np.clip(values, lower, upper)
            for i in range(dof):
                q[i] = float(values[i])
        return None

    rng = random.Random(13)
    requested = PyKDL.Rotation.RPY(0.0, math.pi / 2, 0.0)
    solved = 0
    residuals = []
    trials = 200
    for _ in range(trials):
        seed = PyKDL.JntArray(dof)
        for i in range(dof):
            seed[i] = rng.uniform(lower[i], upper[i])
        target_frame = tip(seed)
        target_p = (target_frame.p.x(), target_frame.p.y(), target_frame.p.z())
        q = solve_position(target_p, seed)
        if q is None:
            continue
        achieved = tip(q)
        error = math.sqrt((achieved.p.x() - target_p[0]) ** 2
                          + (achieved.p.y() - target_p[1]) ** 2
                          + (achieved.p.z() - target_p[2]) ** 2)
        if error > 1e-3:
            continue
        solved += 1
        residuals.append(math.degrees(
            (achieved.M.Inverse() * requested).GetRot().Norm()))
    residuals.sort()
    print('position-only IK: %d/%d solved (%.0f%%)'
          % (solved, trials, 100.0 * solved / trials))
    if residuals:
        print('orientation residual vs a requested 90 deg pitch: '
              'median %.1f deg, max %.1f deg'
              % (residuals[len(residuals) // 2], residuals[-1]))


if __name__ == '__main__':
    main()
