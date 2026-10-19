"""Measure IK on the Lime arm chain: 6 revolute joints, base link1, tip end_effector_link.

Evidence tool for ../2026-10-20-ik-usage.md. It builds the arm chain from the
URDF with PyKDL, the same solver family the MoveIt configuration names
(kdl_kinematics_plugin), and reports:

* the chain and its joint limits,
* FK->IK round trips (can IK drive the arm at all),
* the reachable tip envelope sampled from legal configurations,
* IK success on known-reachable tips with and without a near seed,
* the orientation residual when only position is solved, which is what
  `position_only_ik: True` means for the runtime.

It is not the runtime: no move_group, no controllers, no Gazebo. It answers
"does the geometry admit these targets", not "does the sim execute them".
"""
import itertools
import math

import PyKDL
import urdf_parser_py.urdf


def load_urdf(path):
    return urdf_parser_py.urdf.URDF.from_xml_file(path)


def build_chain(robot, base, tip):
    """Walk the URDF parent chain from ``tip`` back to ``base``."""
    chain_joints = []
    current = tip
    while current != base:
        parent = None
        for joint in robot.joints:
            if joint.child == current:
                parent = joint
                break
        if parent is None:
            raise SystemExit(f'no path from {base} to {tip}: stuck at {current}')
        chain_joints.append(parent)
        current = parent.parent
    chain_joints.reverse()
    return chain_joints


def main():
    import sys
    robot = load_urdf(sys.argv[1])
    base, tip = sys.argv[2], sys.argv[3]
    chain_joints = build_chain(robot, base, tip)
    moves = [j.name for j in chain_joints if j.type in ('revolute', 'prismatic')]
    fixed = [j.name for j in chain_joints if j.type == 'fixed']
    print('chain:', ' -> '.join([base] + [j.child for j in chain_joints]))
    print('moveable joints:', moves)
    print('fixed joints:', fixed)

    chain = PyKDL.Chain()
    limits = []
    for joint in chain_joints:
        kind = PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)
        if joint.type == 'revolute':
            axis = joint.axis
            kind = PyKDL.Joint(
                joint.name, PyKDL.Vector(0, 0, 0), PyKDL.Vector(*axis),
                PyKDL.Joint.RotAxis,
            )
            limits.append((joint.limit.lower, joint.limit.upper))
        elif joint.type == 'prismatic':
            axis = joint.axis
            kind = PyKDL.Joint(
                joint.name, PyKDL.Vector(0, 0, 0), PyKDL.Vector(*axis),
                PyKDL.Joint.TransAxis,
            )
            limits.append((joint.limit.lower, joint.limit.upper))
        origin = joint.origin
        rot = origin.rpy if origin is not None else [0, 0, 0]
        xyz = origin.xyz if origin is not None else [0, 0, 0]
        frame = PyKDL.Frame(
            PyKDL.Rotation.RPY(*rot), PyKDL.Vector(*xyz)
        )
        chain.addSegment(PyKDL.Segment(joint.child, kind, frame))

    dof = chain.getNrOfJoints()
    print('chain DOF:', dof)
    print('joint limits (rad or m):',
          [(round(low, 3), round(high, 3)) for low, high in limits])
    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    # The MoveIt KDL plugin uses a joint-limit aware position solver, so measure
    # with the same kind of solver rather than an unconstrained LMA.
    ik_vel = PyKDL.ChainIkSolverVel_pinv(chain)
    lower = PyKDL.JntArray(dof)
    upper = PyKDL.JntArray(dof)
    for index, (low, high) in enumerate(limits):
        lower[index] = low
        upper[index] = high
    ik = PyKDL.ChainIkSolverPos_NR_JL(
        chain, lower, upper, fk, ik_vel, 200, 1e-6
    )

    home = PyKDL.JntArray(dof)
    for index in range(dof):
        home[index] = 0.0
    rest = PyKDL.Frame()
    fk.JntToCart(home, rest)
    print('FK at zero: p=(%.4f, %.4f, %.4f)'
          % (rest.p.x(), rest.p.y(), rest.p.z()))

    # Round trip: pose the arm at a legal configuration, then ask IK for that
    # exact tip position. This is the honest "can IK drive this arm" test.
    round_trips = 0
    round_trip_ok = 0
    for trial in range(40):
        config = PyKDL.JntArray(dof)
        for index in range(dof):
            low, high = limits[index]
            frac = ((trial * 7 + index * 11) % 20) / 19.0
            config[index] = low + (high - low) * frac
        pose = PyKDL.Frame()
        fk.JntToCart(config, pose)
        solution = PyKDL.JntArray(dof)
        round_trips += 1
        if ik.CartToJnt(config, pose, solution) < 0:
            continue
        achieved = PyKDL.Frame()
        fk.JntToCart(solution, achieved)
        if (achieved.p - pose.p).Norm() <= 1e-3:
            round_trip_ok += 1
    print('FK->IK round trips recovered: %d/%d (%.0f%%)'
          % (round_trip_ok, round_trips,
             100.0 * round_trip_ok / max(1, round_trips)))

    # Reachable envelope: FK over legal configurations, then ask IK for a
    # sample of those very tips. This separates "arm cannot reach it" from
    # "solver failed".
    import random
    rng = random.Random(7)
    tips = []
    for _ in range(2000):
        config = PyKDL.JntArray(dof)
        for index in range(dof):
            low, high = limits[index]
            config[index] = rng.uniform(low, high)
        pose = PyKDL.Frame()
        fk.JntToCart(config, pose)
        tips.append((pose.p.x(), pose.p.y(), pose.p.z()))
    xs = [t[0] for t in tips]
    ys = [t[1] for t in tips]
    zs = [t[2] for t in tips]
    radii = [math.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2) for t in tips]
    print('reachable tip envelope: x[%.3f, %.3f] y[%.3f, %.3f] z[%.3f, %.3f]'
          % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
    print('max tip radius from link1: %.3f m' % max(radii))

    reachable_tries = reachable_ok = 0
    for _ in range(100):
        x, y, z = rng.choice(tips)
        target = PyKDL.Frame()
        target.p = PyKDL.Vector(x, y, z)
        solution = PyKDL.JntArray(dof)
        reachable_tries += 1
        solved = False
        for attempt in range(6):
            seed = PyKDL.JntArray(dof)
            for index in range(dof):
                low, high = limits[index]
                seed[index] = rng.uniform(low, high)
            candidate = PyKDL.JntArray(dof)
            if ik.CartToJnt(seed, target, candidate) < 0:
                continue
            achieved = PyKDL.Frame()
            fk.JntToCart(candidate, achieved)
            if (achieved.p - target.p).Norm() <= 1e-3:
                solved = True
                break
        if solved:
            reachable_ok += 1
    print('IK on 100 known-reachable tips: %d/100 solved' % reachable_ok)

    # sweep the forward workspace (the arm works in front of / above link1)
    attempts = successes = within_limits = 0
    worst_reach = 0.0
    solutions = []
    for x, y, z in itertools.product(
            (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25),
            (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25),
            (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)):
        target = PyKDL.Frame()
        target.p = PyKDL.Vector(x, y, z)
        attempts += 1
        # MoveIt's KDL plugin retries from several seeds; a single home seed is
        # not enough for a 6-DOF Newton solver, so measure with restarts.
        result = -1
        solution = PyKDL.JntArray(dof)
        seeds = [home]
        for attempt in range(4):
            seed = PyKDL.JntArray(dof)
            for index in range(dof):
                low, high = limits[index]
                frac = ((index * 7 + attempt * 13) % 10) / 10.0
                seed[index] = low + (high - low) * frac
            seeds.append(seed)
        for seed in seeds:
            candidate = PyKDL.JntArray(dof)
            if ik.CartToJnt(seed, target, candidate) >= 0:
                result = 0
                solution = candidate
                break
        if result < 0:
            continue
        within = all(limits[i][0] - 1e-6 <= solution[i] <= limits[i][1] + 1e-6
                     for i in range(dof))
        achieved = PyKDL.Frame()
        fk.JntToCart(solution, achieved)
        error = (achieved.p - target.p).Norm()
        if error > 1e-3:
            continue
        successes += 1
        worst_reach = max(worst_reach, math.sqrt(x * x + y * y + z * z))
        if within:
            within_limits += 1
            solutions.append((x, y, z, error))
    print('targets tried:', attempts, 'solved within 1 mm:', successes,
          f'({100.0 * successes / attempts:.0f}%)')
    print('of those, inside joint limits:', within_limits,
          f'({100.0 * within_limits / attempts:.0f}% of the grid)')
    print('max target distance solved:', round(worst_reach, 3), 'm')

    # orientation is not constrained by a position IK solver: show the gap
    target = PyKDL.Frame()
    target.p = PyKDL.Vector(0.20, 0.0, 0.10)
    target.M = PyKDL.Rotation.RPY(0.0, math.pi / 2, 0.0)
    solution = PyKDL.JntArray(dof)
    result = ik.CartToJnt(home, target, solution)
    achieved = PyKDL.Frame()
    fk.JntToCart(solution, achieved)
    delta = achieved.M.Inverse() * target.M
    angle = math.degrees(delta.GetRot().Norm())
    print('position residual: %.6f m' % (achieved.p - target.p).Norm())
    print('orientation residual for a requested 90 deg pitch: %.2f deg' % angle)

    # Orientation residual across the solved grid: a position-only solver has no
    # reason to match the requested rotation.
    residuals = []
    for x, y, z, _ in solutions:
        target = PyKDL.Frame()
        target.p = PyKDL.Vector(x, y, z)
        target.M = PyKDL.Rotation.RPY(0.0, math.pi / 2, 0.0)
        solution = PyKDL.JntArray(dof)
        if ik.CartToJnt(home, target, solution) < 0:
            continue
        achieved = PyKDL.Frame()
        fk.JntToCart(solution, achieved)
        residuals.append(math.degrees(
            (achieved.M.Inverse() * target.M).GetRot().Norm()
        ))
    if residuals:
        residuals.sort()
        print('orientation residual over %d solved targets: median %.1f deg,'
              ' max %.1f deg' % (len(residuals),
                                 residuals[len(residuals) // 2],
                                 residuals[-1]))


if __name__ == '__main__':
    main()
