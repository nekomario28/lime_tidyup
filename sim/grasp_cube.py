#!/usr/bin/env python3
"""Grasp the detected cube: IK, direct joint trajectory, close, lift.

Why this bypasses MoveIt's planner
----------------------------------
Measured in this sim:
  * a pose goal through move_group aborts for every target, including poses that
    /compute_ik solves in a single call, because pymoveit2 caps
    `allowed_planning_time` at 0.5 s and OMPL cannot sample a goal region for
    this arm in that window ("Insufficient states in sampleable goal region");
  * calling /compute_ik from inside cm1's actor callback fails ("wait set index
    too big"), so the step cannot live in a behaviour tree;
  * the collision-checked planner also rejects the floor-level grasp pose,
    though the fingers are ~1 cm thick and there is nothing there but the floor.

So: solve IK here, in a process of its own, and send the solution straight to
`arm_controller` as a FollowJointTrajectory goal. The controller is the same one
MoveIt would use; only the planner is skipped.
"""
from __future__ import annotations

import json
import sys
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
EE_TO_PADS_M = 0.0443      # end_effector_link sits this far past the finger pads
GRIPPER_CLOSED = -0.010
GRIPPER_EFFORT = 20.0
GRASP_ORIENTATION = (1.0, 0.0, 0.0, 0.0)   # palm down


class Grasper(Node):
    def __init__(self):
        super().__init__('grasp_cube')
        self.latest = None
        self.create_subscription(String, '/cube_pose_result', self.on_result, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')

    # -- perception -------------------------------------------------------
    def on_result(self, msg):
        self.latest = msg.data

    def wait_for_cube(self, timeout=20.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.latest:
                data = json.loads(self.latest)
                best = None
                for detection in data.get('detections', []):
                    pose = detection.get('pose_3d') or {}
                    if pose.get('position_m') and len(pose['position_m']) == 3:
                        if best is None or detection['confidence'] > best[0]:
                            best = (detection['confidence'], pose)
                if best is not None:
                    return best[1]
            time.sleep(0.2)
        return None

    def to_base_link(self, pose, timeout=10.0):
        frame = pose.get('frame_id')
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', frame, rclpy.time.Time())
                break
            except Exception:  # noqa: BLE001  (transform not published yet)
                time.sleep(0.2)
        else:
            raise SystemExit('no transform base_link <- %s' % frame)
        x, y, z = pose['position_m']
        q = transform.transform.rotation
        ux, uy, uz, uw = q.x, q.y, q.z, q.w
        cross = (uy * z - uz * y, uz * x - ux * z, ux * y - uy * x)
        dot = ux * x + uy * y + uz * z
        uu = ux * ux + uy * uy + uz * uz
        rotated = [2.0 * dot * u + (uw * uw - uu) * p + 2.0 * uw * c
                   for u, p, c in zip((ux, uy, uz), (x, y, z), cross)]
        t = transform.transform.translation
        return [rotated[0] + t.x, rotated[1] + t.y, rotated[2] + t.z]

    # -- motion -----------------------------------------------------------
    def solve_ik(self, position, quat_xyzw, avoid_collisions=False, timeout=5.0):
        if not self.ik_client.wait_for_service(timeout_sec=timeout):
            raise SystemExit('/compute_ik is not available')
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'arm'
        request.ik_request.ik_link_name = 'end_effector_link'
        request.ik_request.pose_stamped.header.frame_id = 'base_link'
        pose = request.ik_request.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = quat_xyzw
        request.ik_request.avoid_collisions = avoid_collisions
        request.ik_request.timeout.sec = 2
        response = self.ik_client.call(request)
        if response is None or response.error_code.val != 1:
            self.get_logger().warn(
                'ik rejected %s (code %s)'
                % ([round(v, 3) for v in position],
                   None if response is None else response.error_code.val))
            return None
        by_name = dict(zip(response.solution.joint_state.name,
                           response.solution.joint_state.position))
        return [by_name[name] for name in ARM_JOINTS]

    def fingertip_midpoint(self, timeout=5.0):
        """Where the pads actually are, in base_link.

        The pads sit 4.43 cm past end_effector_link, but with position-only IK the
        achieved tool direction is whatever the solver picked, so subtracting a
        fixed vertical offset aims the wrist, not the fingers. Measured here
        instead, from TF.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            positions = []
            for link in ('gripper_left_link', 'gripper_right_link'):
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'base_link', link, rclpy.time.Time())
                except Exception:  # noqa: BLE001
                    break
                t = transform.transform.translation
                positions.append([t.x, t.y, t.z])
            if len(positions) == 2:
                return [(positions[0][i] + positions[1][i]) / 2.0
                        for i in range(3)]
            time.sleep(0.2)
        return None

    def send_arm(self, joints, seconds=5.0):
        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            raise SystemExit('/arm_controller is not available')
        goal = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = list(ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joints]
        point.velocities = [0.0] * len(ARM_JOINTS)
        point.time_from_start = Duration(seconds=seconds).to_msg()
        trajectory.points = [point]
        goal.trajectory = trajectory
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise SystemExit('arm goal rejected')
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=seconds + 20.0)
        return result.result() is not None

    def send_gripper(self, position, effort=GRIPPER_EFFORT):
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise SystemExit('/gripper_controller is not available')
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(effort)
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise SystemExit('gripper goal rejected')
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=20.0)
        return True


def main():
    lift_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    rclpy.init()
    node = Grasper()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    pose = node.wait_for_cube()
    if pose is None:
        raise SystemExit('no cube pose with a 3D position')
    point = node.to_base_link(pose)
    node.get_logger().info('cube in base_link: %.3f %.3f %.3f' % tuple(point))

    # Aim the pads at the cube centre: end_effector_link is EE_TO_PADS_M further
    # along the tool axis, so its target sits that much higher.
    grasp = [point[0], point[1], point[2] + EE_TO_PADS_M]
    approach = [point[0], point[1], point[2] + EE_TO_PADS_M + 0.12]

    # Opening the fingers before the descent matters: arriving with them shut
    # pushes the cube instead of straddling it.
    node.get_logger().info('opening the gripper')
    node.send_gripper(0.019)
    time.sleep(1.0)

    # Close the loop on the pads themselves: solve, move, measure where the pads
    # ended up, correct the end-effector target by that error, repeat.
    # Solve, move, measure the pads, correct. Two problems are handled here:
    # the measured centre can be out of the arm's reach (KDL returns -31), and
    # with position-only IK the pads do not sit a fixed distance below
    # end_effector_link, so the offset has to be measured rather than assumed.
    target = list(grasp)
    joints = None
    for pull in range(4):
        joints = node.solve_ik(target, GRASP_ORIENTATION)
        if joints is not None:
            break
        target[0] -= 0.03
        node.get_logger().warn('out of reach, target pulled to %s'
                               % [round(v, 3) for v in target])
    if joints is None:
        raise SystemExit('ik found no solution within reach (cube at %s)'
                         % [round(v, 3) for v in grasp])

    for attempt in range(3):
        if attempt:
            joints = node.solve_ik(target, GRASP_ORIENTATION)
            if joints is None:
                raise SystemExit('ik lost the corrected target %s'
                                 % [round(v, 3) for v in target])
        node.get_logger().info('attempt %d: ee target %s joints %s'
                               % (attempt + 1, [round(v, 3) for v in target],
                                  [round(v, 3) for v in joints]))
        node.send_arm(joints, seconds=5.0)
        time.sleep(1.0)
        pads = node.fingertip_midpoint()
        if pads is None:
            raise SystemExit('no fingertip transform after the move')
        error = [point[i] - pads[i] for i in range(3)]
        node.get_logger().info('pads at %s, error %.3f m'
                               % ([round(v, 3) for v in pads],
                                  max(abs(v) for v in error)))
        if max(abs(v) for v in error) < 0.010:
            break
        target = [target[i] + error[i] for i in range(3)]
    time.sleep(0.5)

    node.get_logger().info('closing the gripper')
    node.send_gripper(GRIPPER_CLOSED)
    time.sleep(2.5)

    lifted = list(joints)
    lifted[1] += 0.35          # shoulder up
    lifted[2] -= 0.45          # elbow compensates, keeping the wrist above
    node.get_logger().info('lifting')
    node.send_arm(lifted, seconds=lift_seconds)
    time.sleep(2.0)
    print('grasp attempt finished')


if __name__ == '__main__':
    main()
