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

import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, Image, JointState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
EE_TO_PADS_M = 0.0443      # end_effector_link sits this far past the finger pads
#: Extra distance to leave between the pads and the cube centre. Zero: measured,
#: the pads (gripper_left/right_link origins) are the contact surfaces and they
#: stop 7.7 cm behind the cube, so leaving a "finger length" made them close on
#: air behind it. The gap has to be driven in full with the arm pose held, which
#: is safe because the open jaws are 8 cm wide and the cube is 5.7 cm.
FINGER_REACH_M = 0.0
GRIPPER_CLOSED = -0.010
GRIPPER_EFFORT = 6.0
# Measured with the sim's own IK service at the post-approach pose (cube 0.31 m
# ahead, on the floor): a palm-down grasp (roll 180) has NO solution anywhere
# within reach, and so does a horizontal front grasp (pitch 90). Pitch 105-120
# degrees -- palm angled forward-down, fingers straddling the cube from behind --
# does solve, at both 4.4 cm and 7 cm behind the cube. That is the reachable
# version of a front grasp for this arm.
#: (pitch degrees, height offset from the measured cube centre).
#: Measured with the sim's IK service at 0.28 m: pitch 120 with the centre is the
#: only solution at that distance, but aiming 3 cm *lower* -- the cube's lower
#: half, which is where the pads can physically reach -- opens pitch 120-150.
#: The pairs are tried in this order, centre first because it is the honest aim.
#: The low aim comes first: measured pads sit 4.4 cm above the cube centre in the
#: centre-aim pose, so the jaws close over the cube's top instead of around it.
GRASP_CANDIDATES = (
    (120.0, -0.03),
    (135.0, -0.03),
    (150.0, -0.03),
    (120.0, 0.0),
)


class Grasper(Node):
    def __init__(self):
        super().__init__('grasp_cube')
        self.bridge = CvBridge()
        overlay_dir = os.environ.get('CCCSIM_OVERLAY_DIR', '')
        self.overlay_dir = overlay_dir
        self.overlay_index = 0
        self.latest = None
        self.create_subscription(String, '/cube_pose_result', self.on_result, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gripper_position = None
        self.latest_image = None
        self.camera_info = None
        self.overlay_pub = self.create_publisher(Image, '/grasp_debug_image', 1)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self.on_image, 1)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self.on_camera_info, 1)
        self.create_subscription(JointState, '/joint_states',
                                 self.on_joints, 10)

    def on_image(self, msg):
        self.latest_image = msg

    def on_camera_info(self, msg):
        self.camera_info = msg

    def draw_overlay(self, note, target=None):
        """Draw what perception recognised and where the grasp is aiming.

        Everything drawn comes from recognition and TF: the box and the 3-D centre
        from /cube_pose_result, the aim point by projecting the base_link target
        through the camera. Nothing here reads ground truth from Gazebo.
        """
        image_msg = self.latest_image
        info = self.camera_info
        if image_msg is None or info is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as error:  # noqa: BLE001
            self.get_logger().warn('overlay skipped: %s' % error)
            return
        frame = frame.copy()

        detection = None
        if self.latest:
            data = json.loads(self.latest)
            best = 0.0
            for candidate in data.get('detections', []):
                if candidate.get('confidence', 0.0) > best:
                    best = candidate['confidence']
                    detection = candidate
        if detection is not None:
            box = detection.get('box_xyxy') or []
            if len(box) == 4:
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 190, 255), 2)
                cv2.putText(frame, 'cube %.2f' % detection.get('confidence', 0.0),
                            (x0, max(14, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 190, 255), 1, cv2.LINE_AA)
            pose3d = detection.get('pose_3d') or {}
            status = detection.get('pose_evidence', {}).get('pose_status', '')
            cv2.putText(frame, 'pose: %s' % status, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)

        if target is not None:
            point = self.project_to_pixel(target, image_msg.header.frame_id)
            if point is not None:
                u, v = (int(round(value)) for value in point)
                cv2.drawMarker(frame, (u, v), (60, 60, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.circle(frame, (u, v), 6, (60, 60, 255), 2)
                cv2.putText(frame, 'grasp target', (u + 10, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1,
                            cv2.LINE_AA)
        cv2.putText(frame, note, (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        message.header = image_msg.header
        self.overlay_pub.publish(message)
        if self.overlay_dir:
            path = '%s/%05d.jpg' % (self.overlay_dir, self.overlay_index)
            self.overlay_index += 1
            cv2.imwrite(path, frame)

    def project_to_pixel(self, point_base, image_frame):
        """Project a base_link point into the camera image."""
        info = self.camera_info
        if info is None:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                image_frame, 'base_link', rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        q = transform.transform.rotation
        t = transform.transform.translation
        x, y, z = point_base
        ux, uy, uz, uw = q.x, q.y, q.z, q.w
        cross = (uy * z - uz * y, uz * x - ux * z, ux * y - uy * x)
        dot = ux * x + uy * y + uz * z
        uu = ux * ux + uy * uy + uz * uz
        rotated = [2.0 * dot * u + (uw * uw - uu) * p + 2.0 * uw * c
                   for u, p, c in zip((ux, uy, uz), (x, y, z), cross)]
        camera_point = [rotated[i] + (t.x, t.y, t.z)[i] for i in range(3)]
        if camera_point[2] <= 0.01:
            return None
        k = info.k
        u = k[0] * camera_point[0] / camera_point[2] + k[2]
        v = k[4] * camera_point[1] / camera_point[2] + k[5]
        return u, v

    def on_joints(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name == 'gripper_left_joint':
                self.gripper_position = float(position)
                return

    def drive_forward(self, metres, speed=0.05):
        """Creep the base forward; the tree stops short of the arm's real reach."""
        seconds = abs(metres) / speed
        message = Twist()
        message.linear.x = speed if metres > 0 else -speed
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.cmd_pub.publish(message)
            time.sleep(0.1)
        self.cmd_pub.publish(Twist())
        time.sleep(1.0)

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

    def close_slowly(self, settle=1.6, step=0.003, dwell=0.35):
        """Close the jaws in small steps.

        Jumping straight to GRIPPER_CLOSED makes the fingers arrive with enough
        speed that the physics engine resolves the contact after they are already
        through the cube, so nothing is gripped. Ramping the commanded position
        gives each contact a few physics steps to form.
        """
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise SystemExit('/gripper_controller is not available')
        position = 0.019
        while position > GRIPPER_CLOSED:
            position = max(GRIPPER_CLOSED, position - step)
            goal = GripperCommand.Goal()
            goal.command.position = float(position)
            goal.command.max_effort = float(GRIPPER_EFFORT)
            self.gripper_client.send_goal_async(goal)
            time.sleep(dwell)
            actual = self.gripper_position
            # Stop on contact instead of driving to the closed limit: squeezing
            # 7 mm into a 57 mm cube stores enough energy that the lift throws it
            # (measured: cube flung to (-1.66, -1.56) instead of held).
            if actual is not None and actual > position + 0.002:
                # Two more steps build a clamping force without the 7 mm squeeze
                # that threw the cube. Then hold still: gazebo_grasp_fix needs
                # four contact observations at 4 Hz before it keeps the grip, so
                # lifting immediately gives it nothing to hold.
                for extra in range(2):
                    position = max(GRIPPER_CLOSED, position - step)
                    goal = GripperCommand.Goal()
                    goal.command.position = float(position)
                    goal.command.max_effort = float(GRIPPER_EFFORT)
                    self.gripper_client.send_goal_async(goal)
                    time.sleep(dwell)
                self.get_logger().info(
                    'contact at %.4f (commanded %.4f); clamping and holding 2 s'
                    % (actual, position))
                time.sleep(2.0)
                return self.gripper_position, True
        time.sleep(settle)
        # Reaching the command means the jaws passed each other with nothing
        # between them; only a lagging joint is contact.
        return self.gripper_position, False

    def close_and_check(self, settle=1.8):
        """Close the fingers and report whether something stopped them.

        The jaws close to GRIPPER_CLOSED on an empty hand; a 57 mm cube stops them
        near q = 0.0075. That difference is the tactile signal, so no ground-truth
        query is needed: if the joint stalls short of the command, the cube is
        between the pads.
        """
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise SystemExit('/gripper_controller is not available')
        goal = GripperCommand.Goal()
        goal.command.position = float(GRIPPER_CLOSED)
        goal.command.max_effort = float(GRIPPER_EFFORT)
        self.gripper_client.send_goal_async(goal)
        time.sleep(settle)
        return self.gripper_position

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
    # The base creeps forward after the tree aims (a longer first walk pushes the
    # cube out of the camera view), so the cube measured here is that much closer
    # than the estimate taken before the creep. Compensating keeps the grasp from
    # depending on depth at 0.2 m, which is at the sensor's near limit.
    creep_m = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    lift_seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    rclpy.init()
    node = Grasper()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    pose = node.wait_for_cube()
    if pose is None:
        raise SystemExit('no cube pose with a 3D position')
    point = node.to_base_link(pose)
    node.get_logger().info('cube in base_link: %.3f %.3f %.3f' % tuple(point))
    # No base movement up front: IK is feasible at the approach distance but not
    # when the cube is 0.17 m away (the arm cannot fold that tightly), so the base
    # closes the horizontal residual *after* the arm has taken its pose. The pads
    # then travel with the base and land on the cube without a second IK call.
    del creep_m

    # Aim the pads at the cube centre. With the palm pitched by p, the pads lie
    # EE_TO_PADS_M ahead of end_effector_link along the tool axis, i.e. the
    # effector has to sit behind the cube by that much sin(p) and below it by
    # that much cos(p).
    def grasp_pose(pitch_deg, dz=0.0):
        import math
        sin = math.sin(math.radians(pitch_deg))
        cos = math.cos(math.radians(pitch_deg))
        return [point[0] - EE_TO_PADS_M * sin, point[1],
                point[2] + dz - EE_TO_PADS_M * cos]

    def pitch_quat(pitch_deg):
        import math
        half = math.radians(pitch_deg) / 2.0
        return (0.0, math.sin(half), 0.0, math.cos(half))

    grasp = grasp_pose(*GRASP_CANDIDATES[0])

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
    # One pose, one measurement, one base correction. Driving the base does not
    # change the base_link gap between the pads and the cube (both ride on the
    # base), so re-solving IK after a drive only walks the arm back: that mistake
    # kept the error at 7.7 cm forever and once shoved the cube 1.2 m.
    target = list(grasp)
    joints = None
    quat = None
    for pitch_deg, dz in GRASP_CANDIDATES:
        target = grasp_pose(pitch_deg, dz)
        quat = pitch_quat(pitch_deg)
        joints = node.solve_ik(target, quat)
        node.get_logger().info('pitch %.0f dz %+.3f at %s -> %s'
                               % (pitch_deg, dz,
                                  [round(v, 3) for v in target],
                                  'ok' if joints else 'no solution'))
        if joints is not None:
            break
    if joints is None:
        raise SystemExit('ik found no solution within reach (cube at %s)'
                         % [round(v, 3) for v in point])

    node.get_logger().info('ee target %s joints %s'
                           % ([round(v, 3) for v in target],
                              [round(v, 3) for v in joints]))
    node.draw_overlay('aim: pitch %.0f target %.3f,%.3f,%.3f'
                      % (GRASP_CANDIDATES[0][0], *target), target=target)
    node.send_arm(joints, seconds=5.0)
    time.sleep(1.0)
    pads = node.fingertip_midpoint()
    if pads is None:
        raise SystemExit('no fingertip transform after the move')
    error = [point[i] - pads[i] for i in range(3)]
    node.get_logger().info('pads at %s, error (%.3f, %.3f, %.3f) m'
                           % ([round(v, 3) for v in pads], *error))

    drive_m = error[0] - FINGER_REACH_M - 0.03
    if drive_m > 0.005:
        node.get_logger().info('driving %.3f m (gap %.3f less finger reach %.3f)'
                               % (drive_m, error[0], FINGER_REACH_M))
        node.drive_forward(drive_m, speed=0.02)
    else:
        node.get_logger().info('already at the cube: gap %.3f m <= finger reach'
                               % error[0])
    time.sleep(0.5)

    # Creep-and-close: the jaws are wider than the cube when open, so creeping
    # forward straddles it rather than pushing it; closing then either stalls on
    # the cube (grip) or reaches the closed position (empty). Each empty close
    # buys another centimetre of approach.
    gripped = False
    for attempt in range(8):
        position, stalled = node.close_slowly()
        node.get_logger().info(
            'close attempt %d: gripper at %s stalled=%s'
            % (attempt + 1, 'unknown' if position is None else '%.4f' % position,
               stalled))
        if stalled:
            gripped = True
            node.get_logger().info('fingers stopped on the cube: gripping')
            break
        # Nothing between the jaws: open and creep 5 mm closer, so the search
        # converges on the cube from behind instead of overshooting into it.
        node.send_gripper(0.019)
        time.sleep(0.6)
        node.get_logger().info('creeping 5 mm closer')
        node.drive_forward(0.005, speed=0.015)
        time.sleep(0.4)

    if not gripped:
        node.get_logger().warn('the jaws never stalled on the cube; '
                               'lifting anyway to record what happens')
    time.sleep(1.0)

    lifted = list(joints)
    lifted[1] += 0.25          # shoulder up
    lifted[2] -= 0.30          # elbow compensates, keeping the wrist above
    node.get_logger().info('lifting slowly')
    node.send_arm(lifted, seconds=max(lift_seconds, 8.0))
    time.sleep(2.0)
    print('grasp attempt finished')


if __name__ == '__main__':
    main()
