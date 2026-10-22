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
import math
import sys
import threading
import time

import os

import cv2
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
#: Where the pads actually contact, measured from the finger meshes
#: (gripper_*_palm.stl, scaled by 0.001 in the URDF): the finger runs from
#: z = -23 mm to z = +65 mm along the tool axis relative to
#: gripper_left/right_link, so the pad's contact band centres about +25 mm past
#: the link origin. With the grasp pitch of 120 deg that is 0.025*sin(120) =
#: 0.022 m of forward reach beyond the point TF reports, which is exactly the
#: gap that made the jaws close on air (0.05 assumed) or shove the cube (0.0).
FINGER_REACH_M = 0.022
#: Where the cube should sit in base_link when the arm takes its pose. The tree
#: leaves it near 0.29 m, and at that distance the only IK-feasible orientation
#: cannot bring the pads down to the cube's height; closer, the arm has room.
GRASP_TARGET_DISTANCE_M = 0.24
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
GRASP_PITCH_ORDER = (150.0, 120.0, 135.0, 105.0)
#: (pitch degrees, height offset from the measured cube centre).
#: Measured, not guessed: with the cube at the pre-grasp distance the successful run
#: closed empty at dz -0.03 and again at -0.04, and only reached the cube at -0.05
#: (close 3: gripper 0.0155, stalled=True). Pitch 150 is the one that solves there;
#: 120 and 135 are refused. The rest of the list stays as a fallback for other
#: distances.
GRASP_CANDIDATES = (
    (150.0, -0.05),
    (150.0, -0.03),
    (120.0, -0.03),
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

    @staticmethod
    def robot_world_x():
        """The robot's world x, used only to prove that a creep moved it.

        A base move cannot change base_link coordinates -- both the cube and the
        pads live there -- so the earlier "pads moved 0.0000" check could never
        have detected a stalled drive.
        """
        import subprocess
        try:
            output = subprocess.run(
                ['gz', 'model', '--model-name', 'turtlebot3_lime_system',
                 '--info'], capture_output=True, text=True, timeout=25).stdout
        except Exception as error:  # noqa: BLE001
            print('robot_world_x failed:', type(error).__name__, error)
            return None
        lines = output.splitlines()
        if not lines:
            print('robot_world_x: gz model returned nothing')
        for index, line in enumerate(lines):
            if line.strip() == 'position {':
                for offset in (1, 2, 3):
                    if index + offset < len(lines) and 'x:' in lines[index + offset]:
                        try:
                            return float(lines[index + offset].split(':')[1])
                        except ValueError:
                            return None
        return None

    def drive_forward(self, metres, speed=0.03):
        """Creep the base forward; the tree stops short of the arm's real reach.

        Speed matters: at 0.015 m/s the differential drive does not move at all
        (measured: eight 5 mm creeps left the cube untouched and the pads in
        place), so the command is never issued below 0.03 m/s.
        """
        speed = max(abs(speed), 0.03)
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

    def wait_for_pose(self, timeout=6.0, maximum_age=3.0):
        """The latest cube pose, provided it is recent enough to act on.

        Requiring a *newer* message than the last one was tried and deadlocked: the
        publisher pauses for tens of seconds at a time (measured: a 73 s gap while
        the base was being driven), so every check timed out and the loop kept
        re-issuing the same move. Age is the honest test -- the pose carries the
        node's own timestamp, on this clock.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.latest:
                data = json.loads(self.latest)
                age = time.time() - float(data.get('timestamp') or 0.0)
                if age <= float(maximum_age):
                    best = None
                    for detection in data.get('detections', []):
                        pose = detection.get('pose_3d') or {}
                        if len(pose.get('position_m') or []) == 3:
                            if best is None or detection['confidence'] > best[0]:
                                best = (detection['confidence'], pose)
                    if best is not None:
                        return best[1]
            time.sleep(0.1)
        return None

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
                for extra in range(1):
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


def grasp_pose(point, pitch_deg, dz=0.0):
    """End-effector target that puts the pads on the cube centre.

    With the palm pitched by p the pads lie EE_TO_PADS_M ahead of
    end_effector_link along the tool axis, so the effector sits behind the cube by
    that much sin(p) and below it by that much cos(p).
    """
    sin = math.sin(math.radians(pitch_deg))
    cos = math.cos(math.radians(pitch_deg))
    return [point[0] - EE_TO_PADS_M * sin, point[1],
            point[2] + dz - EE_TO_PADS_M * cos]


def pitch_quat(pitch_deg):
    half = math.radians(pitch_deg) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


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

    # 1. Range. From where the tree leaves the robot the pads cannot be posed low
    #    enough, and closer they can: with the cube at 0.24 m instead of 0.29 m the
    #    pitch 150 solution exists where 120 and 135 are refused. The drive is
    #    measured (0.05 m/s for 2 s moved 0.098 m), so the target moves with it.
    if point[0] > GRASP_TARGET_DISTANCE_M:
        approach = min(0.06, point[0] - GRASP_TARGET_DISTANCE_M)
        node.get_logger().info('closing range by %.3f m to %.2f m'
                               % (approach, point[0] - approach))
        node.drive_forward(approach, speed=0.03)
        point[0] -= approach
        time.sleep(0.5)

    # 2. Pose. The jaws must be open before the descent, or they push the cube
    #    instead of straddling it.
    node.get_logger().info('opening the gripper')
    node.send_gripper(0.019)
    time.sleep(1.0)

    target = None
    joints = None
    for pitch_deg, dz in GRASP_CANDIDATES:
        target = grasp_pose(point, pitch_deg, dz)
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

    # 3. Verify. Re-read the cube pose and measure the pads, then drive the
    #    residual in; close only once they are within tolerance. Driving forward by
    #    d brings the cube d closer in base_link, so when the publisher is silent
    #    (it stops while the base moves -- measured a 73 s gap) the same arithmetic
    #    advances the target instead of stalling the loop.
    tolerance_m = 0.012
    verified = False
    last_drive = 0.0
    for attempt in range(4):
        pads = node.fingertip_midpoint()
        if pads is None:
            raise SystemExit('no fingertip transform')
        fresh = node.wait_for_pose()
        if fresh is None:
            node.get_logger().info(
                'no fresh pose; applying the commanded %.3f m to the target'
                % last_drive)
        else:
            point = node.to_base_link(fresh)
            last_drive = 0.0
            node.get_logger().info(
                're-measured cube at %.3f %.3f %.3f (%s)'
                % (point[0], point[1], point[2],
                   fresh.get('geometry_id', '?')))
        pads_target = point[0] - FINGER_REACH_M
        residual_x = pads_target - pads[0]
        residual_y = point[1] - pads[1]
        node.get_logger().info(
            'check %d: pads %.3f target %.3f -> dx %.3f, dy %.3f'
            % (attempt + 1, pads[0], pads_target, residual_x, residual_y))
        if abs(residual_x) <= tolerance_m and abs(residual_y) <= 0.015:
            verified = True
            node.get_logger().info('verified: pads within %.0f mm of the cube'
                                   % (tolerance_m * 1000))
            break
        drive_m = max(-0.06, min(0.06, residual_x))
        if abs(drive_m) > 0.004:
            node.get_logger().info('adjusting base by %+.3f m' % drive_m)
            node.drive_forward(drive_m, speed=0.03)
            point[0] -= drive_m
            last_drive = drive_m
            time.sleep(0.6)
    if not verified:
        node.get_logger().warn('could not verify the pads around the cube; '
                               'closing anyway and recording what happens')

    # 4. Close, lowering the aim a centimetre at a time when the jaws pass each
    #    other. IK refusing is the answer to how far this arm can reach down.
    for lower in range(4):
        position, stalled = node.close_slowly()
        node.get_logger().info('close: gripper at %s stalled=%s'
                               % ('unknown' if position is None
                                  else '%.4f' % position, stalled))
        if stalled:
            node.get_logger().info('fingers stopped on the cube: gripping')
            break
        node.get_logger().info('jaws closed empty; lowering the aim by 1 cm')
        node.send_gripper(0.019)
        time.sleep(0.6)
        target[2] -= 0.01
        solved = None
        for pitch_deg in GRASP_PITCH_ORDER:
            solved = node.solve_ik(target, pitch_quat(pitch_deg))
            if solved is not None:
                node.get_logger().info('lowered: pitch %.0f at %s'
                                       % (pitch_deg,
                                          [round(v, 3) for v in target]))
                break
        if solved is None:
            node.get_logger().warn('ik will not go %.2f m lower; this arm cannot '
                                   'reach the cube from here'
                                   % (0.01 * (lower + 1)))
            break
        joints = solved
        node.send_arm(joints, seconds=4.0)
        time.sleep(0.8)
    time.sleep(1.0)

    # 5. Lift.
    lifted = list(joints)
    lifted[1] += 0.25          # shoulder up
    lifted[2] -= 0.30          # elbow compensates, keeping the wrist above
    node.get_logger().info('lifting')
    node.send_arm(lifted, seconds=lift_seconds)
    time.sleep(2.0)
    print('grasp attempt finished')


if __name__ == '__main__':
    main()
