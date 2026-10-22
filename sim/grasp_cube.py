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
#: Pitches the arm is offered, coarse to fine. Nothing here says which one works:
#: grasp_pose() is offered each in turn and the choice is made from where the pads
#: actually end up (see choose_pitch), because the answer changes with distance --
#: at 0.29 m only 150 solves, at 0.24 m the successful runs used 150 and 135.
PITCH_GRID = (150.0, 135.0, 120.0, 105.0, 90.0)
#: Aiming a touch below the cube's centre is where the pads can reach: measured, the
#: pads sit 4.4 cm above the centre when the aim is the centre itself. Treated as a
#: first guess, not a constant to tune -- aim_and_measure measures the difference and
#: subtracts it, so this only decides where the first measurement is taken.
FIRST_AIM_DZS = (0.0, -0.02, -0.03, -0.05, -0.07)
#: The pads' shortfall from the aim is measured and removed over this many rounds.
AIM_ROUNDS = 3
#: Where the jaws give up on a forward step: half the cube, because beyond that the
#: pads are behind its far face.
REACH_MARGIN_M = 0.01

#: The pads' shortfall from the aim is measured, not tuned: every round moves the
#: command by whatever the pads did not do, which converges because the arm is close
#: to linear over a centimetre. Measured over the runs, one correction takes the error
#: from up to 77 mm to under 0.5 mm, so this gate only has to absorb TF rounding --
#: a 12 mm gate was discarding accuracy the arm had already delivered and handing the
#: sweep a centimetre of it.
PAD_TOLERANCE_M = 0.003
#: Touch sweep, verified over four runs (grips at 0.0, 0.0, 0.5 and 1.0). The steps are
#: deliberately fixed: an adaptive doubling step was tried and gripped in none of three
#: runs, and the failure was not understood well enough to keep it.
TOUCH_CREEP_M = 0.005
TOUCH_CREEP_STEPS = 8
TOUCH_LOWER_M = 0.005
TOUCH_LOWER_STEPS = 4
#: How much forward room to leave between the cube and the arm's reachable limit. The
#: base distance is chosen from that, not from a constant: the sweep can only search
#: into the cube if the arm has somewhere to go, and how much room there is depends on
#: the pitch, which is chosen at runtime.
#: The largest forward shortfall a sweep has actually needed is 25 mm, so the reserve
#: is that plus one first step -- enough to reach the cube and still be searching
#: rather than shoved up against the limit.
REACH_MARGIN_M = 0.035
#: Search bounds for the reachable pad range at a given pitch.
REACH_SEARCH_M = (0.12, 0.42)
REACH_BISECTIONS = 7


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


def reachable_pad_x(node, pitch_deg, height):
    """The furthest forward the pads can be posed at this pitch and height.

    Measured with the IK service rather than assumed: the arm's envelope is what
    decides where the base should stop, and it changes with the pitch (only 150
    solves at 0.29 m, while at 0.24 m 150 and 135 both do). Bisection on "does IK
    solve here" gives the boundary in a handful of service calls.
    """
    low, high = REACH_SEARCH_M
    probe = grasp_pose([high, 0.0, height], pitch_deg, 0.0)
    if node.solve_ik(probe, pitch_quat(pitch_deg)) is not None:
        return high
    for _ in range(REACH_BISECTIONS):
        middle = 0.5 * (low + high)
        probe = grasp_pose([middle, 0.0, height], pitch_deg, 0.0)
        if node.solve_ik(probe, pitch_quat(pitch_deg)) is not None:
            low = middle
        else:
            high = middle
    return low


def aim_at_pads(node, point):
    """Command the arm until the pads themselves land on `point`.

    The pads do not go where the arm is asked to go: measured shortfalls are 2.2 cm
    forward at pitch 150 and 4.4 cm in height at the centre aim. Tuning those numbers
    by hand is what GRASP_CANDIDATES used to be. Instead each round moves the command
    by exactly the error that was measured.

    The first guess only has to be reachable -- the rounds do the aiming -- but one
    guess is not enough: at the pre-grasp distance the centre aim is refused at every
    pitch while 3 cm lower solves at 150, so (pitch, first guess) is walked until a
    pose is accepted and converges. Returns (aim, joints, pitch) with the arm already
    there, or (None, None, None).
    """
    goal = list(point)
    for pitch_deg in PITCH_GRID:
        for first_dz in FIRST_AIM_DZS:
            aim = [goal[0], goal[1], goal[2] + first_dz]
            joints = None
            converged = False
            for round_index in range(AIM_ROUNDS):
                target = grasp_pose(aim, pitch_deg, 0.0)
                joints = node.solve_ik(target, pitch_quat(pitch_deg))
                if joints is None:
                    break
                node.send_arm(joints, seconds=4.0)
                time.sleep(0.8)
                pads = node.fingertip_midpoint()
                if pads is None:
                    break
                error = [goal[axis] - pads[axis] for axis in range(3)]
                node.get_logger().info(
                    'pitch %.0f dz %+.2f round %d: pads off by %+.3f %+.3f %+.3f'
                    % (pitch_deg, first_dz, round_index + 1,
                       error[0], error[1], error[2]))
                if max(abs(value) for value in error) <= PAD_TOLERANCE_M:
                    converged = True
                    break
                for axis in range(3):
                    aim[axis] += error[axis]
            if converged:
                return aim, joints, pitch_deg
    return None, None, None


def hold_aim(node, aim, pitch_deg):
    """Put the arm back on the corrected aim; None when IK refuses."""
    target = grasp_pose(aim, pitch_deg, 0.0)
    joints = node.solve_ik(target, pitch_quat(pitch_deg))
    if joints is None:
        return None
    node.send_arm(joints, seconds=2.5)
    time.sleep(0.4)
    return joints


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

    # 2. Pose with the pads, not the wrist: aim_at_pads measures where the pads
    #    actually go and corrects for it, and the pitch is whichever one converges.
    node.get_logger().info('opening the gripper')
    node.send_gripper(0.019)
    time.sleep(1.0)

    # Where the base stops is what makes the sweep possible: the pads have to have
    # somewhere forward to go, or the search dies against the arm's own limit. Both
    # sides of that comparison are measured -- where the pads are now (TF) and how far
    # forward they can be posed at that height (IK bisection) -- so no guess about the
    # grasp height is needed. A failed aim is answered the same way: drive forward,
    # which is the only direction that brings the cube into the arm's envelope.
    aim = joints = None
    pitch_deg = PITCH_GRID[0]
    for attempt in range(3):
        aim, joints, pitch_deg = aim_at_pads(node, point)
        if joints is None:
            creep = 0.02
            node.get_logger().info('nothing reachable for the cube at %.3f m; driving '
                                   'forward %.3f m' % (point[0], creep))
            node.drive_forward(creep, speed=0.03)
            point[0] -= creep
            time.sleep(0.5)
            continue
        pads = node.fingertip_midpoint()
        if pads is None:
            break
        reach_x = reachable_pad_x(node, pitch_deg, pads[2])
        room = reach_x - pads[0]
        if room >= REACH_MARGIN_M:
            node.get_logger().info(
                'pads at %.3f m, reach %.3f m at that height: %.3f m of search room'
                % (pads[0], reach_x, room))
            break
        creep = REACH_MARGIN_M - room
        node.get_logger().info(
            'pads at %.3f m, reach %.3f m: only %.3f m of room, driving forward %.3f m'
            % (pads[0], reach_x, room, creep))
        node.drive_forward(creep, speed=0.03)
        point[0] -= creep
        time.sleep(0.5)
    if joints is None:
        raise SystemExit('ik found no solution within reach (cube at %s)'
                         % [round(v, 3) for v in point])

    # 3. Hand over to touch. The camera is no longer a reference here: the arm sits
    #    between it and the cube, and re-measuring made the cube walk forward about
    #    2 cm per step (0.240 -> 0.261 -> 0.277). What is trustworthy is the jaws
    #    stalling, so close, and on an empty close open, push the aim forward 5 mm and
    #    close again; when a whole forward sweep at one height finds nothing, lower a
    #    centimetre and sweep again. An adaptive doubling step was tried and gripped in
    #    none of three runs, so the fixed step is what ships until the failure is
    #    understood.
    gripped = False
    for level in range(TOUCH_LOWER_STEPS + 1):
        for creep in range(TOUCH_CREEP_STEPS + 1):
            position, stalled = node.close_slowly()
            node.get_logger().info(
                'touch %d.%d: gripper at %s stalled=%s'
                % (level, creep, 'unknown' if position is None
                   else '%.4f' % position, stalled))
            if stalled:
                node.get_logger().info('fingers stopped on the cube: gripping')
                gripped = True
                break
            node.send_gripper(0.019)
            time.sleep(0.5)
            if creep == TOUCH_CREEP_STEPS:
                break
            if level == 0 and creep == 0:
                # the aim already sits where the pads landed, which is the best
                # estimate there is; the sweep moves out from there
                pass
            aim[0] += TOUCH_CREEP_M
            if hold_aim(node, aim, pitch_deg) is None:
                node.get_logger().info('arm cannot reach %.3f m; ending this sweep'
                                       % aim[0])
                aim[0] -= TOUCH_CREEP_M
                break
        if gripped or level == TOUCH_LOWER_STEPS:
            break
        node.get_logger().info('nothing at this height; lowering the aim by %.0f mm'
                               % (TOUCH_LOWER_M * 1000))
        aim[0] -= TOUCH_CREEP_M * TOUCH_CREEP_STEPS
        aim[2] -= TOUCH_LOWER_M
        if hold_aim(node, aim, pitch_deg) is None:
            node.get_logger().warn('ik will not go lower; this arm cannot reach the '
                                   'cube from here')
            break

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
