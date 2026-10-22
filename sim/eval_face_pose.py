#!/usr/bin/env python3
"""Measure the single-view face pose on real sim frames.

Recognition only. For each distance it spawns the cube, grabs a camera frame, asks
the perception box for the visible face, and solves that face with IPPE. The colour
label decides which of the six faces it is, which is what fixes yaw: without it the
same square fits six cube poses. The table reports position error, the recovered
face, and whether the colour agreed with it.

Run inside the lime container with the stack up:
    PYTHONPATH=/root/repo/yolo_ws/src/barcode_detector python3 sim/eval_face_pose.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, '/root/repo/yolo_ws/src')

from barcode_detector.cube_single_view_pose import (  # noqa: E402
    FacePoseError, estimate_face_pose)
from barcode_detector.cube_pose_tracking import quaternion_matrix  # noqa: E402

SIDE = 0.057
ROBOT_X = 0.02
BASE_Z = 0.06


def pose_from_matrix(matrix):
    """Quaternion (x, y, z, w) from a rotation matrix."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((matrix[2][1] - matrix[1][2]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale,
                (matrix[1][0] - matrix[0][1]) / scale,
                0.25 * scale)
    if matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return (0.25 * scale, (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[2][1] - matrix[1][2]) / scale)
    if matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return ((matrix[0][1] + matrix[1][0]) / scale, 0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale)
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return ((matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale,
            (matrix[1][0] - matrix[0][1]) / scale)


def quaternion_multiply(first, second):
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def axis_angle(axis, degrees):
    half = math.radians(degrees) / 2.0
    norm = math.sqrt(sum(value * value for value in axis))
    return tuple(value / norm * math.sin(half) for value in axis) + (math.cos(half),)


def angle_between(first, second):
    first_matrix = quaternion_matrix(first)
    second_matrix = quaternion_matrix(second)
    relative = first_matrix.T @ second_matrix
    trace = float(np.trace(relative))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


class Probe(Node):
    def __init__(self):
        super().__init__('eval_cube_tracking')
        self.bridge = CvBridge()
        self.image = None
        self.info = None
        self.result = None
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self.on_image, 1)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self.on_info, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_image(self, msg):
        self.image = msg

    def on_info(self, msg):
        self.info = msg

    def on_result(self, msg):
        self.result = msg.data

    def wait_result(self, timeout=15.0, settle=2.5):
        self.result = None
        start = time.time()
        while time.time() - start < settle:
            self.result = None
            time.sleep(0.2)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.result:
                return json.loads(self.result)
            time.sleep(0.1)
        return None

    def fresh_frame(self, settle=2.5, timeout=15.0):
        self.image = None
        start = time.time()
        while time.time() - start < settle:
            self.image = None
            time.sleep(0.2)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.image is not None and self.info is not None:
                return (self.bridge.imgmsg_to_cv2(self.image, 'bgr8'),
                        self.image.header.frame_id, list(self.info.k))
            time.sleep(0.1)
        return None

    def camera_pose(self, frame, timeout=10.0):
        """Pose of base_link in the camera frame: (position, quaternion)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    frame, 'base_link', rclpy.time.Time())
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        else:
            return None
        t = transform.transform.translation
        q = transform.transform.rotation
        return [t.x, t.y, t.z], (q.x, q.y, q.z, q.w)


def spawn_cube(distance):
    subprocess.run(['ros2', 'service', 'call', '/delete_entity',
                    'gazebo_msgs/srv/DeleteEntity', '{name: rubik_cube}'],
                   capture_output=True, text=True, timeout=20)
    time.sleep(0.5)
    subprocess.run(['python3', '/root/repo/sim/spawn_cube.py', str(distance),
                    '0.0'], capture_output=True, text=True, timeout=40)
    time.sleep(2.0)


def main():
    distances = [float(value) for value in sys.argv[1:]] or [0.4, 0.6, 0.8, 1.0]
    rclpy.init()
    node = Probe()
    node.create_subscription(String, '/cube_pose_result', node.on_result, 1)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    time.sleep(1.5)

    rows = []
    for distance in distances:
        spawn_cube(distance)
        frame = node.fresh_frame()
        result = node.wait_result()
        if frame is None or result is None:
            print('d=%.2f  no frame or no detection' % distance)
            continue
        image, frame_id, key = frame
        camera = node.camera_pose(frame_id)
        if camera is None:
            print('d=%.2f  no transform' % distance)
            continue
        camera_position, camera_quaternion = camera
        truth_base = np.asarray([distance - ROBOT_X, 0.0, SIDE / 2.0 - BASE_Z])
        rotation = quaternion_matrix(camera_quaternion)
        truth_camera = rotation @ truth_base + np.asarray(camera_position)

        box = None
        best = 0.0
        for detection in result.get('detections', []):
            if detection.get('confidence', 0.0) > best and detection.get('box_xyxy'):
                best = detection['confidence']
                box = detection['box_xyxy']
        if box is None:
            print('d=%.2f  no box' % distance)
            continue
        try:
            pose, evidence = estimate_face_pose(image, key, box, side_length_m=SIDE,
                                                maximum_reprojection_error_px=6.0)
        except FacePoseError as error:
            rows.append({'distance': distance, 'status': 'refused: %s' % error})
            print('d=%.2f  refused: %s' % (distance, error))
            continue
        error_m = math.dist(pose['position_m'], truth_camera)
        row = {'distance': distance, 'face': pose['visible_face'],
               'color_label': evidence['face_color_label'],
               'color_matched': evidence['color_matched'],
               'position_error_m': round(error_m, 4),
               'reprojection_px': round(pose['reprojection_error_px'], 2)}
        rows.append(row)
        print('d=%.2f  face %-3s label %-6s matched %-5s pos err %.4f m  reproj %.2f px'
              % (distance, pose['visible_face'], evidence['face_color_label'],
                 evidence['color_matched'], error_m,
                 pose['reprojection_error_px']))

    print('\nsummary')
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == '__main__':
    main()
