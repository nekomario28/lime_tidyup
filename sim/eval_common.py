#!/usr/bin/env python3
"""The parts every cube-measurement eval needs, so each eval is only its measurement.

Three eval tools had grown their own copy of the same harness: spawn the cube, wait for
a frame, wait for a perception result, ask TF for the camera's pose, and turn the spawn
distance into "where the cube is, in the camera frame". The copies had already drifted
-- the same job was spelled `to_base_link` in one and `camera_pose` in another, and one
script's node still registered itself under its sibling's name -- which is the failure
this module removes: the harness is not what the evals are testing.

Each eval keeps its measurement and its report; this owns only the plumbing.
"""
from __future__ import annotations

import json
import os
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

import pose_math

#: The cube's side (sim/spawn_cube.py) and the robot's own geometry: the evals compute
#: truth from the spawn distance, and these convert it. BASE_Z_M is the robot's model
#: frame above base_link; ROBOT_X_M is where the robot settles when Gazebo cannot be
#: asked, since the evals report differences to the centimetre either way.
CUBE_SIDE_M = 0.057
BASE_Z_M = 0.06
ROBOT_X_M = 0.02

COLOR_TOPIC = '/camera/camera/color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'
RESULT_TOPIC = '/cube_pose_result'


class Probe(Node):
    """Subscribes to the camera and the perception result, with a fresh-frame wait."""

    def __init__(self, name='cube_eval_probe'):
        super().__init__(name)
        self.bridge = CvBridge()
        self.image = None
        self.info = None
        self.result = None
        self.create_subscription(Image, COLOR_TOPIC, self.on_image, 1)
        self.create_subscription(CameraInfo, INFO_TOPIC, self.on_info, 1)
        self.create_subscription(String, RESULT_TOPIC, self.on_result, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_image(self, msg):
        self.image = msg

    def on_info(self, msg):
        self.info = msg

    def on_result(self, msg):
        self.result = msg.data

    def wait_image(self, settle=2.5, timeout=15.0):
        """A frame and its intrinsics, both from after the settle."""
        self.image = None
        self.info = None
        start = time.time()
        while time.time() - start < settle:
            self.image = None
            self.info = None
            time.sleep(0.2)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.image is not None and self.info is not None:
                return (self.bridge.imgmsg_to_cv2(self.image, 'bgr8'),
                        self.image.header.frame_id, list(self.info.k))
            time.sleep(0.1)
        return None

    def wait_result(self, settle=2.5, timeout=15.0):
        """A perception result from after the settle, as parsed JSON."""
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

    def point_in_base_link(self, position, frame, timeout=10.0):
        """A point given in `frame`, moved into base_link."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', frame, rclpy.time.Time())
                break
            except Exception:  # noqa: BLE001  (transform not published yet)
                time.sleep(0.2)
        else:
            return None
        rotation = transform.transform.rotation
        translation = transform.transform.translation
        return list(pose_math.transform_point(
            (rotation.x, rotation.y, rotation.z, rotation.w),
            (translation.x, translation.y, translation.z), position))

    def camera_transform(self, frame, timeout=10.0):
        """base_link expressed in `frame`: (translation, quaternion)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    frame, 'base_link', rclpy.time.Time())
                break
            except Exception:  # noqa: BLE001  (transform not published yet)
                time.sleep(0.2)
        else:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return ([translation.x, translation.y, translation.z],
                (rotation.x, rotation.y, rotation.z, rotation.w))


def cube_in_base_link(distance, robot=(ROBOT_X_M, 0.0, BASE_Z_M)):
    """The spawned cube's centre in base_link, from the distance it was spawned at."""
    return [distance - robot[0], -robot[1], CUBE_SIDE_M / 2.0 - robot[2]]


def cube_in_frame(distance, camera, robot=(ROBOT_X_M, 0.0, BASE_Z_M)):
    """The same point expressed in the camera frame.

    `camera` is the transform of base_link in the camera frame, so a point maps as
    p_camera = R @ p_base + t. The inverse form was tried first, which put the cube
    behind the camera for a level mount.
    """
    translation, quaternion = camera
    rotation = np.asarray(pose_math.matrix_from_quaternion(quaternion))
    return rotation @ np.asarray(cube_in_base_link(distance, robot)) \
        + np.asarray(translation)


def spawn_cube(distance, lateral=0.0, yaw=0.0):
    """Replace the cube in the world at a known distance, through Gazebo's services."""
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(['ros2', 'service', 'call', '/delete_entity',
                    'gazebo_msgs/srv/DeleteEntity', '{name: rubik_cube}'],
                   capture_output=True, text=True, timeout=20)
    time.sleep(0.5)
    subprocess.run(['python3', os.path.join(here, 'spawn_cube.py'), str(distance),
                    str(lateral), str(yaw)],
                   capture_output=True, text=True, timeout=40)
    time.sleep(2.0)


def run_over_distances(measure, distances=None):
    """Drive `measure(distance, node, camera)` over the distances and print the rows.

    Owns what all three evals did identically: parse the distances, bring the node up
    once, spin it, and print a summary of the rows as JSON.
    """
    arguments = distances if distances is not None else [
        float(value) for value in sys.argv[1:]] or [0.4, 0.6, 0.8, 1.0]
    rclpy.init()
    node = Probe()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    time.sleep(1.5)
    rows = []
    for distance in arguments:
        spawn_cube(distance)
        rows.append(measure(distance, node))
    print('\nsummary')
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return rows
