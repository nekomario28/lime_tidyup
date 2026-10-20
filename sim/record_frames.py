#!/usr/bin/env python3
"""Save frames from the robot camera and the observer to JPEG, for ffmpeg."""
import os
import sys
import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

OUT = sys.argv[1]
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
TOPICS = {
    'camera': ('/camera/camera/color/image_raw', 'camera'),
    'observer': ('/observer/camera/image_raw', 'observer'),
    # Perception's own annotated frame (bbox, pose status), published every frame.
    'overlay': ('/cube_pose_debug_image', 'overlay'),
    # Where the grasp aimed, published by sim/grasp_cube.py at each decision.
    'grasp': ('/grasp_debug_image', 'grasp'),
}


class Recorder(Node):
    def __init__(self):
        super().__init__('grasp_recorder')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.counts = {name: 0 for name in TOPICS}
        self.done = False
        for name, (topic, directory) in TOPICS.items():
            os.makedirs(f'{OUT}/frames_{directory}', exist_ok=True)
            self.create_subscription(
                Image, topic,
                lambda msg, n=name, d=directory: self.on_image(n, d, msg), 1,
            )
            self.get_logger().info(f'recording {topic}')

    def on_image(self, name, directory, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'{name}: {error}')
            return
        with self.lock:
            index = self.counts[name]
            self.counts[name] = index + 1
        cv2.imwrite(f'{OUT}/frames_{directory}/{index:05d}.jpg', frame)


def main():
    rclpy.init()
    node = Recorder()
    deadline = node.get_clock().now().nanoseconds + int(SECONDS * 1e9)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > deadline:
            break
    node.destroy_node()
    rclpy.shutdown()
    print('frames:', node.counts)


if __name__ == '__main__':
    main()
