import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RoboflowNode(Node):
    def __init__(self):
        super().__init__('roboflow_node')

        self.bridge = CvBridge()
        self.saved = False

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info('Roboflow node started. Waiting for images...')

    def image_callback(self, msg):
        if self.saved:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            save_path = '/tmp/test.jpg'
            cv2.imwrite(save_path, cv_image)

            self.saved = True
            self.get_logger().info(f'Saved image to {save_path}')

        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RoboflowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
