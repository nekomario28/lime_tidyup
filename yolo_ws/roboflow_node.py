import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class RoboflowNode(Node):
    def __init__(self):
        super().__init__('roboflow_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info('Roboflow node started. Waiting for images...')

    def image_callback(self, msg):
        self.get_logger().info('画像を受け取りました')


def main(args=None):
    rclpy.init(args=args)
    node = RoboflowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
