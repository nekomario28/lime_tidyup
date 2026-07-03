import os
import time

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloSaveNode(Node):
    def __init__(self):
        super().__init__('yolo_save_node')

        self.image_topic = '/camera/camera/color/image_raw'

        # 学習済みモデル
        self.model_path = '/root/yolo_ws/runs/pose/train-2/weights/best.pt'

        # 保存先
        self.save_dir = '/root/yolo_ws/runs/pose/predict'

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.saved = False

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info('YOLO save node started')
        self.get_logger().info(f'subscribe: {self.image_topic}')
        self.get_logger().info(f'model: {self.model_path}')
        self.get_logger().info(f'save_dir: {self.save_dir}')

    def image_callback(self, msg):
        if self.saved:
            return

        try:
            self.get_logger().info('Image received. Running YOLO...')

            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            results = self.model(cv_image, verbose=False)

            # 推論結果を描画した画像を作る
            annotated_image = results[0].plot()

            timestamp = int(time.time())
            save_path = os.path.join(
                self.save_dir,
                f'yolo_pose_result_{timestamp}.jpg'
            )

            cv2.imwrite(save_path, annotated_image)

            self.saved = True

            self.get_logger().info(f'Saved YOLO result image to: {save_path}')
            self.get_logger().info('Finished. Shutting down...')

            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'YOLO save error: {e}')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = YoloSaveNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
