import json
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloPoseNode(Node):
    def __init__(self):
        super().__init__('yolo_pose_node')

        self.image_topic = '/camera/camera/color/image_raw'
        self.result_topic = '/cube_pose_result'

        self.model_path = '/root/yolo_ws/runs/pose/train-2/weights/best.pt'

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.last_time = 0.0
        self.interval_sec = 2.0

        self.publisher = self.create_publisher(
            String,
            self.result_topic,
            10
        )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info('YOLO Pose node started')
        self.get_logger().info(f'subscribe: {self.image_topic}')
        self.get_logger().info(f'publish: {self.result_topic}')
        self.get_logger().info(f'model: {self.model_path}')

    def image_callback(self, msg):
        now = time.time()

        # 毎フレーム推論すると重いので、2秒に1回だけ推論
        if now - self.last_time < self.interval_sec:
            return

        self.last_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            results = self.model(cv_image, verbose=False)

            detections = []

            for r in results:
                if r.boxes is None:
                    continue

                boxes = r.boxes.xyxy.cpu().tolist()
                class_ids = r.boxes.cls.cpu().tolist()
                confidences = r.boxes.conf.cpu().tolist()

                if r.keypoints is not None:
                    keypoints_xy = r.keypoints.xy.cpu().tolist()
                else:
                    keypoints_xy = []

                for i in range(len(boxes)):
                    class_id = int(class_ids[i])
                    class_name = r.names[class_id]

                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(confidences[i]),
                        'box_xyxy': boxes[i],
                        'keypoints_xy': keypoints_xy[i] if i < len(keypoints_xy) else []
                    }

                    detections.append(detection)

            result_data = {
                'timestamp': now,
                'detections': detections
            }

            msg_out = String()
            msg_out.data = json.dumps(result_data, ensure_ascii=False)

            self.publisher.publish(msg_out)

            self.get_logger().info(
                f'published {len(detections)} detections'
            )

        except Exception as e:
            self.get_logger().error(f'YOLO pose error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
