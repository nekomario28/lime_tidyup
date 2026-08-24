import json
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO

from .cube_pose_3d import CubePose3DError, estimate_cube_pose_3d


class YoloPoseNode(Node):
    def __init__(self):
        super().__init__('yolo_pose_node')

        self.image_topic = '/camera/camera/color/image_raw'
        self.camera_info_topic = '/camera/camera/color/camera_info'
        self.result_topic = '/cube_pose_result'

        self.declare_parameter(
            'model_path',
            '/root/yolo_ws/runs/pose/train-2/weights/best.pt'
        )
        self.declare_parameter('cube_side_length_m', 0.057)
        self.declare_parameter('keypoint_confidence', 0.50)
        self.declare_parameter('max_reprojection_error_px', 4.0)

        self.model_path = str(self.get_parameter('model_path').value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.camera_info = None

        self.last_time = 0.0
        self.interval_sec = 1.0

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
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        self.get_logger().info('YOLO Pose node started')
        self.get_logger().info(f'subscribe: {self.image_topic}')
        self.get_logger().info(f'subscribe: {self.camera_info_topic}')
        self.get_logger().info(f'publish: {self.result_topic}')
        self.get_logger().info(f'model: {self.model_path}')

    def camera_info_callback(self, msg):
        """Keep the latest static color-camera calibration."""
        self.camera_info = msg

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
                    if r.keypoints.conf is not None:
                        keypoints_confidence = (
                            r.keypoints.conf.cpu().tolist()
                        )
                    else:
                        keypoints_confidence = [
                            [1.0] * len(points)
                            for points in keypoints_xy
                        ]
                else:
                    keypoints_xy = []
                    keypoints_confidence = []

                for i in range(len(boxes)):
                    class_id = int(class_ids[i])
                    class_name = r.names[class_id]

                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(confidences[i]),
                        'box_xyxy': boxes[i],
                        'keypoints_xy': (
                            keypoints_xy[i]
                            if i < len(keypoints_xy)
                            else []
                        ),
                        'keypoints_confidence': (
                            keypoints_confidence[i]
                            if i < len(keypoints_confidence)
                            else []
                        )
                    }

                    pose_3d = self._estimate_pose_3d(
                        msg,
                        now,
                        detection['keypoints_xy'],
                        detection['keypoints_confidence']
                    )
                    if pose_3d is not None:
                        detection['pose_3d'] = pose_3d

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

    def _estimate_pose_3d(
        self,
        image_msg,
        timestamp,
        keypoints_xy,
        keypoint_confidences
    ):
        """Return an accepted camera-frame pose, or ``None`` as fallback."""
        camera_info = self.camera_info
        if camera_info is None:
            return None

        image_frame = image_msg.header.frame_id
        camera_frame = camera_info.header.frame_id
        if image_frame and camera_frame and image_frame != camera_frame:
            self.get_logger().warning(
                '3D pose skipped: image and CameraInfo frame mismatch'
            )
            return None

        try:
            pose = estimate_cube_pose_3d(
                keypoints_xy,
                keypoint_confidences,
                camera_info.k,
                camera_info.d,
                side_length_m=self.get_parameter(
                    'cube_side_length_m'
                ).value,
                minimum_keypoint_confidence=self.get_parameter(
                    'keypoint_confidence'
                ).value,
                maximum_reprojection_error_px=self.get_parameter(
                    'max_reprojection_error_px'
                ).value
            )
        except CubePose3DError as error:
            self.get_logger().debug(f'3D pose rejected: {error}')
            return None

        return pose.as_dict(
            frame_id=camera_frame or image_frame,
            timestamp=timestamp,
            side_length_m=self.get_parameter('cube_side_length_m').value
        )


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
