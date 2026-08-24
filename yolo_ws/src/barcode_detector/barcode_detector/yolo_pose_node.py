import json
import math
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
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter(
            'debug_image_topic',
            '/cube_pose_debug_image'
        )
        self.declare_parameter('debug_image_path', '')

        self.model_path = str(self.get_parameter('model_path').value)
        self.publish_debug_image = bool(
            self.get_parameter('publish_debug_image').value
        )
        self.debug_image_topic = str(
            self.get_parameter('debug_image_topic').value
        )
        self.debug_image_path = str(
            self.get_parameter('debug_image_path').value
        )

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
        self.debug_publisher = None
        if self.publish_debug_image:
            self.debug_publisher = self.create_publisher(
                Image,
                self.debug_image_topic,
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
        if self.debug_publisher is not None:
            self.get_logger().info(
                f'publish debug image: {self.debug_image_topic}'
            )
        if self.debug_image_path:
            self.get_logger().info(
                f'write debug image: {self.debug_image_path}'
            )
        self.get_logger().info(f'model: {self.model_path}')

    def camera_info_callback(self, msg):
        """Keep the latest static color-camera calibration."""
        self.camera_info = msg

    def image_callback(self, msg):
        now = time.time()

        # 毎フレーム推論すると重いので、1秒に1回だけ推論
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

                    pose_3d, pose_debug = self._estimate_pose_3d_with_debug(
                        msg,
                        now,
                        detection['keypoints_xy'],
                        detection['keypoints_confidence']
                    )
                    detection['pose_3d_debug'] = pose_debug
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
            self._publish_debug_image(msg, cv_image, detections)

            self.get_logger().info(
                f'published {len(detections)} detections'
            )

        except Exception as e:
            self.get_logger().error(f'YOLO pose error: {e}')

    def _estimate_pose_3d_with_debug(
        self,
        image_msg,
        timestamp,
        keypoints_xy,
        keypoint_confidences
    ):
        """Return accepted pose plus evidence explaining 3D acceptance."""
        minimum_keypoint_confidence = float(
            self.get_parameter('keypoint_confidence').value
        )
        usable_keypoints = self._count_usable_keypoints(
            keypoints_xy,
            keypoint_confidences,
            minimum_keypoint_confidence
        )
        debug = {
            'status': 'unavailable',
            'reason': '',
            'keypoint_threshold': minimum_keypoint_confidence,
            'usable_keypoint_count': usable_keypoints,
            'total_keypoint_count': len(keypoints_xy),
        }

        camera_info = self.camera_info
        if camera_info is None:
            debug['reason'] = 'CameraInfo unavailable'
            return None, debug

        image_frame = image_msg.header.frame_id
        camera_frame = camera_info.header.frame_id
        if image_frame and camera_frame and image_frame != camera_frame:
            debug['reason'] = 'image/CameraInfo frame mismatch'
            self.get_logger().warning(
                '3D pose skipped: image and CameraInfo frame mismatch'
            )
            return None, debug

        try:
            pose = estimate_cube_pose_3d(
                keypoints_xy,
                keypoint_confidences,
                camera_info.k,
                camera_info.d,
                side_length_m=self.get_parameter(
                    'cube_side_length_m'
                ).value,
                minimum_keypoint_confidence=minimum_keypoint_confidence,
                maximum_reprojection_error_px=self.get_parameter(
                    'max_reprojection_error_px'
                ).value
            )
        except CubePose3DError as error:
            debug['status'] = 'rejected'
            debug['reason'] = str(error)
            self.get_logger().debug(f'3D pose rejected: {error}')
            return None, debug

        pose_dict = pose.as_dict(
            frame_id=camera_frame or image_frame,
            timestamp=timestamp,
            side_length_m=self.get_parameter('cube_side_length_m').value
        )
        debug.update({
            'status': 'accepted',
            'reason': '',
            'correspondence_count': pose.correspondence_count,
            'inlier_count': pose.inlier_count,
            'inlier_mask': list(pose.inlier_mask),
            'reprojection_error_px': pose.reprojection_error_px,
            'z_m': pose.translation_m[2],
        })
        return pose_dict, debug

    def _estimate_pose_3d(
        self,
        image_msg,
        timestamp,
        keypoints_xy,
        keypoint_confidences
    ):
        """Return an accepted camera-frame pose, or ``None`` as fallback."""
        pose, _ = self._estimate_pose_3d_with_debug(
            image_msg,
            timestamp,
            keypoints_xy,
            keypoint_confidences
        )
        return pose

    @staticmethod
    def _count_usable_keypoints(
        keypoints_xy,
        keypoint_confidences,
        minimum_confidence
    ):
        usable = 0
        for point, confidence in zip(keypoints_xy, keypoint_confidences):
            try:
                x = float(point[0])
                y = float(point[1])
                confidence = float(confidence)
            except (IndexError, TypeError, ValueError):
                continue
            if (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(confidence)
                and confidence >= minimum_confidence
                and not (x == 0.0 and y == 0.0)
            ):
                usable += 1
        return usable

    def _publish_debug_image(self, image_msg, cv_image, detections):
        """Overlay measured YOLO/PnP evidence without inventing values."""
        if self.debug_publisher is None and not self.debug_image_path:
            return

        try:
            import cv2
        except ImportError:
            self.get_logger().warning(
                'debug image skipped: OpenCV is unavailable'
            )
            return

        debug_image = cv_image.copy()
        for detection in detections:
            self._draw_detection_debug(cv2, debug_image, detection)

        if self.debug_publisher is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(
                debug_image,
                encoding='bgr8'
            )
            debug_msg.header = image_msg.header
            self.debug_publisher.publish(debug_msg)

        # Keep the last useful persisted evidence. Empty inference frames are
        # still published on the debug topic but must not erase a detection.
        if self.debug_image_path and detections:
            if not cv2.imwrite(self.debug_image_path, debug_image):
                self.get_logger().warning(
                    f'failed to write debug image: {self.debug_image_path}'
                )

    def _draw_detection_debug(self, cv2, image, detection):
        box = detection.get('box_xyxy', [])
        if len(box) != 4:
            return
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        except (TypeError, ValueError):
            return

        debug = detection.get('pose_3d_debug', {})
        status = debug.get('status', 'unavailable')
        if status == 'accepted':
            box_color = (60, 220, 60)
        elif status == 'rejected':
            box_color = (60, 80, 255)
        else:
            box_color = (0, 200, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        class_name = str(detection.get('class_name', 'cube'))
        try:
            confidence = float(detection.get('confidence', 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        self._draw_text(
            cv2,
            image,
            f'{class_name} conf={confidence:.3f}',
            (x1, max(18, y1 - 36))
        )

        usable = int(debug.get('usable_keypoint_count', 0))
        total = int(debug.get('total_keypoint_count', 0))
        if status == 'accepted':
            inliers = int(debug.get('inlier_count', 0))
            correspondences = int(debug.get('correspondence_count', 0))
            rms = float(debug.get('reprojection_error_px', float('nan')))
            z_m = float(debug.get('z_m', float('nan')))
            evidence_text = (
                f'3D PASS kp={usable}/{total} '
                f'inliers={inliers}/{correspondences} '
                f'RMS={rms:.2f}px Z={z_m:.3f}m'
            )
        else:
            reason = str(debug.get('reason', ''))
            prefix = '3D REJECT' if status == 'rejected' else '3D WAIT'
            evidence_text = f'{prefix} kp={usable}/{total} {reason}'

        self._draw_text(
            cv2,
            image,
            evidence_text[:110],
            (x1, max(18, y1 - 14))
        )

        keypoints = detection.get('keypoints_xy', [])
        confidences = detection.get('keypoints_confidence', [])
        threshold = float(debug.get('keypoint_threshold', 0.50))
        inlier_mask = debug.get('inlier_mask', [])
        for index, point in enumerate(keypoints):
            try:
                px = int(round(float(point[0])))
                py = int(round(float(point[1])))
                score = (
                    float(confidences[index])
                    if index < len(confidences)
                    else 1.0
                )
            except (IndexError, TypeError, ValueError):
                continue
            if px == 0 and py == 0:
                continue

            is_usable = math.isfinite(score) and score >= threshold
            is_inlier = (
                index < len(inlier_mask)
                and bool(inlier_mask[index])
            )
            if is_inlier:
                point_color = (60, 220, 60)
                radius = 6
            elif is_usable:
                point_color = (255, 220, 60)
                radius = 5
            else:
                point_color = (150, 150, 150)
                radius = 3

            cv2.circle(image, (px, py), radius, point_color, -1)
            self._draw_text(
                cv2,
                image,
                f'k{index}:{score:.2f}',
                (px + 7, py - 5),
                scale=0.38
            )

    @staticmethod
    def _draw_text(cv2, image, text, origin, scale=0.48):
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
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
