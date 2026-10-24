import hashlib
import json

import numpy as np
import math
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:  # the model-free route must import without the ML stack installed
    from ultralytics import YOLO
except Exception:  # pragma: no cover - absence is a supported configuration
    YOLO = None

from .cube_pose_3d import CubePose3DError, estimate_cube_pose_3d
from .cube_bbox_source import BboxSourceError, choose_bbox
from .cube_single_view_pose import (
    FacePoseError,
    estimate_face_pose,
    face_box_from_color,
)
from .cube_model_free_pose import (
    ModelFreePoseError,
    estimate_model_free_pose,
)
from .cube_overlay import draw_cube_overlay
from .cube_pose_evidence import attach_pose_evidence


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
        # Opt-in evidence stages. Both default off so the published contract
        # and the accepted pose are unchanged unless a caller asks for them.
        self.declare_parameter('enable_visibility_gate', False)
        self.declare_parameter('enable_face_color_selection', False)
        self.declare_parameter('face_color_minimum_chroma', 10.0)
        self.declare_parameter('face_color_minimum_hue_margin_deg', 8.0)
        # Perception route. 'yolov8-pose' is the trained-keypoint path;
        # 'color_frontend' is the model-free geometry+colour path, which needs a
        # detector only for a bounding box.
        self.declare_parameter('perception_backend', 'color_frontend')
        # Where the cube bounding box comes from. 'detector' keeps the previous
        # behaviour; 'depth_blob' needs no learned detector at all and reads a
        # depth image instead (see cube_bbox_source).
        self.declare_parameter('bbox_source', 'detector')
        self.declare_parameter('depth_topic', '')
        self.declare_parameter('depth_scale_m', 0.001)
        # Publish the cube centre as pose_3d when a depth image is available and
        # the two-face pose is not. The centre needs no face semantics: a cube is
        # symmetric about it, so this is the honest 3D target for a top grasp.
        self.declare_parameter('depth_pose_enabled', True)
        self.declare_parameter('depth_pose_minimum_valid_fraction', 0.30)
        self.declare_parameter('depth_pose_maximum_mad_m', 0.02)
        #: Constant offset added to the depth-centre position, in the published
        #: frame. Calibrated against ground truth by sim/eval_cube_recognition.py;
        #: the residual is systematic, so it belongs in one parameter rather than
        #: being guessed per consumer.
        self.declare_parameter('depth_pose_bias_m', '0,0,0')
        #: Off for the calibration sweep, so the raw bias is measurable.
        self.declare_parameter('depth_pose_half_side', True)
        #: How long a pose stays usable when the current frame produces none. A
        #: cube at 1 m is seen a few times a second and the arm does not move that
        #: fast, so a short memory beats publishing nothing (measured: the frame
        #: after a base move often has no solve at all).
        self.declare_parameter('pose_memory_max_age_s', 1.5)
        # Normalised perception ROI for the depth blob ("x0,y0,x1,y1" in 0..1).
        # The arm is inside the same working-distance band as the cube, so the
        # two merge into one blob unless the search is bounded; a real
        # deployment bounds it the same way.
        self.declare_parameter('bbox_roi', '')

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
        self.enable_visibility_gate = bool(
            self.get_parameter('enable_visibility_gate').value
        )
        self.enable_face_color_selection = bool(
            self.get_parameter('enable_face_color_selection').value
        )
        self.face_color_minimum_chroma = float(
            self.get_parameter('face_color_minimum_chroma').value
        )
        self.face_color_minimum_hue_margin_deg = float(
            self.get_parameter('face_color_minimum_hue_margin_deg').value
        )
        self.perception_backend = str(
            self.get_parameter('perception_backend').value
        )
        self.bbox_source = str(self.get_parameter('bbox_source').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.depth_scale_m = float(self.get_parameter('depth_scale_m').value)
        self.bbox_roi = self._parse_roi(
            str(self.get_parameter('bbox_roi').value)
        )
        self.latest_depth = None
        self.depth_frame_id = ''
        self.latest_depth_stamp = None
        self.depth_camera_info = None
        self.latest_image_width = 0
        self.latest_image_height = 0
        self.latest_image = None
        self.pose_memory = None
        self.depth_pose_enabled = bool(
            self.get_parameter('depth_pose_enabled').value
        )
        if self.bbox_source == 'depth_blob' and not self.depth_topic:
            raise RuntimeError(
                'bbox_source=depth_blob requires the depth_topic parameter'
            )
        if self.depth_topic and self.depth_pose_enabled:
            self.create_subscription(
                Image, self.depth_topic, self.depth_callback, 1
            )
            for info_topic in self._depth_info_candidates(self.depth_topic):
                self.create_subscription(
                    CameraInfo, info_topic, self.depth_info_callback, 1
                )
                break
        elif self.bbox_source == 'depth_blob':
            self.create_subscription(
                Image, self.depth_topic, self.depth_callback, 1
            )

        self.bridge = CvBridge()
        self.model_sha256 = ''
        self.model = None
        # Load the detector only when the route needs it: with bbox_source
        # depth_blob and backend color_frontend nothing here uses it, and the
        # module must run in an image without ultralytics.
        # The detector is only needed when it supplies the box itself.
        needs_detector = self.bbox_source == 'detector'
        if needs_detector:
            if YOLO is None:
                raise RuntimeError(
                    'ultralytics is not installed but the detector is required'
                )
            self.model_sha256 = self._compute_file_sha256(self.model_path)
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
        self.get_logger().info(
            f'visibility gate: {self.enable_visibility_gate}, '
            f'face colour selection: {self.enable_face_color_selection}'
        )
        self.get_logger().info(
            f'perception backend: {self.perception_backend}'
        )
        if self.debug_image_path:
            self.get_logger().info(
                f'write debug image: {self.debug_image_path}'
            )
        if self.model is None:
            # Say so explicitly: a stale "model: ..." line in a no-detector run
            # reads as if a checkpoint were loaded.
            self.get_logger().info(
                'no detector loaded (model-free route, bbox from '
                f'{self.bbox_source})'
            )
        else:
            self.get_logger().info(f'model: {self.model_path}')
            if self.model_sha256:
                self.get_logger().info(
                    f'model sha256: {self.model_sha256}'
                )

    @staticmethod
    def _parse_roi(text):
        """Parse "x0,y0,x1,y1" (normalised) into a tuple, or None."""
        if not text.strip():
            return None
        parts = [part.strip() for part in text.split(',')]
        if len(parts) != 4:
            raise RuntimeError(f'bbox_roi must be "x0,y0,x1,y1", got {text!r}')
        values = [float(part) for part in parts]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise RuntimeError('bbox_roi values must be normalised 0..1')
        if values[2] <= values[0] or values[3] <= values[1]:
            raise RuntimeError('bbox_roi must be x0<x1 and y0<y1')
        return tuple(values)

    def depth_callback(self, msg):
        """Keep the latest depth frame in metres."""
        try:
            height, width = msg.height, msg.width
            if msg.encoding in ('32FC1', '64FC1'):
                raw = np.frombuffer(msg.data, dtype=np.float32 if
                                    msg.encoding == '32FC1' else np.float64)
                depth = raw.reshape(height, width).astype(np.float32)
            elif msg.encoding in ('16UC1', 'mono16'):
                raw = np.frombuffer(msg.data, dtype=np.uint16)
                depth = (raw.reshape(height, width).astype(np.float32)
                         * self.depth_scale_m)
            else:
                self.get_logger().warning(
                    f'unsupported depth encoding: {msg.encoding}'
                )
                return
            self.latest_depth = depth
            self.depth_frame_id = msg.header.frame_id
            self.latest_depth_stamp = msg.header.stamp.sec + (
                msg.header.stamp.nanosec * 1e-9)
        except Exception as error:  # noqa: BLE001 - never kill the callback
            self.get_logger().warning(f'depth decode failed: {error}')

    @staticmethod
    def _depth_info_candidates(depth_topic):
        """CameraInfo topics a depth sensor may publish, most specific first."""
        if depth_topic.endswith('/image_raw'):
            base = depth_topic[:-len('/image_raw')]
            return (base + '/camera_info',)
        return (depth_topic + '/camera_info',)

    def depth_info_callback(self, msg):
        """Keep the depth intrinsics (the colour ones may be a different camera)."""
        self.depth_camera_info = msg

    def _remember_pose(self, detection, timestamp):
        """Keep the last accepted pose, and reuse it when this frame produced none.

        A box with no solve is the common failure -- the depth centre needs a valid
        depth blob and the face solve needs a complete quad -- and publishing nothing
        for those frames costs a consumer a whole cycle. The reused pose is tagged, so
        a consumer can tell a remembered pose from a measured one.
        """
        pose = detection.get('pose_3d')
        if pose is not None:
            self.pose_memory = (dict(pose), timestamp)
            return
        remembered = self._pose_from_memory(timestamp)
        if remembered is None:
            return
        detection['pose_3d'] = remembered
        detection['pose_evidence']['pose_status'] = 'memory'
        detection['pose_evidence']['pose_reason'] = (
            'no solve in this frame; the last accepted pose is reused'
        )

    def _pose_from_memory(self, timestamp):
        """The last accepted pose, if it is fresh enough to stand in."""
        if self.pose_memory is None:
            return None
        pose, stamp = self.pose_memory
        age = abs(float(timestamp) - float(stamp))
        if age > float(self.get_parameter('pose_memory_max_age_s').value):
            return None
        remembered = dict(pose)
        remembered['geometry_id'] = '%s-memory' % pose.get('geometry_id', 'pose')
        remembered['from_memory'] = True
        remembered['memory_age_s'] = round(age, 3)
        return remembered

    def depth_age_seconds(self, timestamp):
        """How old the depth frame is compared with the colour frame."""
        if self.latest_depth_stamp is None:
            return float('nan')
        return abs(float(timestamp) - float(self.latest_depth_stamp))

    def _depth_centre_pose(self, box_xyxy, timestamp):
        """Cube centre from the box plus depth, in the depth optical frame.

        Only the middle of the box is sampled: its edges include the floor, whose
        depth would drag the estimate backwards. A wide spread or a mostly
        invalid patch means the box straddles an edge, and the answer is withheld
        rather than guessed.
        """
        depth = self.latest_depth
        info = self.depth_camera_info
        if depth is None or info is None:
            self.get_logger().debug(
                'depth centre: no depth=%s no info=%s'
                % (depth is None, info is None))
            return None
        rows, cols = depth.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in box_xyxy)
        inset_x = 0.25 * (x1 - x0)
        inset_y = 0.25 * (y1 - y0)
        # the box is in the colour image; scale into the depth image
        scale_x = cols / float(self.latest_image_width or cols)
        scale_y = rows / float(self.latest_image_height or rows)
        u0 = max(0, int((x0 + inset_x) * scale_x))
        u1 = min(cols, int((x1 - inset_x) * scale_x) + 1)
        v0 = max(0, int((y0 + inset_y) * scale_y))
        v1 = min(rows, int((y1 - inset_y) * scale_y) + 1)
        if u1 <= u0 or v1 <= v0:
            return None
        window = depth[v0:v1, u0:u1]
        valid = window[np.isfinite(window) & (window > 0.02) & (window < 10.0)]
        valid_fraction = valid.size / float(window.size) if window.size else 0.0
        if valid.size == 0 or (
                valid_fraction
                < float(self.get_parameter(
                    'depth_pose_minimum_valid_fraction').value)):
            self.get_logger().debug(
                'depth centre: valid fraction %.2f (window %dx%d, finite %d)'
                % (valid_fraction, window.shape[1], window.shape[0], valid.size))
            return None
        distance = float(np.median(valid))
        spread = float(np.median(np.abs(valid - distance)))
        if spread > float(self.get_parameter('depth_pose_maximum_mad_m').value):
            self.get_logger().debug(
                'depth centre: spread %.3f m at %.3f m rejected'
                % (spread, distance))
            return None
        k = info.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None
        u = 0.5 * (x0 + x1) * scale_x
        v = 0.5 * (y0 + y1) * scale_y
        point = [(u - cx) * distance / fx, (v - cy) * distance / fy, distance]
        # The samples land on the face facing the camera, so the centre is half a
        # side *further along the measured ray*. Measured against the robot's real
        # pose the uncorrected error is a constant 2.94-3.02 cm at 0.4-1.0 m, i.e.
        # exactly one half side (0.057/2 = 2.85 cm); the earlier 1-8 cm swings came
        # from assuming base_link sat at x = 0.02 when it settles near 0.00.
        # Scaling the ray point is exact for a fronto-parallel face at any pitch:
        # adding a fixed vector instead was measured to move the point backwards.
        if self.get_parameter('depth_pose_half_side').value:
            half_side = 0.5 * float(
                self.get_parameter('cube_side_length_m').value)
            scale = (distance + half_side) / distance
            point = [value * scale for value in point]
        bias = [float(value) for value in
                str(self.get_parameter('depth_pose_bias_m').value).split(',')]
        if len(bias) == 3:
            point = [point[i] + bias[i] for i in range(3)]
        return {
            'geometry_id': 'depth-centre-v1',
            'frame_id': info.header.frame_id or self.depth_frame_id,
            'timestamp': timestamp,
            'side_length_m': float(
                self.get_parameter('cube_side_length_m').value
            ),
            'position_m': point,
            'orientation_xyzw': [0.0, 0.0, 0.0, 1.0],
            'reprojection_error_px': None,
            'correspondence_count': int(valid.size),
            'inlier_count': int(valid.size),
            'inlier_mask': [],
            'color_fixed_semantics_resolved': False,
            'depth_spread_m': spread,
            'perception_backend': 'color_frontend',
        }

    def camera_info_callback(self, msg):
        """Keep the latest static color-camera calibration."""
        self.camera_info = msg

    def image_callback(self, msg):
        self.latest_image = None
        self.latest_image_width = int(msg.width)
        self.latest_image_height = int(msg.height)
        now = time.time()

        # 毎フレーム推論すると重いので、1秒に1回だけ推論
        if now - self.last_time < self.interval_sec:
            return

        self.last_time = now

        # No detector at all: the depth blob supplies the box and the rest is
        # geometry + colour. This is the intended route on the real robot.
        if (self.perception_backend == 'color_frontend'
                and self.bbox_source != 'detector'):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='bgr8'
                )
            except Exception as error:  # noqa: BLE001
                self.get_logger().error(f'image decode failed: {error}')
                return
            self.latest_image = cv_image
            detections = self._publish_detectionless_frame(
                cv_image, msg, now
            )
            if detections:
                source = detections[0]['pose_evidence'].get(
                    'bbox_source', '?')
                self.get_logger().info(
                    'published %d model-free detections (%s)'
                    % (len(detections), source)
                )
            return

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

                    if self.perception_backend == 'color_frontend':
                        detection = self._apply_model_free_pose(
                            cv_image, detection, msg, now
                        )
                    else:
                        pose_3d, pose_debug = self._estimate_pose_3d_with_debug(
                            msg,
                            now,
                            detection['keypoints_xy'],
                            detection['keypoints_confidence']
                        )
                        detection['pose_3d_debug'] = pose_debug
                        if pose_3d is not None:
                            detection['pose_3d'] = pose_3d

                    detection = attach_pose_evidence(
                        cv_image,
                        self.camera_info,
                        detection,
                        side_length_m=float(
                            self.get_parameter(
                                'cube_side_length_m'
                            ).value
                        ),
                        frame_id=(
                            self.camera_info.header.frame_id
                            or msg.header.frame_id
                        ),
                        timestamp=now,
                        enable_visibility_gate=self.enable_visibility_gate,
                        enable_face_color_selection=(
                            self.enable_face_color_selection
                        ),
                        minimum_keypoint_confidence=float(
                            self.get_parameter(
                                'keypoint_confidence'
                            ).value
                        ),
                        maximum_reprojection_error_px=float(
                            self.get_parameter(
                                'max_reprojection_error_px'
                            ).value
                        ),
                        minimum_chroma=self.face_color_minimum_chroma,
                        minimum_hue_margin_deg=(
                            self.face_color_minimum_hue_margin_deg
                        ),
                    )

                    detections.append(detection)

            result_data = {
                'timestamp': now,
                'model_path': self.model_path,
                'model_sha256': self.model_sha256,
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

    def _single_view_pose(self, bbox_xyxy, image_msg, timestamp):
        """Solve the visible face; None when the frame does not support it."""
        if self.camera_info is None or self.latest_image is None:
            return None
        pose, evidence = estimate_face_pose(
            self.latest_image,
            tuple(float(value) for value in self.camera_info.k),
            bbox_xyxy,
            side_length_m=float(self.get_parameter('cube_side_length_m').value),
            maximum_reprojection_error_px=6.0,
        )
        return {
            'geometry_id': 'face-quad-v1',
            'frame_id': image_msg.header.frame_id,
            'timestamp': timestamp,
            'side_length_m': float(self.get_parameter('cube_side_length_m').value),
            'position_m': list(pose['position_m']),
            'orientation_xyzw': list(pose['orientation_xyzw']),
            'reprojection_error_px': pose['reprojection_error_px'],
            'correspondence_count': 4,
            'inlier_count': 4,
            'inlier_mask': [],
            'color_fixed_semantics_resolved': bool(
                pose['color_fixed_semantics_resolved']),
            'visible_face': pose['visible_face'],
            'face_color_label': evidence['face_color_label'],
            'bbox_source': evidence['quad_source'],
            'perception_backend': 'color_frontend',
        }

    def _model_free_pose_dict(self, pose, frame_id, timestamp):
        """Build the /cube_pose_result pose payload for one model-free pose."""
        return {
            'geometry_id': 'cube-axis-v1',
            'frame_id': frame_id,
            'timestamp': timestamp,
            'side_length_m': float(
                self.get_parameter('cube_side_length_m').value
            ),
            'position_m': list(pose.translation_m),
            'orientation_xyzw': list(pose.orientation_xyzw),
            'reprojection_error_px': pose.reprojection_error_px,
            'correspondence_count': pose.correspondence_count,
            'inlier_count': pose.inlier_count,
            'inlier_mask': list(pose.inlier_mask),
            'color_fixed_semantics_resolved': (
                pose.color_fixed_semantics_resolved
            ),
            'hypothesis_color_margin': pose.color_margin,
            'hypothesis_color_agreement': pose.color_agreement,
            'visible_face_colors': dict(pose.visible_face_colors),
            'label_swaps': list(pose.label_swaps),
            'bbox_source': pose.bbox_source,
            'perception_backend': 'color_frontend',
        }

    def _model_free_confidence(self, pose):
        """Geometry quality as a detector-like confidence.

        Colour resolution is deliberately not folded in: an unresolved cube is
        still something to approach, and withholding the approach would prevent
        the second face from ever coming into view. The flag travels in pose_3d.
        """
        geometry = 1.0 / (1.0 + float(pose.reprojection_error_px) / 4.0)
        return float(max(0.35, min(1.0, geometry * max(0.5, pose.bbox_quality))))

    def _publish_detectionless_frame(self, cv_image, image_msg, timestamp):
        """Run the model-free route with a depth-supplied box, no detector."""
        frame_id = image_msg.header.frame_id
        detections = []
        if self.bbox_source == 'depth_blob' and self.latest_depth is None:
            return detections

        # The bounding box is perception in its own right: the behaviour tree
        # approaches on box + confidence and only optionally uses pose_3d
        # (ccc.xml runs with use_pose_3d=0). Requiring a full pose before
        # publishing anything is what made the cube undetectable at 1 m, where
        # the two-face geometry has too few pixels but the blob is obvious.
        height, width = cv_image.shape[:2]
        roi = None
        if self.bbox_roi is not None:
            x0, y0, x1, y1 = self.bbox_roi
            roi = (x0 * width, y0 * height, x1 * width, y1 * height)
        try:
            # The source decides what it needs and ignores the rest (see
            # cube_bbox_source for what each one accepts).
            box = choose_bbox(
                self.bbox_source,
                image_bgr=cv_image,
                depth_image=self.latest_depth,
                roi_xyxy=roi,
            )
        except BboxSourceError as error:
            self.get_logger().debug(f'bbox rejected: {error}')
            return detections

        detection = {
            'class_name': 'cube',
            'confidence': float(max(0.35, min(1.0, box.quality))),
            'box_xyxy': [float(v) for v in box.box_xyxy],
            'keypoints_xy': [],
            'keypoints_confidence': [],
            'pose_evidence': {
                'backend': 'color_frontend',
                'bbox_source': box.source,
                'bbox_quality': float(box.quality),
                'bbox_detail': dict(box.detail),
            },
        }

        if True:  # keep the indentation of the original block below
            try:
                pose = estimate_model_free_pose(
                    cv_image,
                    self.camera_info.k if self.camera_info else (),
                    depth_image=self.latest_depth,
                    bbox_source='depth_blob',
                    roi_xyxy=roi,
                    side_length_m=float(
                        self.get_parameter('cube_side_length_m').value
                    ),
                    maximum_reprojection_error_px=float(
                        self.get_parameter('max_reprojection_error_px').value
                    ),
                    minimum_chroma=self.face_color_minimum_chroma,
                    minimum_hue_margin_deg=(
                        self.face_color_minimum_hue_margin_deg
                    ),
                )
            except ModelFreePoseError as error:
                # Box-only detection: the tree can approach, the pose stays
                # unavailable and is reported as such.
                detection['pose_evidence']['pose_status'] = 'rejected'
                detection['pose_evidence']['pose_reason'] = str(error)
                # At the real mount only one face is visible, so the single-view
                # face solve is the route that produces an orientation at all: the
                # face's colour decides which face it is and therefore the yaw.
                # Measured 1.4-2.4 cm on position with reprojection under 0.5 px,
                # against 1.0-1.3 cm for the depth centre -- comparable position,
                # but with the rotation the depth route cannot give.
                face_pose = None
                try:
                    # The visible face's own box, not the cube's: the label the solve
                    # reads comes from the middle of the box, and the cube's box can
                    # put that on a tile border (measured 2026-10-24).
                    face_pose = self._single_view_pose(
                        face_box_from_color(cv_image, box.box_xyxy)
                        or box.box_xyxy,
                        image_msg, timestamp)
                except FacePoseError as error:
                    # A refusal is per-frame and expected: the solve refuses whenever
                    # the quad is incomplete or a vertex is behind the camera, which is
                    # what the depth centre is the fallback for. Measured at 0.24-0.30 m
                    # the face solve does succeed with the face identified, so a refusal
                    # is about this frame, not about the range.
                    self.get_logger().debug('face pose refused: %s' % error)
                # The depth centre keeps the primary slot on measurement: through this
                # node at 0.4/0.6/0.8 m it reports 0.10/0.31/0.50 cm and the face solve
                # 0.33/0.59/0.53 cm -- comparable, with the depth centre ahead over the
                # range a grasp works in and the two swapping order by 0.8 m. The face
                # solve ships beside it rather than replacing it, for the orientation it
                # carries, which cannot align anything by itself: it reports which face
                # it sees, quantised to the face grid.
                depth_pose = self._depth_centre_pose(box.box_xyxy, timestamp)
                if depth_pose is not None:
                    detection['pose_3d'] = depth_pose
                    detection['pose_evidence']['pose_status'] = 'depth_centre'
                    detection['pose_evidence']['pose_reason'] = (
                        'two-face pose unavailable; centre measured from depth'
                    )
                self.get_logger().debug(
                    'ingredients box=%s depth=%s face=%s stamp_delta=%.3f'
                    % ([round(v, 1) for v in box.box_xyxy],
                       None if depth_pose is None
                       else [round(v, 3) for v in depth_pose['position_m']],
                       None if face_pose is None
                       else [round(v, 3) for v in face_pose['position_m']],
                       self.depth_age_seconds(timestamp)))
                if face_pose is not None:
                    detection['pose_3d_face'] = face_pose
                    detection['pose_evidence']['face_pose_status'] = (
                        'accepted' if face_pose['color_fixed_semantics_resolved']
                        else 'unresolved')
                detections.append(detection)
                # The overlay reports what this frame produced, which may be a pose from
                # the fallback routes even though the two-face route refused: drawn as
                # 'bbox_only' it said no pose on every frame of a run that was publishing
                # one, which is exactly the window a reader trusts to explain a frame.
                # Draw the single-view face pose: with the face's own box it lands on
                # the cube (verified offline on recorded ccc2 frames 2026-10-24:
                # single-face, colours resolved, wireframe on the visible face). The
                # depth-centre pose is deliberately not drawn: its orientation is a
                # guess, and a wireframe that sits plausibly while being 90 deg out is
                # worse than none.
                self._publish_model_free_overlay(
                    cv_image, image_msg, timestamp,
                    pose=face_pose,
                    bbox_xyxy=box.box_xyxy,
                    status=('single_face' if face_pose is not None
                            else detection['pose_evidence'].get(
                                'pose_status', 'rejected')),
                    # Say which route drew the wireframe. Not `str(error)`: the inner
                    # `except FacePoseError as error` deletes that name at the end of
                    # its block, so referring to it here raised UnboundLocalError
                    # whenever the face solve refused (found 2026-10-24: the perception
                    # node died on that path).
                    reason=('pose from the single visible face'
                            if face_pose is not None
                            else detection['pose_evidence'].get('pose_reason', '')),
                    bbox_source=box.source,
                )
            else:
                detection['confidence'] = self._model_free_confidence(pose)
                detection['box_xyxy'] = [float(v) for v in pose.bbox_xyxy]
                detection['pose_3d'] = self._model_free_pose_dict(
                    pose, frame_id, timestamp
                )
                detection['pose_evidence'].update({
                    'bbox_source': pose.bbox_source,
                    'bbox_quality': float(pose.bbox_quality),
                    'label_swaps': list(pose.label_swaps),
                    'visible_faces': list(pose.visible_faces),
                    'faces': pose.face_evidence,
                    'pose_status': 'accepted',
                })
                detections.append(detection)
                self._publish_model_free_overlay(
                    cv_image, image_msg, timestamp, pose=pose, status=(
                        'accepted' if pose.color_fixed_semantics_resolved
                        else 'unresolved'
                    ),
                    reason='' if pose.color_fixed_semantics_resolved
                    else pose.unresolved_reason,
                )
        for detection in detections:
            self._remember_pose(detection, timestamp)
        message = String()
        message.data = json.dumps({
            'timestamp': timestamp,
            'model_path': '',
            'model_sha256': '',
            'detections': detections,
        }, ensure_ascii=False)
        self.publisher.publish(message)
        return detections

    def _publish_model_free_overlay(
        self, cv_image, image_msg, timestamp, *, pose=None, status='unknown',
        reason='', bbox_source='', bbox_xyxy=None,
    ):
        """Publish the annotated frame rqt shows, for accepted and rejected ones.

        Called on every processed frame, not only on success: the most useful
        thing this window does is explain a frame that produced no pose.
        """
        if self.debug_publisher is None and not self.debug_image_path:
            return
        try:
            import cv2
        except ImportError:
            self.get_logger().warning('overlay skipped: OpenCV unavailable')
            return

        camera_matrix = (
            tuple(float(value) for value in self.camera_info.k)
            if self.camera_info is not None else None
        )
        pose_dict = None
        face_regions = None
        observed = None
        visible_faces = None
        resolved = None
        swaps = None
        rms = None
        source = bbox_source
        quality = None
        box = None
        if isinstance(pose, dict):
            # a route that carries its pose as a dict (the single-view face solve)
            pose_dict = {
                'position_m': list(pose['position_m']),
                'orientation_xyzw': list(pose['orientation_xyzw']),
            }
        elif pose is not None:
            pose_dict = {
                'position_m': list(pose.translation_m),
                'orientation_xyzw': list(pose.orientation_xyzw),
            }
            face_regions = {
                slot: evidence.get('quad')
                for slot, evidence in (pose.face_evidence or {}).items()
                if isinstance(evidence, dict) and evidence.get('quad')
            }
            observed = dict(pose.visible_face_colors)
            visible_faces = tuple(pose.visible_faces)
            resolved = bool(pose.color_fixed_semantics_resolved)
            swaps = tuple(pose.label_swaps)
            rms = float(pose.reprojection_error_px)
            source = pose.bbox_source
            quality = float(pose.bbox_quality)
            box = pose.bbox_xyxy
        elif bbox_xyxy is not None:
            box = [float(value) for value in bbox_xyxy]
            source = source or 'box only'
        elif self.latest_depth is not None:
            source = source or 'depth_blob'

        annotated = draw_cube_overlay(
            cv_image,
            camera_matrix=camera_matrix,
            pose=pose_dict,
            bbox_xyxy=box,
            bbox_source=source,
            bbox_quality=quality,
            face_regions=face_regions,
            observed_colors=observed,
            visible_faces=visible_faces,
            status=status,
            reason=reason,
            reprojection_error_px=rms,
            color_resolved=resolved,
            label_swaps=swaps,
            side_length_m=float(
                self.get_parameter('cube_side_length_m').value
            ),
        )

        if self.debug_publisher is not None:
            message = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            message.header = image_msg.header
            self.debug_publisher.publish(message)

        if self.debug_image_path and pose is not None:
            if not cv2.imwrite(self.debug_image_path, annotated):
                self.get_logger().warning(
                    f'failed to write debug image: {self.debug_image_path}'
                )

    def _apply_model_free_pose(self, cv_image, detection, image_msg, timestamp):
        """Fill pose evidence from the model-free geometry+colour route."""
        camera_info = self.camera_info
        if camera_info is None:
            detection['pose_3d_debug'] = {
                'status': 'unavailable', 'reason': 'CameraInfo unavailable',
            }
            return detection
        try:
            pose = estimate_model_free_pose(
                cv_image,
                camera_info.k,
                detection['box_xyxy'],
                side_length_m=float(
                    self.get_parameter('cube_side_length_m').value
                ),
                maximum_reprojection_error_px=float(
                    self.get_parameter('max_reprojection_error_px').value
                ),
                minimum_chroma=self.face_color_minimum_chroma,
                minimum_hue_margin_deg=(
                    self.face_color_minimum_hue_margin_deg
                ),
            )
        except ModelFreePoseError as error:
            detection['pose_3d_debug'] = {
                'status': 'rejected',
                'reason': str(error),
                'backend': 'color_frontend',
            }
            return detection

        frame_id = camera_info.header.frame_id or image_msg.header.frame_id
        evidence = {
            'backend': 'color_frontend',
            'label_swaps': list(pose.label_swaps),
            'visible_faces': list(pose.visible_faces),
            'hypothesis_count': pose.hypothesis_count,
            'faces': pose.face_evidence,
        }
        detection['pose_evidence'] = {'model_free': evidence}
        if not pose.color_fixed_semantics_resolved:
            detection['pose_3d_debug'] = {
                'status': 'unresolved',
                'reason': pose.unresolved_reason,
                'backend': 'color_frontend',
                'observed': dict(pose.visible_face_colors),
            }
            return detection

        detection['pose_3d'] = {
            'geometry_id': 'cube-axis-v1',
            'frame_id': frame_id,
            'timestamp': timestamp,
            'side_length_m': float(
                self.get_parameter('cube_side_length_m').value
            ),
            'position_m': list(pose.translation_m),
            'orientation_xyzw': list(pose.orientation_xyzw),
            'reprojection_error_px': pose.reprojection_error_px,
            'correspondence_count': pose.correspondence_count,
            'inlier_count': pose.inlier_count,
            'inlier_mask': list(pose.inlier_mask),
            'color_fixed_semantics_resolved': True,
            'hypothesis_color_margin': pose.color_margin,
            'hypothesis_color_agreement': pose.color_agreement,
            'visible_face_colors': dict(pose.visible_face_colors),
            'label_swaps': list(pose.label_swaps),
            'perception_backend': 'color_frontend',
        }
        detection['pose_3d_debug'] = {
            'status': 'accepted',
            'reason': '',
            'backend': 'color_frontend',
        }
        return detection

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

    def _compute_file_sha256(self, path):
        """Return a stable checkpoint identity without changing inference."""
        digest = hashlib.sha256()
        try:
            with open(path, 'rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
        except OSError as error:
            self.get_logger().warning(
                f'model sha256 unavailable for {path}: {error}'
            )
            return ''
        return digest.hexdigest()

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
        if self.model_sha256:
            self._draw_text(
                cv2,
                debug_image,
                f'model={self.model_sha256[:12]}',
                (10, 18),
                scale=0.40
            )
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
