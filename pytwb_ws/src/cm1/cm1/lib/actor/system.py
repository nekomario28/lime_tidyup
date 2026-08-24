from typing import List
from math import radians, degrees
import numpy as np
import math
import time
import os
from operator import add
import json
from std_msgs.msg import String

import pickle

from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from tf2_ros.buffer import Buffer 
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import Twist
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates, LinkStates

import transforms3d
from pymoveit2 import MoveIt2, GripperInterface
from vector_map import get_map, get_map_ROS, SimulationSpace, init_visualize
import pyrealsense2 as rs

from ros_actor import actor, SubSystem
from .approach_action import ApproachAction
from .cognitive import CognitiveNetwork
from ..cube_depth import (
    CubeDepthError,
    consistent_depth_median,
    estimate_cube_depth,
)
from ..cube_pose import accepted_pose_distance
from .manipulator import ManipulatorNetwork
from .tools import Tools
from .voice import VoiceNetwork

#######################################################
#
#   Turtlebot3 dependent definitions
#
#######################################################

MOVE_GROUP_ARM: str = "arm"
MOVE_GROUP_GRIPPER: str = "gripper"

OPEN_GRIPPER_JOINT_POSITIONS: List[float] = [0.019, 0.019]
# CLOSED_GRIPPER_JOINT_POSITIONS: List[float] = [0.007, 0.007]
CLOSED_GRIPPER_JOINT_POSITIONS: List[float] = [0.015, 0.015]

def joint_names() -> List[str]:
    return [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

def base_link_name() -> str:
    return "base_link"

def end_effector_name() -> str:
    return "end_effector_link"

def gripper_joint_names() -> List[str]:
    return [
        "gripper_left_joint",
        "gripper_right_joint",
    ]

class Tb3(SubSystem):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.add_subsystem('navigation', Tb3NavigationSystem)
        self.add_subsystem('manipulator', Tb3ManipulatorSystem)
        self.add_subsystem('camera', Tb3CameraSystem)
        self.register_subscriber('model_states', ModelStates, 'model_states', 10)
        self.register_subscriber('link_states', LinkStates, 'link_states', 10)
        self.add_network(Tools)
        self.add_network(VoiceNetwork)
        node = self.get_value('node')
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, node)
        self.set_value('tf_buffer', tf_buffer)
    
    def get_trans(self, from_frame, to_frame):
        tf_buffer = self.get_value('tf_buffer')
        if not tf_buffer.can_transform(
                to_frame,
                from_frame,
                rclpy.time.Time()):
            return None
        tf = None
        try:
            tf = tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time())
        except TransformException as ex:
            print(ex)
        return tf

    @actor
    def map_trans(self, src="camera_link"):
        while True:
            ret = self.get_trans(src, "map")
            if ret: return ret
            self.run_actor('sleep', 1)
    
    @actor
    def var_trans(self, target="link1"):
        while True:
            ret = self.get_trans("camera_link", target)
            if ret: return ret
            self.run_actor('sleep', 1)
    
    @actor
    def uni_trans(self, src="camera_link", target="map"):
        while True:
            ret = self.get_trans(src, target)
            if ret: return ret
            self.run_actor('sleep', 1)
    
    @actor
    def base_trans(self):
        while True:
            ret = self.get_trans("camera_link", "base_link")
            if ret: return ret
            self.run_actor('sleep', 1)
    
    @actor
    def gripper_trans(self):
        while True:
            ret = self.get_trans("link5", "base_link")
            if ret: return ret
            self.run_actor('sleep', 1)
    
    @actor
    def sleep(self, st):
        time.sleep(st)
    
class Tb3NavigationSystem(SubSystem):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.register_action('navigate', NavigateToPose, "/navigate_to_pose")
        self.register_publisher('motor', Twist, 'cmd_vel', 10)
        self.register_subscriber('odom', Odometry, 'odom', 10)
        self.register_subscriber('cube_pose_result',String,'/cube_pose_result',1)
        self.add_network(ApproachAction)
        self.set_value('current_pose', (0.0, 0.0, 0.0))

    def _cube_depth_from_box(
        self,
        box,
        rgb_width=848.0,
        rgb_height=480.0,
        roi_fraction=0.0,
        minimum_valid_ratio=0.0,
        maximum_mad_m=10.0
    ):
        """
        YOLOのバウンディングボックス内側から品質付きDepthを取得する。

        引数:
            box:
                [x_min, y_min, x_max, y_max]

            rgb_width, rgb_height:
                YOLOに使ったRGB画像の大きさ

        戻り値:
            距離[m]
            取得失敗時はNone
        """

        # -----------------------------
        # 1. Depth画像を受信
        # -----------------------------
        try:
            depth_msg = self.run_actor('depth')
        except Exception as error:
            print(
                f'_cube_depth_from_box: '
                f'Depthメッセージ取得失敗: {error}'
            )
            return None

        # -----------------------------
        # 2. ROS画像をNumPy配列へ変換
        # -----------------------------
        try:
            bridge = CvBridge()

            depth_image = bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding='passthrough'
            )

        except CvBridgeError as error:
            print(
                f'_cube_depth_from_box: '
                f'CvBridge変換失敗: {error}'
            )
            return None

        except Exception as error:
            print(
                f'_cube_depth_from_box: '
                f'Depth画像変換失敗: {error}'
            )
            return None

        if depth_image is None or depth_image.size == 0:
            print(
                '_cube_depth_from_box: '
                'Depth画像が空です'
            )
            return None

        encoding = getattr(
            depth_msg,
            'encoding',
            ''
        )
        try:
            estimate = estimate_cube_depth(
                depth_image,
                box,
                encoding=encoding,
                rgb_width=rgb_width,
                rgb_height=rgb_height,
                roi_fraction=roi_fraction,
                minimum_valid_ratio=minimum_valid_ratio,
                maximum_mad_m=maximum_mad_m,
            )
        except CubeDepthError as error:
            print(
                f'_cube_depth_from_box: '
                f'Depth品質不足: {error}'
            )
            return None

        depth_height, depth_width = depth_image.shape[:2]
        print(
            f'_cube_depth_from_box: '
            f'encoding={encoding}, '
            f'depth_size={depth_width}x{depth_height}, '
            f'roi={estimate.roi_xyxy}, '
            f'samples={estimate.sample_count}, '
            f'valid_ratio={estimate.valid_ratio:.3f}, '
            f'mad={estimate.mad_m:.3f}m, '
            f'distance={estimate.distance_m:.3f}m'
        )

        return estimate.distance_m

    def _read_best_cube_detection(self):
        """
        /cube_pose_resultからYOLOの検出結果を取得し、
        confidenceが最も高い検出を返す。

        戻り値:
            (best_detection, best_confidence)

            検出できなかった場合:
                (None, 0.0)
        """

        msg = self.run_actor('cube_pose_result')

        try:
            data = json.loads(msg.data)

        except (
            json.JSONDecodeError,
            AttributeError,
            TypeError
        ) as error:
            print(
                f'_read_best_cube_detection: '
                f'JSON解析失敗: {error}'
            )
            return None, 0.0

        detections = data.get('detections', [])

        best_detection = None
        best_confidence = 0.0

        for detection in detections:
            try:
                confidence = float(
                    detection.get('confidence', 0.0)
                )
            except (
                TypeError,
                ValueError
            ):
                continue

            if confidence > best_confidence:
                best_confidence = confidence
                best_detection = detection

        return best_detection, best_confidence
    
    def create_move_base_goal(self, x, y, theta):
        """ Creates a MoveBaseGoal message from a 2D navigation pose """
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        node = self.get_value('node')
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        quat = transforms3d.euler.euler2quat(0, 0, theta)
        goal.pose.pose.orientation.w = quat[0]
        goal.pose.pose.orientation.x = quat[1]
        goal.pose.pose.orientation.y = quat[2]
        goal.pose.pose.orientation.z = quat[3]
        return goal
    @actor
    def rotate_angle(self, angle_rad, angular_speed=0.35):
        """
        指定された角度だけ、その場で回転する。

        angle_rad:
            正数なら左回転
            負数なら右回転
        """

        angle_rad = float(angle_rad)
        angular_speed = abs(float(angular_speed))

        if abs(angle_rad) < 0.01:
            return True

        move_msg = Twist()

        if angle_rad > 0:
            move_msg.angular.z = angular_speed
        else:
            move_msg.angular.z = -angular_speed

        # 回転時間 = 回転角度 ÷ 角速度
        rotate_time = abs(angle_rad) / angular_speed

        print(
            f'回転開始: '
            f'角度={math.degrees(angle_rad):.1f}度, '
            f'時間={rotate_time:.2f}秒'
        )

        self.run_actor('motor', move_msg)
        self.run_actor('sleep', rotate_time)
        self.run_actor('motor', Twist())

        print('回転終了')
        return True

    @actor
    def goto(self, x, y, theta):
        goal = self.create_move_base_goal(x, y, theta)
        result = self.run_actor('navigate', goal)
        self.set_value('current_pose', (x, y, theta))
#        return (result.status == GoalStatus.STATUS_SUCCEEDED)
        return True

# #       2026-3-18 taga 追加 小刻み改善用判定

#     @actor
#     def control(self, x1, x2, y1, y2, theta1, theta2):
#         self.x1 = x1

#         #rはロボットの現在の座標が欲しい。その座標がある一定の範囲にあれば、動きを止める
#         if x1 <= r <= x2 and

    @actor
    def goto_deg(self, x, y, degree):
        rad = radians(degree)
        theta = round(rad, 2)
        x, y = float(x), float(y)

        goal = self.create_move_base_goal(x, y, theta)
        result = self.run_actor('navigate', goal)
        self.set_value('current_pose', (x, y, theta))
#        return (result.status == GoalStatus.STATUS_SUCCEEDED)
        return True

    
    @actor
    def migrate(self, dx=0.0, dy=0.0, dtheta=0.0):
        pose = self.get_value('current_pose')
        pose = list(map(add, pose, (dx, dy, dtheta)))
        self.run_actor('goto', *pose)
    
    
    @actor
    def search_cube(
            self,
            threshold=0.20,
            center_tolerance=120.0,
            image_width=848.0
        ):
        """
        キューブを探し、ある程度中央に入ったら探索完了にする。
        """

        threshold = float(threshold)
        center_tolerance = float(center_tolerance)
        image_width = float(image_width)

        image_center_x = image_width / 2.0
        center_count = 0
        while True:
            msg = self.run_actor('cube_pose_result')

            try:
                data = json.loads(msg.data)

            except (
                json.JSONDecodeError,
                AttributeError,
                TypeError
            ) as error:
                print(f'cube_pose_resultの解析失敗: {error}')
                self.run_actor('sleep', 0.1)
                continue

            detections = data.get('detections', [])

            best_detection = None
            best_confidence = 0.0

            for detection in detections:
                confidence = float(
                    detection.get('confidence', 0.0)
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = detection

            print(f'confidence: {best_confidence:.3f}')

            # 検出されていない場合
            # 検出されていない場合
            if best_detection is None:
                center_count = 0

                print('キューブ未検出：回転して探索します')

                rotate_msg = Twist()
                rotate_msg.linear.x = 0.0
                rotate_msg.angular.z = 0.20

                # 0.6秒間、繰り返しcmd_velを送る
                for _ in range(6):
                    self.run_actor('motor', rotate_msg)
                    self.run_actor('sleep', 0.1)

                self.run_actor('motor', Twist())
                self.run_actor('sleep', 0.2)

                continue

            box = best_detection.get('box_xyxy', [])

            if len(box) != 4:
                continue

            x_min = float(box[0])
            x_max = float(box[2])

            cube_center_x = (x_min + x_max) / 2.0
            error_x = cube_center_x - image_center_x

            print(
                f'center_x={cube_center_x:.1f}, '
                f'error_x={error_x:.1f}'
            )

            # 信頼度が十分で、中央付近なら探索完了
            if (
                    best_confidence >= threshold
                    and abs(error_x) <= center_tolerance
                ):
                    center_count += 1

                    self.run_actor('motor', Twist())

                    print(
                        f'中央判定: {center_count}/3 '
                        f'error_x={error_x:.1f}'
                    )

                    if center_count >= 2:
                        self.set_value(
                            'cube_detection',
                            best_detection
                        )

                        print('キューブを中央に捉えました')
                        return best_detection

                    self.run_actor('sleep', 0.15)
                    continue

            else:
                center_count = 0
        

            rotate_msg = Twist()

            # 左端に見えている場合は左へ大きく回転
            if error_x < -center_tolerance:
                rotate_msg.angular.z = 0.20
                print('キューブが左端なので左へ大きく回転')

            # 右端に見えている場合は右へ大きく回転
            elif error_x > center_tolerance:
                rotate_msg.angular.z = -0.20
                print('キューブが右端なので右へ大きく回転')

            # キューブが見つからない、または信頼度不足
            else:
                rotate_msg.angular.z = 0.20
                print('信頼度不足なので探索を継続')

            # 約0.6秒間、繰り返し回転命令を送る
            for _ in range(6):
                self.run_actor('motor', rotate_msg)
                self.run_actor('sleep', 0.1)

            # 停止
            self.run_actor('motor', Twist())
            self.run_actor('sleep', 0.2)


    @actor
    def go_front_cube(
        self,
        threshold=0.50,
        stop_distance=0.30,
        forward_speed=0.03,
        turn_speed=0.20,
        target_offset_px=40.0,
        pose_max_reprojection_error_px=4.0,
        use_pose_3d=1.0,
        depth_sample_count=1.0,
        depth_roi_fraction=0.0,
        depth_minimum_valid_ratio=0.0,
        depth_maximum_mad_m=10.0,
        depth_maximum_spread_m=10.0
    ):
        """
        YOLOと品質ゲート済みDepthを取得し、

        1. キューブの横ずれから旋回秒数を計算
        2. 計算した時間だけ旋回
        3. Depth距離から前進秒数を計算
        4. 一気に前進
        5. 停止

        を行うActor。

        戻り値:
            True:
                旋回と前進が完了した、
                またはすでに停止距離以内だった

            False:
                YOLO、box_xyxy、Depthの取得に失敗した
        """

        threshold = float(threshold)
        stop_distance = float(stop_distance)
        forward_speed = abs(float(forward_speed))
        turn_speed = abs(float(turn_speed))
        target_offset_px = float(target_offset_px)
        pose_max_reprojection_error_px = float(
            pose_max_reprojection_error_px
        )
        use_pose_3d = float(use_pose_3d) >= 0.5
        depth_sample_count = max(1, int(float(depth_sample_count)))
        depth_roi_fraction = float(depth_roi_fraction)
        depth_minimum_valid_ratio = float(depth_minimum_valid_ratio)
        depth_maximum_mad_m = float(depth_maximum_mad_m)
        depth_maximum_spread_m = float(depth_maximum_spread_m)

        # 外から変更しない固定値
        horizontal_fov = 60.0
        image_width = 848.0
        image_height = 480.0
        command_interval = 0.1

        miss_count = 0

        # 速度が0だと時間計算で0除算になる
        if turn_speed == 0.0:
            print('go_front_cube: turn_speedが0です')
            return False

        if forward_speed == 0.0:
            print('go_front_cube: forward_speedが0です')
            return False

        # --------------------------------
        # 1. YOLOの検出結果を1回取得
        # --------------------------------
        # YOLOを最大5回確認する
        best_detection = None
        best_confidence = 0.0
        max_retry = 5

        for retry_count in range(1, max_retry + 1):
            best_detection, best_confidence = (
                self._read_best_cube_detection()
            )

            print(
                f'go_front_cube: YOLO確認 '
                f'{retry_count}/{max_retry}, '
                f'confidence={best_confidence:.3f}'
            )

            if (
                best_detection is not None
                and best_confidence >= threshold
            ):
                break

            self.run_actor('sleep', 0.3)

        else:
            print(
                f'go_front_cube: YOLO取得失敗 '
                f'{max_retry}回すべて失敗'
            )

            self.run_actor('motor', Twist())
            return False

        # --------------------------------
        # 2. YOLOのboxを取得
        # --------------------------------
        box = best_detection.get('box_xyxy', [])

        if len(box) != 4:
            miss_count += 1

            print(
                f'go_front_cube: YOLOのbox_xyxy取得失敗 '
                f'miss_count={miss_count}, '
                f'box={box}'
            )

            return False

        # --------------------------------
        # 3. 検証済み3D Poseを優先し、なければDepthへ戻る
        # --------------------------------
        distance = None
        distance_source = 'depth'

        if use_pose_3d:
            distance = accepted_pose_distance(
                best_detection,
                maximum_reprojection_error_px=(
                    pose_max_reprojection_error_px
                )
            )
            distance_source = 'pose_3d'

        if distance is None:
            depth_distances = []
            for sample_index in range(depth_sample_count):
                sample_distance = self._cube_depth_from_box(
                    box,
                    rgb_width=image_width,
                    rgb_height=image_height,
                    roi_fraction=depth_roi_fraction,
                    minimum_valid_ratio=depth_minimum_valid_ratio,
                    maximum_mad_m=depth_maximum_mad_m,
                )
                if sample_distance is None:
                    distance = None
                    break
                depth_distances.append(sample_distance)
                if sample_index + 1 < depth_sample_count:
                    self.run_actor('sleep', 0.1)
            else:
                try:
                    distance = consistent_depth_median(
                        depth_distances,
                        maximum_spread_m=depth_maximum_spread_m,
                    )
                except CubeDepthError as error:
                    print(f'go_front_cube: Depth不整合: {error}')
                    distance = None
            distance_source = f'depth_median_{len(depth_distances)}'

        if distance is None:
            miss_count += 1

            print(
                f'go_front_cube: Depth取得失敗 '
                f'miss_count={miss_count}'
            )

            return False

        # --------------------------------
        # 4. キューブの横位置を計算
        # --------------------------------
        x_min = float(box[0])
        x_max = float(box[2])

        cube_center_x = (x_min + x_max) / 2.0

        # 画像中央よりtarget_offset_pxだけ右を目標にする
        target_center_x = (
            image_width / 2.0
            + target_offset_px
        )

        # 正数：キューブが目標より右
        # 負数：キューブが目標より左
        error_x = cube_center_x - target_center_x

        # --------------------------------
        # 5. ピクセルのずれを角度に変換
        # --------------------------------
        horizontal_fov_rad = np.deg2rad(
            horizontal_fov
        )

        turn_angle = (
            error_x / image_width
        ) * horizontal_fov_rad

        # キューブが右なら右回転
        if error_x > 0.0:
            turn_direction = -1.0

        # キューブが左なら左回転
        else:
            turn_direction = 1.0

        # 時間 = 必要角度 ÷ 旋回速度
        turn_seconds = (
            abs(turn_angle) / turn_speed
        )

        print(
            f'go_front_cube: '
            f'confidence={best_confidence:.3f}, '
            f'distance={distance:.3f}m, '
            f'distance_source={distance_source}, '
            f'cube_center_x={cube_center_x:.1f}px, '
            f'target_center_x={target_center_x:.1f}px, '
            f'error_x={error_x:.1f}px, '
            f'turn_angle={np.rad2deg(turn_angle):.2f}deg, '
            f'turn_seconds={turn_seconds:.2f}s'
        )

        # --------------------------------
        # 6. 計算した時間だけ一気に旋回
        # --------------------------------
        if turn_seconds > 0.0:
            turn_msg = Twist()

            turn_msg.angular.z = (
                turn_speed * turn_direction
            )

            elapsed = 0.0

            while elapsed < turn_seconds:
                remaining = turn_seconds - elapsed

                sleep_time = min(
                    command_interval,
                    remaining
                )

                self.run_actor('motor', turn_msg)
                self.run_actor('sleep', sleep_time)

                elapsed += sleep_time

            # 旋回終了後に停止
            self.run_actor('motor', Twist())

        # --------------------------------
        # 7. 前進する距離を計算
        # --------------------------------
        forward_distance = (
            distance - stop_distance
        )

        # すでに停止距離以内なら前進しない
        if forward_distance <= 0.0:
            print(
                f'go_front_cube: 現在距離が'
                f'{distance:.3f}mなので前進しません'
            )

            return True

        # 時間 = 距離 ÷ 速度
        forward_seconds = (
            forward_distance / forward_speed
        )

        print(
            f'go_front_cube: '
            f'forward_distance={forward_distance:.3f}m, '
            f'forward_speed={forward_speed:.3f}m/s, '
            f'forward_seconds={forward_seconds:.2f}s'
        )

        # --------------------------------
        # 8. 計算した時間だけ一気に前進
        # --------------------------------
        forward_msg = Twist()
        forward_msg.linear.x = forward_speed

        elapsed = 0.0

        while elapsed < forward_seconds:
            remaining = forward_seconds - elapsed

            sleep_time = min(
                command_interval,
                remaining
            )

            self.run_actor('motor', forward_msg)
            self.run_actor('sleep', sleep_time)

            elapsed += sleep_time

        # 前進完了後に停止
        self.run_actor('motor', Twist())

        print(
            f'go_front_cube: '
            f'約{stop_distance:.2f}m手前までの移動完了'
        )

        return True
    
    @actor
    def take_aim(
        self,
        threshold=0.50,
        turn_speed=0.20,
        center_tolerance=20.0
    ):
        """
        YOLOを1回だけ取得し、キューブが画像中央へ来るように
        旋回秒数を計算して、一度の旋回で中央合わせを行う。

        引数:
            threshold:
                YOLOの最低confidence

            turn_speed:
                旋回速度[rad/s]

            center_tolerance:
                中央とみなす許容範囲[px]

        戻り値:
            True:
                旋回完了、またはすでに中央付近

            False:
                YOLOまたはbox_xyxyの取得に失敗
        """

        threshold = float(threshold)
        turn_speed = abs(float(turn_speed))
        center_tolerance = abs(float(center_tolerance))

        # 外から変更しない固定値
        horizontal_fov = 60.0
        image_width = 848.0
        command_interval = 0.1

        miss_count = 0

        # 旋回速度が0だと0除算になる
        if turn_speed == 0.0:
            print('take_aim: turn_speedが0です')
            return False

        # --------------------------------
        # 1. YOLOを1回だけ取得
        # --------------------------------
        # YOLOを最大5回確認する
        best_detection = None
        best_confidence = 0.0
        max_retry = 5

        for retry_count in range(1, max_retry + 1):
            best_detection, best_confidence = (
                self._read_best_cube_detection()
            )

            print(
                f'take_aim: YOLO確認 '
                f'{retry_count}/{max_retry}, '
                f'confidence={best_confidence:.3f}'
            )

            if (
                best_detection is not None
                and best_confidence >= threshold
            ):
                break

            self.run_actor('sleep', 0.3)

        else:
            print(
                f'take_aim: YOLO取得失敗 '
                f'{max_retry}回すべて失敗'
            )

            self.run_actor('motor', Twist())
            return False

        # --------------------------------
        # 2. バウンディングボックスを取得
        # --------------------------------
        box = best_detection.get('box_xyxy', [])

        if len(box) != 4:
            miss_count += 1

            print(
                f'take_aim: box_xyxy取得失敗 '
                f'miss_count={miss_count}, '
                f'box={box}'
            )

            return False

        # --------------------------------
        # 3. キューブ中心と画像中心を計算
        # --------------------------------
        x_min = float(box[0])
        x_max = float(box[2])

        cube_center_x = (x_min + x_max) / 2.0
        image_center_x = image_width / 2.0

        # 正数：キューブが画像の右側
        # 負数：キューブが画像の左側
        error_x = cube_center_x - image_center_x

        print(
            f'take_aim: '
            f'confidence={best_confidence:.3f}, '
            f'cube_center_x={cube_center_x:.1f}px, '
            f'image_center_x={image_center_x:.1f}px, '
            f'error_x={error_x:.1f}px'
        )

        # --------------------------------
        # 4. すでに中央付近なら旋回しない
        # --------------------------------
        if abs(error_x) <= center_tolerance:
            print('take_aim: すでに中央付近です')
            return True

        # --------------------------------
        # 5. ピクセルのずれを旋回角度に変換
        # --------------------------------
        turn_angle_deg = (
            error_x / image_width
        ) * horizontal_fov

        turn_angle_rad = np.deg2rad(
            turn_angle_deg
        )

        # キューブが右なら右回転
        if error_x > 0.0:
            turn_direction = -1.0

        # キューブが左なら左回転
        else:
            turn_direction = 1.0

        # --------------------------------
        # 6. 旋回秒数を計算
        # --------------------------------
        # 時間 = 角度 ÷ 角速度
        turn_seconds = (
            abs(turn_angle_rad) / turn_speed
        )

        print(
            f'take_aim: '
            f'turn_angle={turn_angle_deg:.2f}deg, '
            f'turn_speed={turn_speed:.3f}rad/s, '
            f'turn_seconds={turn_seconds:.2f}s'
        )

        # --------------------------------
        # 7. 計算した時間だけ一気に旋回
        # --------------------------------
        turn_msg = Twist()

        turn_msg.angular.z = (
            turn_speed * turn_direction
        )

        elapsed = 0.0

        while elapsed < turn_seconds:
            remaining = turn_seconds - elapsed

            sleep_time = min(
                command_interval,
                remaining
            )

            self.run_actor('motor', turn_msg)
            self.run_actor('sleep', sleep_time)

            elapsed += sleep_time

        # --------------------------------
        # 8. 旋回終了
        # --------------------------------
        # 旋回終了
        self.run_actor('motor', Twist())

        print(
            f'take_aim: {turn_seconds:.2f}秒旋回して終了'
        )

        # 車体とカメラ画像が安定するまで待つ
        print('take_aim: 画像安定待ち 1.0秒')
        self.run_actor('sleep', 1.0)

        # 旋回前の古い検出結果を破棄
        self.set_value('cube_detection', None)

        return True
class Tb3CameraSystem(SubSystem):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.set_value('cv_bridge', CvBridge())
        self.register_subscriber('pic',Image,"/camera/camera/color/image_raw",10)
        self.register_subscriber('depth',Image,"/camera/camera/depth/image_rect_raw",10)
        self.register_subscriber('camera_info',CameraInfo,"/camera/camera/color/camera_info",10)
        self.add_network(CognitiveNetwork)
    
    @actor
    def init_realsense(self):
        info_mes = self.run_actor('camera_info')
        it = rs.intrinsics()
        it.width = info_mes.width
        it.height = info_mes.height
        it.fx = info_mes.k[0]
        it.ppx = info_mes.k[2]
        it.fy = info_mes.k[4]
        it.ppy = info_mes.k[5]
        self.set_value('intrinsics', it)

class Tb3ManipulatorSystem(SubSystem):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        self.add_network(ManipulatorNetwork)
        
        node = self.get_value('node')
        cg = self.get_value('callback_group')
        arm = MoveIt2(
            node=node,
            joint_names=joint_names(),
            base_link_name=base_link_name(),
            end_effector_name=end_effector_name(),
            group_name=MOVE_GROUP_ARM,
            callback_group=cg,
            use_move_group_action=True,
        )
        gripper = GripperInterface(
            node=node,
            gripper_joint_names=gripper_joint_names(),
            open_gripper_joint_positions=OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=MOVE_GROUP_GRIPPER,
            callback_group=cg,
#            follow_joint_trajectory_action_name="gripper_trajectory_controller/follow_joint_trajectory",
#            gripper_command_action_name="gripper_action_controller/gripper_cmd",
            follow_joint_trajectory_action_name="arm_controller/follow_joint_trajectory",
            gripper_command_action_name="gripper_controller/gripper_cmd",
            use_move_group_action=True,
        )
        arm.max_velocity = 0.5
        arm.max_acceleration = 0.5
        self.set_value('arm', arm)
        self.set_value('gripper', gripper) 
        self.set_value('joint_stat', [0.0, 0.0, 0.0, 0.0])  
        self.register_subscriber('joints',JointState,"/joint_states",10)
    
class MapSystem(SubSystem):
    def __init__(self, name, parent, map_file=None) -> None:
        super().__init__(name, parent)
        cache_file = os.path.expanduser('~/.actordemo/map_cache')
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                world = pickle.load(f)
        else:
            if map_file:
                world = get_map_ROS(self.map_file)
            else:
                qos_profile=QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1)
                self.register_subscriber(
                    'map_topic', 
                    OccupancyGrid,
                    "/map",
                    qos_profile)
                data = self.run_actor('map_topic')
                width = data.info.width #map width, from nav_msgs/msg/MapMetaData
                height = data.info.height #map height, from nav_msgs/msg/MapMetaData
                map_array = np.array(data.data).reshape(height, width)
                map_array = np.flipud(map_array)

                # make ndarray that has the same size as of map_array
                pgm_array = np.zeros((height, width), dtype=np.uint8)
                pgm_array[(map_array == -1)] = 254
                pgm_array[(map_array < 90)] = 254 
                pgm_array[(map_array >= 90)] = 0 

                resolution = data.info.resolution
                o = data.info.origin.position
                origin = (o.x, o.y)                
                world = get_map(pgm_array, resolution, origin)
                
            cache_dir = os.path.expanduser('~/.actordemo')
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(world, f)
        self.set_value('world', world)
        
