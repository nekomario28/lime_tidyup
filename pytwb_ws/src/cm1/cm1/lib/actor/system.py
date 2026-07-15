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
from .manipulator import ManipulatorNetwork
from .tools import Tools

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
            threshold=0.80,
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
                rotate_msg.angular.z = 0.40

                # 0.6秒間、繰り返しcmd_velを送る
                for _ in range(50):
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
                rotate_msg.angular.z = 0.40
                print('キューブが左端なので左へ大きく回転')

            # 右端に見えている場合は右へ大きく回転
            elif error_x > center_tolerance:
                rotate_msg.angular.z = -0.40
                print('キューブが右端なので右へ大きく回転')

            # キューブが見つからない、または信頼度不足
            else:
                rotate_msg.angular.z = 0.30
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
        threshold=0.80,
        center_tolerance=25.0,
        stop_box_width=300.0,
        horizontal_fov=60.0
    ):
        print(
            'go_front_cube actor開始:',
            threshold,
            center_tolerance,
            stop_box_width,
            horizontal_fov
        )

        threshold = float(threshold)
        center_tolerance = float(center_tolerance)
        stop_box_width = float(stop_box_width)
        horizontal_fov = float(horizontal_fov)

        

        image_width = 848.0

        # カメラ中央。
        # グリッパーの中心とずれる場合は後で調整する
        target_center_x = image_width / 2.0

        # 連続で停止条件を満たした回数
        stop_count = 0

        # 検出を連続で失敗した回数
        miss_count = 0

        # 一度でも前進したか
        has_started_forward = False

        # 最後に確認できた値
        last_box_width = 0.0
        last_error_x = 0.0

        try:
            while True:

                # ==============================================
                # 1. YOLOの結果を取得
                # ==============================================
                msg = self.run_actor('cube_pose_result')

                try:
                    data = json.loads(msg.data)

                except (
                    json.JSONDecodeError,
                    AttributeError,
                    TypeError
                ) as error:
                    print(f'JSON解析失敗: {error}')

                    self.run_actor('motor', Twist())
                    self.run_actor('sleep', 0.10)
                    continue

                detections = data.get('detections', [])

                best_detection = None
                best_confidence = 0.0

                for detection in detections:
                    confidence = float(
                        detection.get('confidence', 0.0)
                    )

                    if (
                        confidence >= threshold
                        and confidence > best_confidence
                    ):
                        best_confidence = confidence
                        best_detection = detection

                # ==============================================
                # 2. キューブを見失った場合
                # ==============================================
                if best_detection is None:
                    miss_count += 1
                    stop_count = 0

                    # まず停止
                    self.run_actor('motor', Twist())

                    print(
                        f'キューブ未検出: '
                        f'{miss_count}回連続'
                    )

                    # 一時的な検出抜けなら待つ
                    if miss_count < 5:
                        self.run_actor('sleep', 0.10)
                        continue

                    # 十分近い位置で見失った場合は、
                    # カメラに入りきらないほど近づいた可能性がある
                    if (
                        has_started_forward
                        and last_box_width
                        >= stop_box_width * 0.85
                    ):
                        print(
                            '停止距離付近で見失ったため、'
                            '接近完了と判断します'
                        )

                        return True

                    # まだ遠い場合は、最後に見えた方向へ小さく回す
                    search_msg = Twist()
                    search_msg.linear.x = 0.0

                    if last_error_x > 0:
                        # キューブが左側にあった
                        search_msg.angular.z = 0.12

                    elif last_error_x < 0:
                        # キューブが右側にあった
                        search_msg.angular.z = -0.12

                    else:
                        search_msg.angular.z = 0.12

                    print('最後に見えた方向へ小さく再探索します')

                    self.run_actor('motor', search_msg)
                    self.run_actor('sleep', 0.12)
                    self.run_actor('motor', Twist())
                    self.run_actor('sleep', 0.10)

                    continue

                # 検出できたのでリセット
                miss_count = 0

                # ==============================================
                # 3. バウンディングボックスを取得
                # ==============================================
                box = best_detection.get('box_xyxy', [])

                if len(box) != 4:
                    print(f'box_xyxyが不正です: {box}')

                    self.run_actor('motor', Twist())
                    self.run_actor('sleep', 0.10)
                    continue

                x_min = float(box[0])
                x_max = float(box[2])

                cube_center_x = (
                    x_min + x_max
                ) / 2.0

                box_width = (
                    x_max - x_min
                )

                # 正ならキューブは目標位置より左
                # 負ならキューブは目標位置より右
                error_x = (
                    target_center_x
                    - cube_center_x
                )

                last_box_width = box_width
                last_error_x = error_x

                print(
                    f'confidence={best_confidence:.3f}, '
                    f'center_x={cube_center_x:.1f}, '
                    f'error_x={error_x:.1f}, '
                    f'box_width={box_width:.1f}'
                )
                # ==============================================
                # 強制終了判定
                # 十分近づいたら、中央誤差に関係なく次のActorへ進む
                # ==============================================
                

                                # 十分近づいたら強制的に接近終了
                if box_width >= stop_box_width:
                    print(
                        f'十分近づきました: '
                        f'box_width={box_width:.1f}px, '
                        f'停止基準={stop_box_width:.1f}px, '
                        f'error_x={error_x:.1f}px'
                    )

                    self.run_actor('motor', Twist())
                    self.run_actor('sleep', 0.10)

                    # 少し左へ回転
                    left_turn_msg = Twist()
                    left_turn_msg.linear.x = 0.0
                    left_turn_msg.angular.z = 0.12

                    print('少し左へ向きを調整します')

                    for _ in range(6):
                        self.run_actor('motor', left_turn_msg)
                        self.run_actor('sleep', 0.10)

                    self.run_actor('motor', Twist())
                    self.run_actor('sleep', 0.10)

                    self.set_value(
                        'cube_detection',
                        best_detection
                    )

                    print('接近完了。次のActorへ進みます')
                    return True


                # ==============================================
                # 4. 距離に応じて中央許容範囲を変える
                # ==============================================
                if box_width < 150.0:
                    current_tolerance = min(
                        center_tolerance,
                        45.0
                    )

                elif box_width < 230.0:
                    current_tolerance = min(
                        center_tolerance,
                        30.0
                    )

                else:
                    # 近いほど正確に中央へ合わせる
                    current_tolerance = min(
                        center_tolerance,
                        15.0
                    )

                # ==============================================
                # 5. 十分近く、かつ中央なら停止
                # ==============================================
                if (
                    box_width >= stop_box_width
                    and abs(error_x) <= current_tolerance
                ):
                    stop_count += 1

                    self.run_actor('motor', Twist())

                    print(
                        f'停止判定: {stop_count}/3 '
                        f'box_width={box_width:.1f}, '
                        f'error_x={error_x:.1f}'
                    )

                    # 誤検出を避けるため3回確認
                    if stop_count >= 3:
                        self.set_value(
                            'cube_detection',
                            best_detection
                        )

                        print(
                            'キューブの正面で停止しました'
                        )

                        return True

                    self.run_actor('sleep', 0.10)
                    continue

                stop_count = 0

                # ==============================================
                # 6. 中央から大きく外れていたら回転だけ
                # ==============================================
                if abs(error_x) > current_tolerance:

                    # ピクセル誤差を角速度へ変換
                    angular_speed = error_x * 0.0020

                    # 回転が強すぎないよう制限
                    angular_speed = max(
                        -0.22,
                        min(0.22, angular_speed)
                    )

                    # 小さすぎて動かないのを防止
                    if 0.0 < angular_speed < 0.07:
                        angular_speed = 0.07

                    elif -0.07 < angular_speed < 0.0:
                        angular_speed = -0.07

                    turn_msg = Twist()
                    turn_msg.linear.x = 0.0
                    turn_msg.angular.z = angular_speed

                    print(
                        f'中央調整のみ: '
                        f'許容={current_tolerance:.1f}px, '
                        f'角速度={angular_speed:.3f}rad/s'
                    )

                    # 短時間だけ回して、再び画像を確認
                    self.run_actor('motor', turn_msg)
                    self.run_actor('sleep', 0.12)
                    self.run_actor('motor', Twist())
                    self.run_actor('sleep', 0.08)

                    continue

                # ==============================================
                # 7. 中央付近なら、方向修正しながら前進
                # ==============================================
                has_started_forward = True

                # 近づくほど低速にする
                if box_width < 120.0:
                    forward_speed = 0.070

                elif box_width < 180.0:
                    forward_speed = 0.055

                elif box_width < 240.0:
                    forward_speed = 0.10

                else:
                    forward_speed = 0.020

                # 前進中にも小さく方向修正する
                angular_speed = error_x * 0.0012

                angular_speed = max(
                    -0.08,
                    min(0.08, angular_speed)
                )

                move_msg = Twist()
                move_msg.linear.x = forward_speed
                move_msg.angular.z = angular_speed

                print(
                    f'追従前進: '
                    f'速度={forward_speed:.3f}m/s, '
                    f'角速度={angular_speed:.3f}rad/s, '
                    f'許容={current_tolerance:.1f}px'
                )

                # 一度に長く進まず、0.12秒ごとに再検出する
                self.run_actor('motor', move_msg)
                self.run_actor('sleep', 1.0)

                self.run_actor('motor', Twist())
                self.run_actor('sleep', 0.02)

        finally:
            # 例外や中断時にも必ず停止
            self.run_actor('motor', Twist())

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
        
