import array
import json
import math
import operator
from math import radians, atan2, degrees, pi
import numpy as np
import quaternion
from simple_pid.pid import PID
from time import sleep

from control_msgs.action import GripperCommand as GripperCommandAction

from ros_actor import actor, SubNet
from pymoveit2 import MoveIt2State

adjust_plus = 1.1  
adjust_minus = 1.25

def _frame_pose(tf_buffer, target_frame, source_frame):
    """Return (position, quaternion) of source expressed in target, or (None, None)."""
    if tf_buffer is None or not source_frame:
        return None, None
    try:
        tf = tf_buffer.lookup_transform(target_frame, source_frame,
                                        rclpy_time_now())
    except Exception:  # noqa: BLE001 - a missing frame is normal here
        return None, None
    t = tf.transform.translation
    q = tf.transform.rotation
    return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)


def rclpy_time_now():
    """Latest available transform time (no clock sync needed in sim)."""
    from rclpy.time import Time
    return Time()


def wait_until_executed(arm):
    while(arm.query_state() != MoveIt2State.IDLE):
        sleep(0.5)

class ManipulatorNetwork(SubNet):
    # Lift the end effector straight up from wherever it currently is
    # (relative joint adjustment, not an absolute target), before handing
    # off to home(). home() plans a joint-space move from the current pose
    # with no idea an object is being held, so on a short/close pick it can
    # dip through a low path and drag the object along the ground; lifting
    # clear first avoids that regardless of where the pick happened.
    @actor
    def pick_up(self):
        joint = [0.0, radians(34), radians(87), 0.0, radians(-78), 0.0]
        self.run_actor('move_joint', *joint)
        return True


    # set to home position
    @actor
    def home(self):
#        joint = [0.0, 1.0, 1.575, 0.0, -1.0, 0.0]
        joint = [0.0, radians(-63), radians(107), 0.0, radians(56), 0.0]
        self.run_actor('move_joint', *joint)
        return True

    @actor
    def pos_home(self):
        angle = (0, 90, 0)
        pos = (0.3, 0.0, 0.15)
        self.run_actor('arm_pose', pos, list(map(radians, angle)))
        return True
        
    @actor
    def arm0(self):
        '''
        angle = (0, 90, 0)
        pos = (0.3, 0.0, 0.15)
        self.run_actor('arm_pose', pos, list(map(radians, angle)))
        '''
        joint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.run_actor('move_joint', *joint)
        return True

    @actor    
    def move_joint(self, *joint_values):
        joint_positions = array.array('d', joint_values)
#        node = self.get_value('node')
#        node.get_logger().info(f"Moving to {{joint_positions: {list(joint_positions)}}}")
        res = self.run_actor("move_to_configuration", joint_positions)
        self.set_value('joint_stat', joint_values)
        return res
    
    # joint angle in degree units
    @actor
    def move_joint_degree(self, *args):
        return self.move_joint(*list(map(radians, args)))

    # designate diff value of each joint
    @actor
    def adjust_joint(self, *args):
        value = self.get_value('joint_stat')
        off = list(map(radians, args))
        return self.move_joint(*list(map(operator.add, value, off)))

    @actor
    def adjust_joint_radian(self, *args):
        value = self.get_value('joint_stat')
        return self.move_joint(*list(map(operator.add, value, *args)))

    @staticmethod
    def _link_position_relative(tf_buffer, link):
        """Position of a robot link in base_link."""
        position, _ = _frame_pose(tf_buffer, 'base_link', link)
        return position

    @staticmethod
    def _tool_axis(tf_buffer):
        """Tool (gripper approach) axis in base_link."""
        ee, _ = _frame_pose(tf_buffer, 'base_link', 'end_effector_link')
        wrist, _ = _frame_pose(tf_buffer, 'base_link', 'link7')
        if ee is None or wrist is None:
            return None
        axis = [ee[i] - wrist[i] for i in range(3)]
        norm = math.sqrt(sum(component * component for component in axis))
        if norm < 1e-9:
            return None
        return tuple(component / norm for component in axis)

    @actor
    def arm_to_cube(
        self,
        standoff_m=0.12,
        roll_deg=180.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        minimum_confidence=0.30,
        require_color_fixed=True,
        grasp_at_fingers=True,
    ):
        """Aim the arm at the detected cube, with an explicit orientation.

        The target is the cube position lifted by ``standoff_m`` along
        base_link Z, aimed with (roll, pitch, yaw) in degrees. The cube pose is
        taken from ``/cube_pose_result`` and only used when the colour semantics
        are resolved, because that is what fixes which physical face is which; an
        unresolved pose would aim at a symmetry-equivalent cube.

        Orientation is only honoured when the IK solver weights rotation, i.e.
        when ``position_only_ik`` is false. ``bin/ccc2`` sets that before MoveIt
        starts; with the upstream default the angles here would be ignored.
        """
        try:
            msg = self.run_actor('cube_pose_result')
            data = json.loads(msg.data)
        except (AttributeError, TypeError, json.JSONDecodeError) as error:
            print(f'arm_to_cube: no cube result: {error}')
            return False

        detection = None
        confidence = 0.0
        for candidate in data.get('detections', []):
            value = float(candidate.get('confidence', 0.0))
            if value > confidence:
                detection, confidence = candidate, value
        if detection is None or confidence < float(minimum_confidence):
            print(f'arm_to_cube: no cube above the confidence floor '
                  f'({confidence:.2f} < {float(minimum_confidence):.2f})')
            return False

        pose = detection.get('pose_3d') or {}
        position = pose.get('position_m')
        if not position or len(position) != 3:
            print('arm_to_cube: detection carries no pose_3d')
            return False
        if require_color_fixed and (
                pose.get('color_fixed_semantics_resolved') is not True):
            print('arm_to_cube: colour semantics unresolved; not aiming')
            return False

        frame_id = pose.get('frame_id') or 'camera_color_optical_frame'
        tf_buffer = self.get_value('tf_buffer')
        tf = None
        for _ in range(25):
            if tf_buffer is not None and tf_buffer.can_transform(
                    'base_link', frame_id, rclpy_time_now()):
                tf = tf_buffer.lookup_transform(
                    'base_link', frame_id, rclpy_time_now())
                break
            sleep(0.2)
        if tf is None:
            print(f'arm_to_cube: no transform {frame_id} -> base_link')
            return False

        rotation = quaternion.as_rotation_matrix(np.quaternion(
            tf.transform.rotation.w, tf.transform.rotation.x,
            tf.transform.rotation.y, tf.transform.rotation.z))
        point_camera = np.array([float(value) for value in position])
        translation = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])
        point_base = rotation @ point_camera + translation

        # Aim the *fingertips* at the cube, not the wrist. pymoveit2 moves
        # end_effector_link, which on this URDF sits 11.5 cm past link7 while the
        # finger pads sit at 7.1 cm, so a naive target puts the palm on the cube
        # (measured: wrong by 4.4 cm along the tool axis). The offset is fixed by
        # the mechanism, so read it from TF once per call instead of hard-coding.
        offset_tool = 0.0
        if grasp_at_fingers:
            ee = self._link_position_relative(tf_buffer, 'end_effector_link')
            left = self._link_position_relative(tf_buffer, 'gripper_left_link')
            right = self._link_position_relative(tf_buffer, 'gripper_right_link')
            if ee is not None and left is not None and right is not None:
                grasp_point = ((left[0] + right[0]) / 2.0,
                               (left[1] + right[1]) / 2.0,
                               (left[2] + right[2]) / 2.0)
                # vector from the finger midpoint to end_effector_link
                offset = (ee[0] - grasp_point[0], ee[1] - grasp_point[1],
                          ee[2] - grasp_point[2])
                tool_axis = self._tool_axis(tf_buffer)
                if tool_axis is not None:
                    along = sum(offset[i] * tool_axis[i] for i in range(3))
                    offset_tool = float(along)
                    print(f'arm_to_cube: finger->ee offset {offset_tool:.4f} m '
                          f'along the tool axis')
                target = (
                    float(point_base[0]) + offset[0],
                    float(point_base[1]) + offset[1],
                    float(point_base[2]) + offset[2],
                )
            else:
                print('arm_to_cube: finger links unavailable; using the ee pose')
                target = (float(point_base[0]), float(point_base[1]),
                          float(point_base[2]))
        else:
            target = (float(point_base[0]), float(point_base[1]),
                      float(point_base[2]))
        target = (target[0], target[1], target[2] + float(standoff_m))
        angle = [
            radians(float(roll_deg)),
            radians(float(pitch_deg)),
            radians(float(yaw_deg)),
        ]
        print(f'arm_to_cube: cube in base_link='
              f'({point_base[0]:.3f}, {point_base[1]:.3f}, {point_base[2]:.3f})'
              f' fingertip_target=({target[0]:.3f}, {target[1]:.3f},'
              f' {target[2]:.3f}) ee_offset={offset_tool:.4f} m'
              f' rpy_deg=({float(roll_deg):.0f}, {float(pitch_deg):.0f},'
              f' {float(yaw_deg):.0f})')
        self.run_actor('arm_pose', target, angle)
        return True

    def move_position(self, position, orientation):
        position = array.array('d', position)
        orientation = array.array('d', orientation)
        self.node.get_logger().info(
            f"Moving to {{position: {list(position)}, quat_xyzw: {list(orientation)}}}"
        )
        self.run_actor('move_to_pose', position=position, quat_xyzw=orientation)
    
    # move arm and wait until done
    @actor
    def move_to_configuration(self, joint_positions):
        arm = self.get_value('arm')
        arm.move_to_configuration(joint_positions, joint_names=arm.joint_names)
        wait_until_executed(arm)
        return True         
#        return arm.wait_until_executed() # Never use this
    
    # # open gripper
    # @actor
    # def open(self):
    #     self.run_actor('open_gripper')
    #     try: self.run_actor("detach")
    #     except TypeError: pass
        
    #     self.run_actor('sleep', 2)
    #     return True

    # # close gripper
    # @actor
    # def close(self):
    #     self.run_actor('close_gripper')
    #     self.run_actor("attach")
    #     return True
    
    # open gripper
    @actor
    def open(self):
        self.run_actor('open_gripper')
        self.run_actor('sleep', 2)
        return True

    # close gripper
    @actor
    def close(self):
        self.run_actor('full_close')



    @actor
    def full_close(self):
        gripper = self.get_value('gripper')
        gripper.move_to_position(-0.01)
        self.run_actor('sleep', 2)
        return True

    @actor
    def open_gripper(self):
        gripper = self.get_value('gripper')
        gripper.open()
        wait_until_executed(gripper)
        return True

    @actor
    def close_gripper(self):
        gripper = self.get_value('gripper')
        gripper.close()
        wait_until_executed(gripper)
        return True

    @actor
    def get_status(self):
        return {
            'joint': self.get_value('jstat')
        }
    
    # adjust arm angle
    @actor
    def ad(self):
        x, y, angle = self.run_actor('object_loc')
        if angle > 0:
            angle *= adjust_plus
        else:
            angle *= adjust_minus
        jstat = self.run_actor('jstat')
        cur = []
        for k in ('joint2','joint3','joint4','joint5','joint6'):
            cur.append(jstat[k])
        cur.insert(0,angle)
#        print(f'adjust.angle:{degrees(angle)}')
        self.run_actor('move_joint', *cur)
        self.run_actor('sleep', 1)
        return True
    
    @actor
    def arm_turn(self, value=0):
        jstat = self.run_actor('jstat')
        cur = []
        for k in ('joint2','joint3','joint4','joint5','joint6'):
            cur.append(jstat[k])
        cur.insert(0,value)
#        print(f'arm_turn.angle:{degrees(angle)}')
        self.run_actor('move_joint', *cur)
        self.run_actor('sleep', 1)
        return True

    @actor
    def ad0(self):
        jstat = self.run_actor('jstat')
        cur = []
        for k in ('joint2','joint3','joint4','joint5','joint6'):
            cur.append(jstat[k])
        cur.insert(0,0.0)
        self.run_actor('move_joint', *cur)
        self.run_actor('sleep', 1)
        return True

    # adjust arm base direction to the center of coke can
    # when arm down with variable distance
    @actor
    def fit(self, assumed_distance=0.15):
        _, _, angle, distance = self.run_actor('measure_center', assumed=-1)
        if distance < 0.1:
            print('could not gain depth')
            self.run_actor('fit2')
            return
        print(f'arm angle: {degrees(angle)}, distance: {distance}')
        jstat = self.run_actor('jstat')
        cur = []
        for k in ('joint2','joint3','joint4','joint5','joint6'):
            cur.append(jstat[k])
#        cur.insert(0,angle-radians(0.4))
        cur.insert(0,angle)
        print(f'fit: {cur}')
        self.run_actor('move_joint', *cur)
        self.run_actor('sleep', 1)
        return True

    # adjust arm base direction
    # when arm down and fixed distance (ready for grasp)
    @actor
    def fit2(self, distance):
        jstat = self.run_actor('jstat')
        cur = []
        for k in ('joint1', 'joint2','joint3','joint4','joint5','joint6'):
            cur.append(jstat[k])
        _,_,target_angle,_ = self.run_actor('measure_center2', assumed=distance)
        cur[0] = target_angle
        self.run_actor('move_joint', *cur)
        self.run_actor('sleep', 3)
        return True
    
    @actor
    def tf(self):
        pic_diff, edge = self.run_actor('find_object_pic')
        off = pic_diff - edge - 0.491
        print(f'off:{off}')
        angle = -atan2(off, 1)
        print(f'angle:{degrees(angle)}')
        return True

    # set arm to pick position
    @actor
    def pick(self, dir=0.0):
        '''
        angle = (0, 90, 0)
        pos = (0.3, 0.0, 0.05)
        self.run_actor('arm_pose', pos, list(map(radians, angle)))
        self.run_actor('sleep', 3.0)
        return True
        '''
        # joint = [0.0, 1.75, 1.4, 0.0, -1.6, 0.0]
        joint = [0.0, radians(97), radians(83), 0.0, radians(-83), 0.0]
        joint[0] = dir
        self.run_actor('move_joint', *joint)
        return True
    
    # set arm to place position
    @actor
    def place(self):
        val = (0.0, 50.0, -33.0, -16.0)
        self.run_actor('move_joint', *map(radians, val))
        return self.run_actor('open')
    
    # set arm to place position without opening gripper
    @actor
    def place_pose(self):
        # 1. 先にグリッパーを開く
        self.run_actor('open')
        val = (0.0, 90.0, 90.0, 0.0,-77.0,0.0)

        self.run_actor(
            'move_joint',
            *map(radians, val)
        )

        return True

    @actor    
    def pid(self, *joint_values):
        arm = self.get_value('arm')        
        print(f'planner_id:{arm.planner_id}')
        return True

    @actor    
    def pose0(self, *joint_values):
        angle = (0, 90, 0)
        pos = (0.098, 0.0, 0.2497)
        self.run_actor('arm_pose', pos, list(map(radians, angle)))
        return True
    
    @actor    
    def pose1(self, *joint_values):
        angle = (0, 90, 0)
        pos = (0.3, 0.0, 0.05)
        self.run_actor('arm_pose', pos, list(map(radians, angle)))
        return True
    
    @actor
    def arm_pose(self, pos, angle):       
        q = quaternion.from_euler_angles(angle)
        quat_xyzw = (q.x, q.y, q.z, q.w)
        arm = self.get_value('arm')
        arm.move_to_pose(position=pos, quat_xyzw=quat_xyzw, cartesian=False)
        wait_until_executed(arm)
        return True

    @actor
    def jstat(self):
        arm = self.get_value('arm')
        state = arm.joint_state
        state_dict = dict(zip(state.name, state.position))
        return state_dict
        
    @actor
    def arm_angle(self, j1=0, j2=0, j3=0, j4=0, j5=0, j6=0):
        joint = [radians(j1), radians(j2), radians(j3), radians(j4), radians(j5), radians(j6)]
        self.run_actor('move_joint', *joint)
        return True
    @actor
    def hirou_pose(self):
        joint_degree = (
            0.0,
            90.0,
            95.0,
            0.0,
            -68.0,
            6.0
        )

        return self.run_actor(
            'move_joint_degree',
            *joint_degree
        )
# register_pose {register_name} j1 j2 j3 j4 j5 j6
# で動かせるactor登録actorがあったら良いのになあ
# 尚、登録機はactorでなくてもよい。。。けど、まあactorが良いか。
# 名前被りは"名前被りだ！"って言えるように、全部の関数名を正規表現とかでリストにしたい。つまるところアームだけで良い。
# behaviorにはならんが。
# そういえば、勝手にmigrationしてくれるやつを作りたい。