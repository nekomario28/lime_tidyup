from math import atan2, degrees, radians, hypot
import os

from pyquaternion import Quaternion

from pytwb import lib_main
from ros_actor import SubNet, actor, register_bt
from ..pointlib import PointEx
import cv2
import numpy as np
import pyrealsense2 as rs
import subprocess
import re

class Tools(SubNet):
    # command version
    @actor
    def go(self, *arg):
        arg = list(map(float, arg))
        while len(arg) < 3:
            arg.append(0.0)
        return self.run_actor('goto', arg[0], arg[1], arg[2])

    # update behavior tree name table
    @actor
    def update_bt(self):
        package = lib_main.get_package()
        dir = os.path.join(package.path, 'trees')
        for d in os.listdir(dir):
            if not d.endswith('.xml'): continue
            name = d[:-4]
            register_bt(name)
        return True
    
    # show gripper pose
    @actor
    def gl(self):
        trans = self.run_actor("gripper_trans")
        trans = trans.transform
        offset = trans.translation
        rot = trans.rotation
        point = (0.0, 0.0, 0.0)
        q_rot = Quaternion(rot.w, rot.x, rot.y, rot.z)
        rot_point = q_rot.rotate(point)
        x = rot_point[0] + offset.x
        y = rot_point[1] + offset.y
        print(f'gripper angle:{degrees(atan2(y, x))}')
    
    @actor
    def forward(self, value):
        self.run_actor('adjust_joint', 0.0, value, 0.0, 0.0)
    
    @actor
    def ol(self):
        _,_,target_angle,distance = self.run_actor('measure_center', target='base_link', assumed=0.2)
        print(f'target_angle:{degrees(target_angle)}, distance:{distance}')
    
    @actor
    def tl(self):
        _, _, angle = self.run_actor('object_loc', 'base_link')
        point = self.run_actor('find_object')
        print(f'object angle:{degrees(angle)}, distance:{point.distance}') 
    
    @actor    
    def js(self):
        value = self.get_value('joint_stat')
        d_value = tuple(map(degrees, value))
        print(d_value)
    
    @actor
    def cpos(self):
        root = PointEx()
        ref = PointEx(1.0, 0.0)
        trans = self.run_actor('map_trans')
        if not trans: return
        root.setTransform(trans.transform)
        ref.setTransform(trans.transform)
        print(f'x:{root.x}, y:{root.y}, z:{degrees(root.z)}')
        rx = ref.x - root.x
        ry = ref.y - root.y
        robot_angle = atan2(ry, rx)
        print(f'robot angle to X axis: {degrees(robot_angle)}')
    
    @actor
    def pause(self, is_on=True):
        if is_on: input('debug pause')
        return True
    
    @actor
    def key(self):
        return input()

    @actor
    def angle(self):
        print(f'assumed:{degrees(atan2(0.5, 1.0))}')
        x, y, angle = self.run_actor('object_loc')
        print(f'angle:{degrees(angle)}')

    @actor
    def gripper_angle(self, angle):
        gripper = self.get_value('gripper')
        gripper.move_to_position(angle)
        self.run_actor('sleep', 2)
        return True
    
    @actor
    def get_gripper(self):
        joints = self.run_actor('joints')
        target_joint = 'gripper_left_joint'
        idx = joints.name.index(target_joint)
        pos = round(joints.position[idx], 2)
        return pos
    
    @actor
    def get_arm_angle(self):
        joints = self.run_actor('joints')
        target_joint = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        angle_list = []
        for i in target_joint:
            idx = joints.name.index(i)
            pos = round(degrees(joints.position[idx]), 1)
            angle_list.append(pos)
        return angle_list

    @actor 
    def shot(self, fpath):
        cv_image = self.run_actor('pic_receiver')
        pt = fpath + ".png"
        cv2.imwrite(pt, cv_image)
    
    @actor
    def depth_shot(self, fpath):
        data = self.run_actor('depth')
        cv_bridge = self.get_value('cv_bridge')
        depth_image = cv_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        # 適切な画像になるように正規化している
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = np.uint8(normalized_depth)

        pt = fpath + ".png"
        cv2.imwrite(pt, normalized_depth)

    def pix_to_coordinate(self, x, y, distance):
        intrinsics = self.get_value('intrinsics')
        p = rs.rs2_deproject_pixel_to_point(intrinsics,[x,y], distance)
        return p[2],-p[0]

    @actor
    def go_front(self):
        x, y, theta = self.run_actor("object_loc", "map")
        self.run_actor("goto", x, y, theta)

    @actor
    def attach(self):
        obj = self.run_actor("choose_pick_obj")
        if obj is None:
            print("No object.")
            return False
        
        link = self.run_actor("get_linkname", obj)

        cmd = f"""ros2 service call /ATTACHLINK linkattacher_msgs/srv/AttachLink "{{model1_name: 'turtlebot3_lime_system', link1_name: 'link7', model2_name: '{obj}', link2_name: '{link}'}}" """
        subprocess.run(cmd, shell=True)
        return True
    
    @actor
    def detach(self):
        obj = self.run_actor("choose_pick_obj")
        if obj is None:
            print("No object. Aborted")
            return False
        link = self.run_actor("get_linkname", obj)

        cmd = f"""ros2 service call /DETACHLINK linkattacher_msgs/srv/DetachLink "{{model1_name: 'turtlebot3_lime_system', link1_name: 'link7', model2_name: '{obj}', link2_name: '{link}'}}" """
        subprocess.run(cmd, shell=True)
        return True

    @actor
    def get_link7(self):
        link_info = self.run_actor("link_states")
        names = link_info.name
        index = names.index("turtlebot3_lime_system::link7")

        cod = link_info.pose[index].position
        return (cod.x, cod.y)
    
    @actor
    def get_end_effector(self):
        link_info = self.run_actor("link_states")
        names = link_info.name

        right = names.index("turtlebot3_lime_system::gripper_right_link")
        left = names.index("turtlebot3_lime_system::gripper_left_link")

        r_cod = link_info.pose[right].position
        l_cod = link_info.pose[left].position
        return r_cod.x, r_cod.y, l_cod.x, l_cod.y

    @actor
    def get_model_list(self):
        model_info = self.run_actor("model_states")
        return model_info.name

    @actor
    def get_linkname(self, model_name):
        link_info = self.run_actor("link_states")
        names = link_info.name
        target = next((s for s in names if model_name in s), None)
        match = re.search(rf"^{model_name}::(.+)$", target)
        if match:
            link_name = match.group(1)
        return link_name
        
    @actor 
    def get_modelpos_dict(self, debug:bool = False):
        pos_dict = {}
        model_info = self.run_actor("model_states")
        for i in range(len(model_info.name)):
            if i < 3:
                if debug:
                    print(model_info.name[i])
            else:
                cod = model_info.pose[i].position
                pos_dict[model_info.name[i]] = (cod.x, cod.y)
        return pos_dict
    
    @actor 
    def get_linkpos_dict(self):
        pos_dict = {}
        link_info = self.run_actor("link_states")
        for i in range(len(link_info.name)):
            cod = link_info.pose[i].position
            pos_dict[link_info.name[i]] = (cod.x, cod.y)
        return pos_dict

    @actor
    # def get_near_obj(self, debug:bool =False):
    def choose_pick_obj(self, debug:bool =False):
        gripper = self.run_actor("get_link7")
        model_dict = self.run_actor("get_modelpos_dict")
        name_list = []

        for objname in model_dict:
            dist = hypot(gripper[0]-model_dict[objname][0], gripper[1]-model_dict[objname][1])
            if debug:
                print(f"{objname}: {dist}")

            if dist <= 0.12:
                name_list.append(objname)
                return objname
        return None
        # if len(name_list) == 0:
        #     return False
        # return name_list


    # @actor 
    # def choose_pick_obj(self):
    #     name_list = self.run_actor("get_near_obj")
    #     r_x, r_y, l_x, l_y = self.run_actor("get_end_effector")
    #     model_dict = self.run_actor("get_modelpos_dict")

    #     if not name_list:
    #         return False
        
    #     for n in name_list:
    #         print(model_dict[n])

    #     objname = None
    #     return objname