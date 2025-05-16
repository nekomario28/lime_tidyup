from math import radians, degrees, atan2, sqrt
from simple_pid.pid import PID

from ros_actor import actor, SubNet
from ..pointlib import PointEx
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion 

SPEED = 0.03
TURN = 0.3

class ApproachAction(SubNet):
    # direct drive on nav motor
    def move(self, dir=1, turn=0):
        twist = Twist()
        twist.linear.x = SPEED * dir
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = TURN * turn
        self.run_actor('motor', twist)
    
    @actor
    def stop(self):
        self.move(0)
    
    # direct command to motor
    @actor    
    def mini_walk(self, len=5):
        if len == 0:
            self.move(0)
            return
        if len > 0:
            self.move(1)
        else:
            self.move(-1)
            len = -len
        self.run_actor('sleep', len*0.1)
        self.move(0, 0)
    
    # navigation with visual feedback
    @actor
    def targetted_walk(self, len, fname="control.csv", speed=1.0, off=0.0):
        count = 0
        f = open(fname, mode="w")
        start_x, start_y, _ = self.get_odom()
        assumed = -1
        current = 0
        if speed > 0.5:
            pid = PID(-10, 0, -0.25,
                    output_limits=(radians(-80),radians(80)))
        else:
            pid = PID(-10, 0, -0.05,
                    output_limits=(radians(-80),radians(80)))
            print('PID,-10,0,-0.05', file=f)

        while True:
            log = {}
            _,_,target_angle,distance = self.run_actor('measure_center', target='base_link', assumed=assumed,log=log)
            if distance < 0:
                print('could not get distance value. give up')
                return
            
            x, y, _ = self.get_odom()
            
            diff_angle = target_angle
            control = pid(diff_angle)
            adj_flag = False
            if speed < 0.5:
                if control > 0: control *= 1.05
                if abs(diff_angle) > radians(0.5):
#                    print(f'stop and adjust body angle:{degrees(diff_angle)},{control}')
                    self.run_actor('sleep', 0.03)
                    adj_flag = True
            self.move(speed, control)

            d_x = x - start_x
            d_y = y - start_y
            last_current = current
            current = sqrt(d_x**2 + d_y**2)
            if distance > 0:
                assumed = distance
            else:
                assumed = assumed - (current - last_current)
            print(f'{degrees(diff_angle)},{-degrees(control)},{log["index"]},{distance},{assumed - (current - last_current)},{log["_y"]},{log["y"]},{str(log["assumed"])},{str(adj_flag)}', file=f)
            f.flush()
            if assumed < len:
#                self.move(-1)
#                self.run_actor('sleep', 0.05)
                self.move(0)
                f.close()
                return target_angle
            count += 1
    
    # turn by driving motor directly
    # the move result is not reflected to navigation target
    @actor    
    def mini_turn(self, len=5):
        if len == 0:
            self.move(0, 0)
            return
        if len > 0:
            self.move(0, 1)
            sleep_time = len*0.1 + 0.5
        else:
            self.move(0, -1)
            len = -len
            sleep_time = len*0.1 + 0.5
        self.run_actor('sleep', sleep_time)
        self.move(0)
    
    # debugging tool
    @actor
    def dc(self):
        x, y, _, distance = self.run_actor('measure_center', 'map')
        print(f'x:{x}, y:{y}')
        
    # adjust location
    @actor
    def reach_coke(self, target=0.26):
        trans = self.run_actor('map_trans')
        start = PointEx(0.0, 0.0)
        start.setTransform(trans.transform)
        target_angle = self.run_actor('targetted_walk', target, "pick_control.csv")
        print(f'target_angle:{degrees(target_angle)}')
        self.run_actor('sleep', 3)
        trans = self.run_actor('map_trans')
        actual = PointEx(0.0, 0.0)
        actual.setTransform(trans.transform)
        dx = actual.x - start.x
        dy = actual.y - start.y
        dir = atan2(dy, dx)
        self.run_actor('goto', actual.x, actual.y, dir)
#        joint = [0.0, 1.75, 1.4, 0.0, -1.6, 0.0]
#        self.run_actor('move_joint', *joint)
        return True        
    
    # adjust location by using targetted_walk actor
    # it is assumed that gripper position is lowered for picking
    # and also depth info is sometimes lost
    #   because distance to the subject is too short. 
    @actor
    def shift(self, target):
        trans = self.run_actor('map_trans')
        start = PointEx(0.0, 0.0)
        start.setTransform(trans.transform)
        target_angle = self.run_actor('targetted_walk', target, "shift_control.csv", 0.25)
        self.run_actor('sleep', 3)
        trans = self.run_actor('map_trans')
        actual = PointEx(0.0, 0.0)
        actual.setTransform(trans.transform)
        dx = actual.x - start.x
        dy = actual.y - start.y
        dir = atan2(dy, dx)
        self.run_actor('goto', actual.x, actual.y, dir)
#        self.run_actor('arm_turn', target_angle / 2)
        return True        

    # approach subject with visual feedback by using targetted_walk
    @actor
    def approach(self, target=0.20):
        trans = self.run_actor('map_trans')
        start = PointEx(0.0, 0.0)
        start.setTransform(trans.transform)
        self.move(1)
        with self.run_actor_mode('measure_distance', 'iterator', 'bar') as obj_it:
            for distance in obj_it:
                if distance <= target: break 
        self.move(0)
        self.run_actor('sleep', 1)
        trans = self.run_actor('map_trans')
        actual = PointEx(0.0, 0.0)
        actual.setTransform(trans.transform)
        dx = actual.x - start.x
        dy = actual.y - start.y
        dir = atan2(dy, dx)
        self.run_actor('goto', actual.x, actual.y, dir)
    
    # adjust body angle to face the subject
    @actor
    def face(self, raw=0):
        root = PointEx()
        trans = self.run_actor('map_trans', 'base_link') # base->map
        if not trans:
            print('trans error')
            return False
        root.setTransform(trans.transform)
        x, y, _ = self.run_actor('object_loc', 'map')
        dx = x - root.x # map coordinate based
        dy = y - root.y
        abs_angle = atan2(dy, dx) # angle against map x-axis
#        print(f'abs_angle:{degrees(abs_angle)}')
        self.adjust_body_angle()
#        print(f'x]{root.x},y:{root.y},angle:{degrees(abs_angle)}')
        self.run_actor('goto', root.x, root.y, abs_angle)
        return True

    def get_odom(self):
        odom = self.run_actor('odom')
        ori = odom.pose.pose.orientation
        q = (ori.x, ori.y, ori.z, ori.w)
        _, _, angle = euler_from_quaternion(q)
        pos = odom.pose.pose.position
        return pos.x, pos.y, angle
    
    # turn action with feedback by pixel number
    def adjust_body_angle(self):
        for _ in range(30):
            pl, _ = self.run_actor('find_object_pic')
#            print(f'face.pl:{pl}')
#            if abs(pl-1.0) < 0.05:
#                self.move(0)
#                return
            if pl > 1.3: # turn right
                self.move(0, -0.5)
            elif pl > 1.0:
                self.move(0, -0.25)
            elif pl < 0.7:
                self.move(0, 0.5)
            else:
                self.move(0, 0.25)
#            self.run_actor("sleep", 0.5)
        self.move(0)
        return

    # Check to see if there's a Coke can in sight
    @actor
    def check_coke(self):
        for _ in range(8):
            self.run_actor('sleep', 3)
            coke = self.run_actor('coke_getter')
            if coke[0] >= 0: return True
            print('turn to recover')
            self.run_actor('mini_turn', 45)
        return False
    