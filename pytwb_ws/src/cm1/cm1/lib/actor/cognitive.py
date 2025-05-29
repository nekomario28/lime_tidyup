import numpy as np
from math import radians, atan2, degrees, sqrt, isinf

from ros_actor import actor, SubNet
from lib.pointlib import PointEx, PointBag

import cv2
import pyrealsense2 as rs

class CognitiveNetwork(SubNet):
    @actor
    def carib(self):
        x,y,_ = self.run_actor('object_loc', target='base_link')
        print(f'x:{x}, y:{y}')
        
    # Get the coordinates of the object in the arm coordinate system
    @actor  
    def object_loc(self, target='link1'):
        while True:
            point = self.run_actor('find_object')
            if point: break
            self.run_actor('sleep', 1)
#        print(f'object_loc x:{point._x}, y:{point._y}')
        trans = self.run_actor('var_trans', target)
        point.setTransform(trans.transform)
#        angle = atan2(point.y, point.x) + radians(1.05)
        angle = atan2(point.y, point.x) * 1.34
        return point.x, point.y, angle

    # get target location by map coordinate
    @actor
    def object_glance(self):
        while True:
            trans = self.run_actor('map_trans')
            point = self.run_actor('find_object')
            if point: break
#        trans = self.run_actor('map_trans') # avoid delays in receiving odom
        point.setTransform(trans.transform)
        return point.x, point.y
    
    def register_flist(self, cand_points, point):
        cand = None
        min_d = 0.01
        for found in cand_points:
            d = (found.x-point.x)**2 + (found.y-point.y)**2 + (found.z-point.z)**2
            if d >= min_d: continue
            cand = found
            min_d = d
        if not cand:
            cand = PointBag(point)
#            cand.type = point.type
            cand_points.append(cand)
        else:
            cand.append(point)

    # accumulate detected object locations to get more accurate value
    @actor
    def get_found(self, max_time=10, min_count=10):
        cand_points = []
        while True:
            point = self.run_actor('find_object')
            if not point: return None
            trans = self.run_actor('map_trans')
            point.setTransform(trans.transform)
            if point.valid:
                self.register_flist(cand_points, point)
            max_count = 0
            target = None
            for c in cand_points:
                if c.count > max_count:
                    target = c
                    max_count = c.count
            if max_count >= min_count:
                return target
            
    # realtime object detection for visual feedback
    @actor('measure_distance', 'multi')
    def measure_distance(self, callback, target):
        bridge = self.get_value('cv_bridge')
        def stub(data):
            depth_image = bridge.imgmsg_to_cv2(data)
            mid_line = depth_image[len(depth_image)//2]
            return callback(min(mid_line)/320)
            
        depth_tran = self.run_actor_mode('depth', 'multi', stub)
        return ('close', lambda tran: depth_tran.close(depth_tran)),

    @actor
    def measure_center(self, target='link1', assumed=0.25, log=None):
        data = self.run_actor('depth')
        cv_bridge = self.get_value('cv_bridge')
        depth_image = cv_bridge.imgmsg_to_cv2(data)
        index = len(depth_image) - 220
        det_line = depth_image[index]
        index = det_line.argmin()
        distance = det_line[index] / 1000
        actual_distance = distance
        center = self.run_actor('pic_find')
        index = center[0] # by pic cell
        zp = center[1] # by pic cell
        if not center: return 0, 0, 0, 0
        if distance < 0.1:
            if assumed > 0:
                distance = assumed
            else: return 0, 0, 0, -1
            if log != None: log['assumed'] = True
        else:
            if log != None: log['assumed'] = False
        if log != None:
            log['index'] = index
            log['distance'] = distance
        if index == 0: print('index 0!!!')
        x, y = self.pix_to_coordinate(index, zp, distance)
        angle = atan2(y, x+0.07)
        if log != None:
            log['_y'] = y
            log['y'] = y
            log['dangle'] = angle
        return x, y, angle, actual_distance

    @actor
    def measure_center2(self, assumed=0.25, log=None):
        data = self.run_actor('depth')
        cv_bridge = self.get_value('cv_bridge')
        depth_image = cv_bridge.imgmsg_to_cv2(data)
        zp = len(depth_image) - 200
        det_line = depth_image[zp]
        d_index = det_line.argmin()
        distance = det_line[d_index] / 1000
        mes_distance = distance
        min_index = 600
        max_index = 0
        for i, v in enumerate(det_line):
            if v > 300: continue
            if i < min_index: min_index = i
            if i > max_index: max_index = i
        index = int((min_index + max_index) / 2)
        if distance < 0.1:
            if assumed > 0:
                distance = assumed
            else: return 0, 0, 0, -1
            if log != None: log['assumed'] = True
        else:
            if log != None: log['assumed'] = False
        if log != None:
            log['index'] = index
            log['distance'] = distance
        if index == 0: print('index 0!!!')
        x, y = self.pix_to_coordinate(index, zp, distance)
        angle = atan2(y, x+0.07)
        if log != None:
            log['_y'] = y
            log['y'] = y
            log['dangle'] = angle
        return x, y, angle, mes_distance

    # detect can center
    @actor
    def center_angle(self, assumed=0.25):
        data = self.run_actor('depth')
        cv_bridge = self.get_value('cv_bridge')
        depth_image = cv_bridge.imgmsg_to_cv2(data)
        det_line = depth_image[-220]
        index = det_line.argmin() + 19
        distance = det_line[index] / 640
        if distance < 0.1 and assumed > 0: distance = assumed
        x, y = self.pix_to_coordinate(index, distance, depth_image)
        angle = atan2(y, x)
        return angle, distance

    def pix_to_coordinate(self, x, y, distance):
        intrinsics = self.get_value('intrinsics')
        p = rs.rs2_deproject_pixel_to_point(intrinsics,[x,y], distance)
        return p[2],-p[0]
    
    def pic_to_depth(self, yp, zp):
        loc_z = zp / self.pic_shape[1] * self.depth_shape[1]
        loc_y = yp / self.pic_shape[0] * self.depth_shape[0]
        loc_z = int(loc_z)
        loc_y = int(loc_y)
        return loc_y, loc_z

    # find object location
    @actor
    def find_object(self):
        center = self.run_actor('pic_find')
        if not center: return None
        data = self.run_actor('depth')
        cv_bridge = self.get_value('cv_bridge')
        depth_image = cv_bridge.imgmsg_to_cv2(data)
        yp = center[0] # by pic cell
        zp = center[1] # by pic cell
        self.depth_shape = depth_image.shape
        rel_yp, rel_zp = self.adjust(yp, zp, self.depth_shape)
        distance = depth_image[rel_zp][rel_yp] / 1000
        if isinf(distance): return None
        if distance == 0:
            print('find_object zero distance')
            distance = 0.2
        target_x, target_y = self.pix_to_coordinate(yp, zp, distance)        
        point = PointEx(target_x, target_y)
        point.v_x = target_x
        point.distance = distance 
#        print(f'find_object x:{target_x}, y:{target_y}, distance:{distance}')
        return point

    # show object location in a format of pix number
    @actor
    def find_object_pic(self):
        center = self.run_actor('pic_find')
        if not center: return None
        data = self.run_actor('depth')
        cv_bridge = self.get_value('cv_bridge')
        depth_image = cv_bridge.imgmsg_to_cv2(data)
        yp = center[0] # by pic cell
        zp = center[1] # by pic cell
        self.depth_shape = depth_image.shape
        rel_yp, rel_zp = self.adjust(yp, zp, self.depth_shape)
        mid_y = self.depth_shape[1] // 2
        target_rate = rel_yp / mid_y
        det_line = depth_image[180]
        for i, v in enumerate(det_line):
            if v == 0: break
        edge_rage = i / mid_y
        return target_rate, edge_rage

    # find out target object from RGB image input
    @actor
    def pic_find(self):
        ret = None
        with self.run_actor_mode('pic_receiver', 'timed_iterator', 10) as pic_iter:
            for cv_image in pic_iter:
                ret = self.find_coke(cv_image)
                if ret[0] >= 0:
                    self.cv_image = cv_image
                    self.pic_shape = cv_image.shape
                    break
        if ret[0] < 0: return None
        return ret
    
    @actor
    def coke_getter(self):
        cv_image = self.run_actor('pic_receiver')
        return self.find_coke(cv_image)
        
    # get raw RGB image
    @actor('pic_receiver', 'multi')
    def pic_receiver(self, callback):
        def stub(data):
            cv_image = None
            cv_bridge = self.get_value('cv_bridge')
            try:
                cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
            return callback(cv_image)
        pic_tran = self.run_actor_mode('pic', 'multi', stub)
        return ('close', lambda tran: pic_tran.close(pic_tran)),

    def adjust(self, yp, zp, size):
        mid_y = size[1] // 2
        mid_z = size[0] // 2
        off_y = yp - mid_y
        off_z = zp - mid_z
        off_y = int(off_y * 0.75) + mid_y 
        off_z = int(off_z * 1.0) + mid_z
#        print(f'off_y:{off_y}, off_z:{off_z}')
        return off_y, off_z
    
    @actor
    def cdisp(self):
        center = self.run_actor('pic_find')
        if not center: return None
        data = self.run_actor('depth')
        cv_bridge = self.get_value('cv_bridge')
        depth_image = cv_bridge.imgmsg_to_cv2(data)
        yp = center[0] # by pic cell
        zp = center[1] # by pic cell
        self.depth_shape = depth_image.shape
#        s = self.depth_shape
#        print(f'shape_y:{s[1]}, shape_z:{s[0]}')
#        print(f'find_object yp:{yp}, zp:{zp}')

        rel_yp, rel_zp = self.adjust(yp, zp, self.depth_shape)
        distance = depth_image[rel_zp][rel_yp] / 1000
        if isinf(distance): return None
        radius = 20
        color = (0, 255, 0)
        print(f'distance:{distance}')
        cv2.circle(depth_image, (yp, zp), radius, 7000)
        cv2.imshow('test', depth_image)
        cv2.waitKey(0)

        target_x, target_y = self.pix_to_coordinate(yp, distance, depth_image)        
        point = PointEx(target_x, target_y)
        point.v_x = target_x
        point.distance = distance 
#        print(f'find_object x:{target_x}, y:{target_y}, distance:{distance}')
        return point
    
    def find_coke(self, cv_img):
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        hsv_min = np.array([150, 70, 0])
        hsv_max = np.array([180,255,255])
        m1 = cv2.inRange(hsv_img, hsv_min, hsv_max)

#        hsv_min = np.array([0, 70, 0])
#        hsv_max = np.array([30,255,255])
#        m2 = cv2.inRange(hsv_image, hsv_min, hsv_max)

        mask = m1

        cv_img  = cv2.bitwise_and(cv_img, cv_img, mask = mask)

        bw_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        m = cv2.moments(bw_img, True)
        w = m["m00"]
        x = m["m10"]
        y = m["m01"]
        if w == 0:
            return -1, -1
        else:
            return int(x / w), int(y / w)   
