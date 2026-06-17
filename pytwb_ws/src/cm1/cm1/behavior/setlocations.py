import py_trees

from pytwb.common import behavior
from ros_actor import get_value
from lib.actor_bt import ActorBT

#
## decide visit points based on information from vector map
#
@behavior
class SetLocations(py_trees.behaviour.Behaviour):
    desc = 'create candidates of visit locations by vector map'
    
    def __init__(self, name):
        super(SetLocations, self).__init__(name)
    
    def initialise(self) -> None:
        bb = py_trees.blackboard.Blackboard()
        self.bb = bb
        world = get_value('world')
        r = world.get_root_region()
        self.region = r
        pose_list = []
        for sr in r.get_subregions():
            c = sr.get_weight_center()
            pose_list.append([float(c.x), float(c.y), 0])
        bb.set("pose_list", pose_list)

    def update(self):
        return py_trees.common.Status.SUCCESS



# Akaoka Yuu

import math
@behavior
class SetLocQR(py_trees.behaviour.Behaviour):

    def __init__(self, name):
        super().__init__("set_loc_qr")
        self.value = None

    def initialise(self):

        QRdict ={
            "A" : [-0.48, -1, 270],
            "B" : [0.17, -1, 270],
            "C" : [0.8, -1, 270],
        }
        bb = py_trees.blackboard.Blackboard()
        self.bb = bb
        QRdata = bb.get("latest_qr")
        
        self.value = QRdict[QRdata]
        self.value = list(self.value)
        self.value = [float(self.value[0]), float(self.value[1]), math.radians(self.value[2])]
        self.value = tuple(self.value)
        self.bb.set("target_pose", self.value)
    
    def update(self):
        return py_trees.common.Status.SUCCESS