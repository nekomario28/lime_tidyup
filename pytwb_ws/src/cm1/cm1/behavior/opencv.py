import py_trees
from lib.actor_bt import ActorBT
from pytwb.common import behavior


@behavior
class QRScan(ActorBT):
    def __init__(self, name, node):
        super().__init__(name, "qr_scan")
