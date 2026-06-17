import py_trees

from pytwb.common import behavior
from lib.actor_bt import ActorBT

@behavior
class QRScan(ActorBT):
    def __init__(self, name):
        super().__init__(name, 'qr_scan')
