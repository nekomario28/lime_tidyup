import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

IMAGE_PATH = "/tmp/test.jpg"
MODEL_PATH = "runs/pose/train-2/weights/best.pt"

class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")
        self.model = YOLO(MODEL_PATH)
        self.pub = self.create_publisher(String, "/cube_pose_result", 10)
        self.timer = self.create_timer(1.0, self.detect)

    def detect(self):
        if not os.path.exists(IMAGE_PATH):
            self.get_logger().info("まだ画像がありません")
            return

        results = self.model.predict(source=IMAGE_PATH, conf=0.3, save=False)
        r = results[0]

        data = {
            "boxes": r.boxes.xyxy.tolist() if r.boxes is not None else [],
            "classes": r.boxes.cls.tolist() if r.boxes is not None else [],
            "conf": r.boxes.conf.tolist() if r.boxes is not None else [],
            "keypoints": r.keypoints.xy.tolist() if r.keypoints is not None else []
        }

        msg = String()
        msg.data = json.dumps(data)
        self.pub.publish(msg)
        self.get_logger().info("YOLO結果をpublishしました")

def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()