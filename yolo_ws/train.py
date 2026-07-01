from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

model.train(
    data="datasets/Cube_Pose_Estimation.v1-test1.yolov8/data.yaml",
    epochs=100,
    imgsz=640
)