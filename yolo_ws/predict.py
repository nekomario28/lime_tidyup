from ultralytics import YOLO

MODEL_PATH = "/root/yolo_ws/runs/pose/train-2/best.pt"
IMAGE_PATH = "/tmp/test.jpg"

model = YOLO(MODEL_PATH)

results = model(IMAGE_PATH, save=True)

for r in results:
    print("names:", r.names)

    print("boxes xyxy:")
    if r.boxes is not None:
        print(r.boxes.xyxy)
        print("classes:")
        print(r.boxes.cls)
        print("confidence:")
        print(r.boxes.conf)
    else:
        print(None)

    print("keypoints xy:")
    if r.keypoints is not None:
        print(r.keypoints.xy)
    else:
        print(None)

    print("saved to:", r.save_dir)
