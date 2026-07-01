from ultralytics import YOLO

# 学習済みモデルを読み込む
model = YOLO("runs/pose/train-2/weights/best.pt")

# 推論
results = model.predict(
    source="/tmp/test.jpg",
    save=True,
    show=False
)

print(results)