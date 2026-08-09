from pathlib import Path

from ultralytics import YOLO


project_root = Path(__file__).resolve().parents[1]

model = YOLO(str(project_root / "yolov8n.pt"))

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    conf=0.25,
    save=True,
    project=str(project_root / "outputs"),
    name="pretrained_prediction",
    exist_ok=True,
)

print("检测框数量：", len(results[0].boxes))
print("结果保存位置：", results[0].save_dir)