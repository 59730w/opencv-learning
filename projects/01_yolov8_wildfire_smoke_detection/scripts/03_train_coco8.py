from pathlib import Path

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[1]
    model = YOLO(str(project_root / "yolov8n.pt"))

    model.train(
        data="coco8.yaml",
        epochs=3,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(project_root / "outputs"),
        name="coco8",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()