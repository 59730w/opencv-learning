from pathlib import Path

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "yolov8n.pt"
    data_path = (
        project_root
        / "datasets"
        / "wildfire_smoke"
        / "data.yaml"
    )

    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        patience=10,
        seed=42,
        project=str(project_root / "outputs"),
        name="wildfire_train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
