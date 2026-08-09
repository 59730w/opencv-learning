from pathlib import Path

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path = (
        project_root
        / "outputs"
        / "wildfire_train"
        / "weights"
        / "best.pt"
    )
    data_path = (
        project_root
        / "datasets"
        / "wildfire_smoke"
        / "data.yaml"
    )

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        project=str(project_root / "outputs"),
        name="evaluation",
        exist_ok=True,
    )

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
