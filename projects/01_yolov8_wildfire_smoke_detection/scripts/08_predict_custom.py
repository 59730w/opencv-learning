from pathlib import Path

from ultralytics import YOLO


def predict_group(model, input_dir, output_root, group_name):
    results = model.predict(
        source=str(input_dir),
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_root),
        name=group_name,
        exist_ok=True,
    )

    detected_images = sum(len(result.boxes) > 0 for result in results)
    total_boxes = sum(len(result.boxes) for result in results)

    print(f"{group_name} 图片数量：{len(results)}")
    print(f"{group_name} 检出图片数量：{detected_images}")
    print(f"{group_name} 检测框总数：{total_boxes}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path = (
        project_root
        / "outputs"
        / "wildfire_train"
        / "weights"
        / "best.pt"
    )

    model = YOLO(str(model_path))
    output_root = project_root / "outputs" / "predictions"

    predict_group(
        model,
        project_root / "test_images" / "smoke",
        output_root,
        "smoke",
    )
    predict_group(
        model,
        project_root / "test_images" / "no_smoke",
        output_root,
        "no_smoke",
    )


if __name__ == "__main__":
    main()