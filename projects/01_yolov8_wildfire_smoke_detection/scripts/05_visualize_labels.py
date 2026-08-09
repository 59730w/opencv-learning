import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "datasets" / "wildfire_smoke"
    image_dir = dataset_root / "train" / "images"
    label_dir = dataset_root / "train" / "labels"

    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    random.seed(42)
    sampled_images = random.sample(image_paths, 20)

    figure, axes = plt.subplots(4, 5, figsize=(16, 12))

    for axis, image_path in zip(axes.flat, sampled_images):
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        label_path = label_dir / f"{image_path.stem}.txt"

        axis.imshow(image)
        axis.set_title(image_path.name, fontsize=7)
        axis.axis("off")

        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()

            if len(values) != 5:
                raise ValueError(f"标签格式错误：{label_path}")

            class_id = int(values[0])
            x_center, y_center, box_width, box_height = map(float, values[1:])

            if class_id != 0:
                raise ValueError(f"发现非法类别编号：{label_path}")

            if not all(
                0 <= value <= 1
                for value in (x_center, y_center, box_width, box_height)
            ):
                raise ValueError(f"标注坐标超出范围：{label_path}")

            left = (x_center - box_width / 2) * image_width
            top = (y_center - box_height / 2) * image_height
            width = box_width * image_width
            height = box_height * image_height

            box = Rectangle(
                (left, top),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axis.add_patch(box)
            axis.text(
                left,
                top,
                "smoke",
                color="white",
                fontsize=7,
                backgroundcolor="red",
            )

    output_dir = project_root / "outputs" / "dataset_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_labels_preview.jpg"

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print("已随机检查图片数量：", len(sampled_images))
    print("标注预览图：", output_path)


if __name__ == "__main__":
    main()