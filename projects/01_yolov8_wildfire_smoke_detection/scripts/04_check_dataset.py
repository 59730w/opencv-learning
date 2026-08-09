from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    project_root = Path(__file__).resolve().parents[1]
    yaml_path = project_root / "datasets" / "wildfire_smoke" / "data.yaml"

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("类别数量：", config["nc"])
    print("类别名称：", config["names"])

    if config["nc"] != len(config["names"]):
        raise ValueError("nc 与 names 的数量不一致")

    for split in ("train", "val", "test"):
        image_dir = (yaml_path.parent / config[split]).resolve()
        label_dir = image_dir.parent / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"{split} 的图片或标签目录不存在")

        image_names = {
            path.stem
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
        label_names = {path.stem for path in label_dir.glob("*.txt")}

        missing_labels = image_names - label_names
        extra_labels = label_names - image_names

        print(
            f"{split}: 图片={len(image_names)}, "
            f"标签={len(label_names)}, "
            f"缺失标签={len(missing_labels)}, "
            f"多余标签={len(extra_labels)}"
        )

        if missing_labels or extra_labels:
            raise RuntimeError(f"{split} 的图片与标签没有正确配对")

    print("数据集目录与文件配对检查通过")


if __name__ == "__main__":
    main()