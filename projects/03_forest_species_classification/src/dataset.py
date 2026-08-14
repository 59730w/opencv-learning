import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "validation", "test"}


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        data_root,
        manifest_path,
        class_map_path,
        split,
        transform=None,
    ):
        if split not in VALID_SPLITS:
            raise ValueError(f"未知数据子集：{split}")

        self.data_root = Path(data_root)
        self.transform = transform
        self.split = split
        self.class_to_idx = json.loads(
            Path(class_map_path).read_text(encoding="utf-8")
        )

        with Path(manifest_path).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            self.records = [
                row
                for row in csv.DictReader(file)
                if row["split"] == split
            ]

        if not self.records:
            raise ValueError(f"数据子集为空：{split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        relative_path = record["relative_path"]
        image_path = self.data_root / Path(relative_path)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(record["class_index"])
        expected_label = self.class_to_idx[record["class_name"]]

        if label != expected_label:
            raise ValueError(
                f"类别索引不一致：{relative_path}，"
                f"清单={label}，映射={expected_label}"
            )

        return image, label, relative_path
