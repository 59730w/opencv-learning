import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ManifestImageDataset
from src.model import build_resnet18
from src.transforms import build_eval_transform


DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "BarkVN-50"
    / "v1"
    / "images"
    / "BarkVN-50_mendeley"
)
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "split_manifest.csv"
CLASS_MAP_PATH = PROJECT_ROOT / "datasets" / "processed" / "class_to_idx.json"


def select_balanced_indices(dataset, class_count=8, images_per_class=2):
    selected_classes = []
    selected_counts = Counter()
    selected_indices = []

    for index, record in enumerate(dataset.records):
        class_name = record["class_name"]

        if class_name not in selected_classes:
            if len(selected_classes) >= class_count:
                continue
            selected_classes.append(class_name)

        if selected_counts[class_name] < images_per_class:
            selected_indices.append(index)
            selected_counts[class_name] += 1

        if (
            len(selected_classes) == class_count
            and all(
                selected_counts[name] == images_per_class
                for name in selected_classes
            )
        ):
            break

    return selected_indices, selected_classes


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split="train",
        transform=build_eval_transform(),
    )

    indices, class_names = select_balanced_indices(dataset)
    tiny_dataset = Subset(dataset, indices)

    loader = DataLoader(
        tiny_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    images, labels, _ = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    model = build_resnet18(
        num_classes=50,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )

    final_loss = None
    final_accuracy = None

    for step in range(1, 201):
        model.train()
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

        final_loss = loss.item()
        final_accuracy = accuracy

        if step == 1 or step % 20 == 0:
            print(
                f"步骤 {step:03d}/200 | "
                f"损失 {final_loss:.4f} | "
                f"准确率 {final_accuracy:.2%}"
            )

    print("设备:", device)
    print("小样本数量:", len(tiny_dataset))
    print("涉及类别数:", len(class_names))
    print("最终损失:", round(final_loss, 4))
    print("最终准确率:", f"{final_accuracy:.2%}")

    assert len(tiny_dataset) == 16
    assert len(class_names) == 8
    assert final_accuracy >= 0.95
    assert final_loss < 0.20

    print("第四天第四步验证成功")


if __name__ == "__main__":
    main()