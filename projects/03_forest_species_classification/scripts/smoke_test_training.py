import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ManifestImageDataset
from src.model import build_resnet18
from src.transforms import build_train_transform


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


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split="train",
        transform=build_train_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    model = build_resnet18(
        num_classes=50,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=0.001,
        weight_decay=0.0001,
    )

    images, labels, _ = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    old_fc_weight = model.fc.weight.detach().clone()

    model.train()
    optimizer.zero_grad()

    logits = model(images)
    loss = criterion(logits, labels)

    assert torch.isfinite(loss), "损失出现 NaN 或 Inf"

    loss.backward()
    optimizer.step()

    fc_changed = not torch.equal(old_fc_weight, model.fc.weight.detach())
    backbone_has_grad = any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    )

    print("设备:", device)
    print("批次形状:", tuple(images.shape))
    print("输出形状:", tuple(logits.shape))
    print("损失:", round(loss.item(), 4))
    print("分类头参数已更新:", fc_changed)
    print("冻结骨干网络产生梯度:", backbone_has_grad)

    assert logits.shape == (16, 50)
    assert fc_changed
    assert not backbone_has_grad

    print("第四天第三步验证成功")


if __name__ == "__main__":
    main()