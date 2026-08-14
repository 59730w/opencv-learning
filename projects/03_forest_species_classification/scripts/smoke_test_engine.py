import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ManifestImageDataset
from src.engine import evaluate, train_one_epoch
from src.model import build_resnet18
from src.transforms import build_eval_transform, build_train_transform


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


def build_dataset(split, transform):
    return ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split=split,
        transform=transform,
    )


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    full_train_dataset = build_dataset(
        split="train",
        transform=build_train_transform(),
    )
    full_validation_dataset = build_dataset(
        split="validation",
        transform=build_eval_transform(),
    )

    # 只取少量数据测试引擎，避免运行完整训练
    train_dataset = Subset(full_train_dataset, range(32))
    validation_dataset = Subset(
        full_validation_dataset,
        range(32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = build_resnet18(
        num_classes=50,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=0.001,
        weight_decay=0.0001,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
        init_scale=1024.0,
    )

    old_fc_weight = model.fc.weight.detach().clone()

    train_metrics = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        use_amp=use_amp,
        freeze_backbone=True,
    )

    validation_metrics = evaluate(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
        num_classes=50,
        use_amp=use_amp,
    )

    fc_changed = not torch.equal(
        old_fc_weight,
        model.fc.weight.detach(),
    )

    print("设备:", device)
    print("AMP启用:", use_amp)
    print("训练样本数:", len(train_dataset))
    print("验证样本数:", len(validation_dataset))
    print("训练指标:", train_metrics)
    print("验证指标:", validation_metrics)
    print("分类头参数已更新:", fc_changed)

    assert math.isfinite(train_metrics["loss"])
    assert math.isfinite(validation_metrics["loss"])
    assert 0 <= train_metrics["accuracy"] <= 1
    assert 0 <= validation_metrics["accuracy"] <= 1
    assert 0 <= validation_metrics["macro_f1"] <= 1
    assert fc_changed

    print("第四天第六步验证成功")


if __name__ == "__main__":
    main()
