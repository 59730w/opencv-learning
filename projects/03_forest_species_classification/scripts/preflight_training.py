import json
import math
import sys
from pathlib import Path

import torch


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
CONFIG_PATH = PROJECT_ROOT / "configs" / "resnet18_baseline.json"


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert torch.cuda.is_available(), "CUDA不可用，暂不启动正式训练"

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    train_dataset = ManifestImageDataset(
        DATA_ROOT,
        MANIFEST_PATH,
        CLASS_MAP_PATH,
        "train",
        build_eval_transform(),
    )
    validation_dataset = ManifestImageDataset(
        DATA_ROOT,
        MANIFEST_PATH,
        CLASS_MAP_PATH,
        "validation",
        build_eval_transform(),
    )

    model = build_resnet18(
        num_classes=config["num_classes"],
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    head_trainable = count_trainable_parameters(model)

    for parameter in model.parameters():
        parameter.requires_grad = True

    finetune_trainable = count_trainable_parameters(model)

    train_batches = math.ceil(
        len(train_dataset) / config["batch_size"]
    )
    validation_batches = math.ceil(
        len(validation_dataset) / config["batch_size"]
    )
    total_epochs = (
        config["head_epochs"] + config["finetune_epochs"]
    )

    print("Python:", sys.executable)
    print("设备:", device)
    print("显卡:", gpu_name)
    print("训练集:", len(train_dataset))
    print("验证集:", len(validation_dataset))
    print("批次大小:", config["batch_size"])
    print("每轮训练批次数:", train_batches)
    print("每轮验证批次数:", validation_batches)
    print("分类头阶段可训练参数:", head_trainable)
    print("微调阶段可训练参数:", finetune_trainable)
    print("分类头训练轮数:", config["head_epochs"])
    print("全模型微调轮数:", config["finetune_epochs"])
    print("总轮数:", total_epochs)
    print("AMP启用:", config["use_amp"])

    assert len(train_dataset) == 3891
    assert len(validation_dataset) == 844
    assert config["batch_size"] == 16
    assert head_trainable == 25650
    assert finetune_trainable == 11202162
    assert total_epochs == 15
    assert config["use_amp"] is True

    print("第四天第八步验证成功，可以启动正式训练")


if __name__ == "__main__":
    main()