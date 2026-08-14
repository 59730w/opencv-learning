import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


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
CONFIG_PATH = PROJECT_ROOT / "configs" / "resnet18_baseline.json"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "resnet18_baseline"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18_baseline"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(split, transform):
    return ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split=split,
        transform=transform,
    )


def save_checkpoint(
    path,
    model,
    config,
    epoch,
    stage,
    validation_metrics,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "stage": stage,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    set_seed(config["seed"])

    device = torch.device(
        "cuda"
        if config["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    use_amp = config["use_amp"] and device.type == "cuda"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(
        "train",
        build_train_transform(),
    )
    validation_dataset = build_dataset(
        "validation",
        build_eval_transform(),
    )

    generator = torch.Generator()
    generator.manual_seed(config["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_resnet18(
        num_classes=config["num_classes"],
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
        init_scale=1024.0,
    )

    history = []
    best_macro_f1 = -1.0
    global_epoch = 0

    stages = [
        {
            "name": "head",
            "epochs": config["head_epochs"],
            "learning_rate": config["head_learning_rate"],
            "freeze_backbone": True,
        },
        {
            "name": "finetune",
            "epochs": config["finetune_epochs"],
            "learning_rate": config["finetune_learning_rate"],
            "freeze_backbone": False,
        },
    ]

    print("设备:", device)
    print("训练集:", len(train_dataset))
    print("验证集:", len(validation_dataset))
    print("AMP启用:", use_amp)

    for stage in stages:
        if not stage["freeze_backbone"]:
            for parameter in model.parameters():
                parameter.requires_grad = True

        optimizer = AdamW(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            lr=stage["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        print(
            f"\n开始阶段: {stage['name']} | "
            f"轮数: {stage['epochs']} | "
            f"学习率: {stage['learning_rate']}"
        )

        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                freeze_backbone=stage["freeze_backbone"],
            )
            validation_metrics = evaluate(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                num_classes=config["num_classes"],
                use_amp=use_amp,
            )

            record = {
                "epoch": global_epoch,
                "stage": stage["name"],
                "stage_epoch": stage_epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(record)

            HISTORY_PATH.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            save_checkpoint(
                CHECKPOINT_DIR / "last.pt",
                model,
                config,
                global_epoch,
                stage["name"],
                validation_metrics,
            )

            if validation_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = validation_metrics["macro_f1"]
                save_checkpoint(
                    CHECKPOINT_DIR / "best.pt",
                    model,
                    config,
                    global_epoch,
                    stage["name"],
                    validation_metrics,
                )
                best_mark = "，已保存最佳模型"
            else:
                best_mark = ""

            print(
                f"Epoch {global_epoch:02d} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"train acc {train_metrics['accuracy']:.2%} | "
                f"val loss {validation_metrics['loss']:.4f} | "
                f"val acc {validation_metrics['accuracy']:.2%} | "
                f"val macro-F1 {validation_metrics['macro_f1']:.4f}"
                f"{best_mark}"
            )

    print("\n正式基线训练完成")
    print("最佳验证Macro-F1:", round(best_macro_f1, 4))
    print("最佳模型:", CHECKPOINT_DIR / "best.pt")
    print("训练记录:", HISTORY_PATH)


if __name__ == "__main__":
    main()