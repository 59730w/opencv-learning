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
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_baseline"
    / "best.pt"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    assert checkpoint["epoch"] == 13
    assert checkpoint["stage"] == "finetune"
    assert checkpoint["config"]["num_classes"] == 50

    model = build_resnet18(
        num_classes=50,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    dataset = ManifestImageDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        class_map_path=CLASS_MAP_PATH,
        split="validation",
        transform=build_eval_transform(),
    )

    image, label, relative_path = dataset[0]
    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)

    prediction = probabilities.argmax(dim=1).item()
    confidence = probabilities.max(dim=1).values.item()

    print("设备:", device)
    print("检查点轮次:", checkpoint["epoch"])
    print("检查点阶段:", checkpoint["stage"])
    print(
        "检查点验证Macro-F1:",
        round(checkpoint["validation_metrics"]["macro_f1"], 4),
    )
    print("缺失参数:", load_result.missing_keys)
    print("多余参数:", load_result.unexpected_keys)
    print("验证图片:", relative_path)
    print("真实类别索引:", label)
    print("预测类别索引:", prediction)
    print("预测置信度:", f"{confidence:.2%}")
    print("输出形状:", tuple(logits.shape))

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert logits.shape == (1, 50)
    assert torch.isfinite(logits).all()
    assert torch.isclose(
        probabilities.sum(),
        torch.tensor(1.0, device=device),
        atol=1e-5,
    )

    print("第四天第十二步验证成功")


if __name__ == "__main__":
    main()