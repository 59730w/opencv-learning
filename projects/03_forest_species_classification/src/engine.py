import torch
from sklearn.metrics import f1_score
from torch import nn


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp=True,
    freeze_backbone=False,
):
    model.train()

    # 只训练分类头时，固定骨干网络的BatchNorm统计量
    if freeze_backbone:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    amp_enabled = use_amp and device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("训练损失出现NaN或Inf")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes=50,
    use_amp=True,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    amp_enabled = use_amp and device.type == "cuda"

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": macro_f1,
    }