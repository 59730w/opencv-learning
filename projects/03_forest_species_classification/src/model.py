from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    num_classes=50,
    pretrained=True,
    freeze_backbone=False,
):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model