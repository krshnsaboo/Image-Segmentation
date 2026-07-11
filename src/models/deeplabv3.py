from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet101


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a DeepLabV3-ResNet101 model for semantic segmentation.

    Args:
        num_classes: Number of segmentation classes.
        pretrained: Whether to use ImageNet pretrained backbone weights.

    Returns:
        Configured DeepLabV3 model.
    """

    if pretrained:
        model = deeplabv3_resnet101(
            weights="DEFAULT",
            aux_loss=True,
        )
    else:
        model = deeplabv3_resnet101(
            weights=None,
            aux_loss=True,
        )

    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1
    )

    return model