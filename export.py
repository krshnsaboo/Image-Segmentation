import torch
from torch import nn

from configs.config import (
    NUM_CLASSES,
    BEST_CHECKPOINT,
    TORCHSCRIPT_MODEL,
)

from src.models.deeplabv3 import build_model
from src.training.checkpoints import load_checkpoint


class TorchScriptWrapper(nn.Module):
    """
    Wrapper so TorchScript returns only the segmentation tensor.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def main():

    device = torch.device("cpu")

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    load_checkpoint(
        path=BEST_CHECKPOINT,
        model=model,
        device=device,
    )

    model.eval()

    wrapped_model = TorchScriptWrapper(model)

    example_input = torch.randn(
        1,
        3,
        512,
        512,
    )

    traced_model = torch.jit.trace(
        wrapped_model,
        example_input,
    )

    traced_model.save(TORCHSCRIPT_MODEL)

    print("TorchScript model exported successfully.")


if __name__ == "__main__":
    main()