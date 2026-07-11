import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from configs.config import (
    NUM_CLASSES,
    BEST_CHECKPOINT,
    OUTPUT_DIR,
)

from src.models.deeplabv3 import build_model
from src.inference.predictor import Predictor
from src.training.checkpoints import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Inference"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    load_checkpoint(
        path=BEST_CHECKPOINT,
        model=model,
        device=device,
    )

    predictor = Predictor(
        model=model,
        device=device,
    )

    results = predictor.predict(args.image)

    plt.imsave(
        OUTPUT_DIR / "prediction.png",
        results["mask"],
    )

    plt.imsave(
        OUTPUT_DIR / "overlay.png",
        results["overlay"],
    )

    plt.imsave(
        OUTPUT_DIR / "confidence.png",
        results["confidence"],
        cmap="gray",
    )

    print()

    print("Prediction completed successfully.")

    print(f"Mask saved to       : {OUTPUT_DIR/'prediction.png'}")
    print(f"Overlay saved to    : {OUTPUT_DIR/'overlay.png'}")
    print(f"Confidence saved to : {OUTPUT_DIR/'confidence.png'}")


if __name__ == "__main__":
    main()