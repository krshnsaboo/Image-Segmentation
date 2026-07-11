import torch
from torch.utils.data import DataLoader, random_split

from configs.config import (
    IMAGE_DIR,
    MASK_DIR,
    NUM_CLASSES,
    BATCH_SIZE,
    VAL_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    PERSISTENT_WORKERS,
    BEST_CHECKPOINT,
)

from src.datasets.segmentation_dataset import SegmentationDataset
from src.models.deeplabv3 import build_model
from src.training.checkpoints import load_checkpoint
from src.evaluation.evaluator import Evaluator


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = SegmentationDataset(
        IMAGE_DIR,
        MASK_DIR,
    )

    generator = torch.Generator().manual_seed(42)

    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
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

    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=NUM_CLASSES,
    )

    results = evaluator.evaluate()

    print("\nEvaluation Results")
    print("-" * 40)
    print(f"Mean IoU        : {results['mIoU']:.4f}")
    print(f"Pixel Accuracy  : {results['pixel_accuracy']:.4f}")

    evaluator.plot_per_class_iou(
        results["per_class_iou"]
    )


if __name__ == "__main__":
    main()