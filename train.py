import torch
from torch.utils.data import DataLoader, random_split

from configs.config import (
    IMAGE_DIR,
    MASK_DIR,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    VAL_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    PERSISTENT_WORKERS,
    CHECKPOINT_DIR,
    LATEST_CHECKPOINT,
    EARLY_STOPPING_PATIENCE,
)

from src.datasets.segmentation_dataset import SegmentationDataset
from src.models.deeplabv3 import build_model
from src.training.setup import create_training_components
from src.training.trainer import Trainer
from src.training.checkpoints import load_checkpoint


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = SegmentationDataset(
        IMAGE_DIR,
        MASK_DIR,
    )

    generator = torch.Generator().manual_seed(42)

    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
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

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model.to(device)

    # -------------------------------------------------
    # Training Components
    # -------------------------------------------------

    criterion, optimizer, scheduler, scaler = (
        create_training_components(
            model=model,
            learning_rate=LEARNING_RATE,
        )
    )

    # -------------------------------------------------
    # Resume Training
    # -------------------------------------------------

    start_epoch = 0

    if LATEST_CHECKPOINT.exists():

        checkpoint = load_checkpoint(
            path=LATEST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        start_epoch = checkpoint["epoch"]

        print(
            f"Resuming training from epoch {start_epoch}"
        )

    else:

        print("Starting training from scratch.")

    # -------------------------------------------------
    # Trainer
    # -------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_classes=NUM_CLASSES,
        checkpoint_dir=CHECKPOINT_DIR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    trainer.train(
        num_epochs=NUM_EPOCHS,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()