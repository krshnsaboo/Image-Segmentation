import os
import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    num_classes,
    **metrics,
):
    """
    Save a training checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "num_classes": num_classes,
    }

    checkpoint.update(metrics)

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device="cpu",
):
    """
    Load a training checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if scaler is not None:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    return checkpoint
