import torch
from torch import nn, optim


def create_training_components(model, learning_rate):
    """
    Create all components required for training.

    Returns:
        criterion, optimizer, scheduler, scaler
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5,
    )

    scaler = torch.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    return (
        criterion,
        optimizer,
        scheduler,
        scaler,
    )