import torch
from tqdm import tqdm
from pathlib import Path
from src.training.checkpoints import save_checkpoint

from src.evaluation.metrics import (
    compute_iou,
    compute_pixel_accuracy,
)


class Trainer:
    """
    Trainer class responsible for model training and validation.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        num_classes,
        checkpoint_dir,
        early_stopping_patience,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.
        """

        self.model.train()

        epoch_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]"
        )

        for images, masks in progress_bar:

            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda"
            ):
                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return epoch_loss / len(self.train_loader)

    def validate(self):
        """
        Evaluate the model on the validation set.
        """

        self.model.eval()

        val_loss = 0.0
        iou_scores = []
        acc_scores = []

        with torch.no_grad():

            for images, masks in tqdm(
                self.val_loader,
                desc="Validation"
            ):

                images = images.to(self.device)
                masks = masks.to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda"
                ):
                    outputs = self.model(images)["out"]
                    loss = self.criterion(outputs, masks)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                iou_scores.append(
                    compute_iou(
                        preds,
                        masks,
                        self.num_classes,
                    )
                )

                acc_scores.append(
                    compute_pixel_accuracy(
                        preds,
                        masks,
                    )
                )

        avg_loss = val_loss / len(self.val_loader)
        mean_iou = sum(iou_scores) / len(iou_scores)
        pixel_acc = sum(acc_scores) / len(acc_scores)

        return avg_loss, mean_iou, pixel_acc

    def train(self, num_epochs, start_epoch=0,):
        """
        Complete training loop.
        """

        best_val_iou = 0.0
        self.early_stopping_patience
        epochs_without_improvement = 0

        for epoch in range(start_epoch,num_epochs):

            train_loss = self.train_one_epoch(epoch)

            val_loss, mean_iou, pixel_acc = self.validate()

            self.scheduler.step(mean_iou)

            latest_checkpoint = Path(self.checkpoint_dir) / "latest.pt"

            save_checkpoint(
                path=latest_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch + 1,
                num_classes=self.num_classes,
                val_loss=val_loss,
                val_iou=mean_iou,
                val_acc=pixel_acc,
            )

            print("\n" + "=" * 60)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("=" * 60)
            print(f"Train Loss      : {train_loss:.4f}")
            print(f"Validation Loss : {val_loss:.4f}")
            print(f"Mean IoU        : {mean_iou:.4f}")
            print(f"Pixel Accuracy  : {pixel_acc:.4f}")

            if mean_iou > best_val_iou:
                best_val_iou = mean_iou
                epochs_without_improvement = 0

                print("✓ Best validation IoU improved.")

                best_checkpoint = Path(self.checkpoint_dir) / "best_iou.pt"

                save_checkpoint(
                    path=best_checkpoint,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch + 1,
                    num_classes=self.num_classes,
                    val_loss=val_loss,
                    val_iou=mean_iou,
                    val_acc=pixel_acc,
                )

            else:
                epochs_without_improvement += 1

                print(
                    f"No improvement for "
                    f"{epochs_without_improvement} epoch(s)."
                )

            if (
                epochs_without_improvement
                >= self.early_stopping_patience
            ):
                print("\nEarly stopping triggered.")
                break