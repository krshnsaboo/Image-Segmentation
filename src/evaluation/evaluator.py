import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluates a semantic segmentation model on a dataset.
    """

    def __init__(self, model, dataloader, device, num_classes):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes

    def evaluate(self):
        """
        Run complete evaluation.

        Returns:
            dict containing evaluation metrics.
        """

        confusion_matrix = self._compute_confusion_matrix()

        metrics = self._compute_metrics(confusion_matrix)

        return metrics

    def _compute_confusion_matrix(self):

        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():

            for images, masks in tqdm(
                self.dataloader,
                desc="Evaluating"
            ):

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)["out"]

                predictions = torch.argmax(outputs, dim=1)

                predictions = predictions.view(-1)
                masks = masks.view(-1)

                valid_pixels = (
                    (masks >= 0)
                    & (masks < self.num_classes)
                )

                indices = (
                    self.num_classes * masks[valid_pixels]
                    + predictions[valid_pixels]
                ).cpu()

                batch_confusion = torch.zeros(
                    self.num_classes * self.num_classes,
                    dtype=torch.int64,
                )

                batch_confusion.scatter_add_(
                    dim=0,
                    index=indices,
                    src=torch.ones_like(indices),
                )

                confusion_matrix += batch_confusion.view(
                    self.num_classes,
                    self.num_classes,
                )

        return confusion_matrix

    def _compute_metrics(self, confusion_matrix):

        intersection = torch.diag(confusion_matrix)

        union = (
            confusion_matrix.sum(dim=1)
            + confusion_matrix.sum(dim=0)
            - intersection
        )

        per_class_iou = intersection.float() / union.float()

        mean_iou = torch.nanmean(per_class_iou).item()

        pixel_accuracy = (
            intersection.sum().float()
            / confusion_matrix.sum().float()
        ).item()

        return {
            "mIoU": mean_iou,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": per_class_iou,
            "confusion_matrix": confusion_matrix,
        }

    def plot_per_class_iou(self, per_class_iou):

        plt.figure(figsize=(18, 5))

        plt.bar(
            range(len(per_class_iou)),
            per_class_iou.cpu().numpy(),
        )

        plt.xlabel("Class ID")
        plt.ylabel("IoU")
        plt.title("Per-Class IoU")

        plt.show()