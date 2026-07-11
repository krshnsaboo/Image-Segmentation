import torch


def compute_iou(pred: torch.Tensor,
                target: torch.Tensor,
                num_classes: int) -> float:
    """
    Compute mean Intersection over Union (mIoU).
    """

    ious = []

    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return torch.tensor(ious).nanmean().item()


def compute_pixel_accuracy(pred: torch.Tensor,
                           target: torch.Tensor) -> float:
    """
    Compute pixel accuracy.
    """

    correct = (pred == target).float()

    return (correct.sum() / correct.numel()).item()