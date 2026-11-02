from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import IGNORE_INDEX


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
    smooth: float = 1.0,
) -> torch.Tensor:
    valid = targets != ignore_index
    if not torch.any(valid):
        raise ValueError("Batch contains no valid target pixels for Dice loss.")

    safe_targets = targets.masked_fill(~valid, 0)
    probabilities = logits.softmax(dim=1)
    valid_float = valid.unsqueeze(1).to(probabilities.dtype)
    predicted_sums = (probabilities * valid_float).sum(dim=(0, 2, 3))
    matched_probabilities = probabilities.gather(
        1, safe_targets.unsqueeze(1)
    ).squeeze(1)

    intersections = torch.zeros(
        num_classes, device=logits.device, dtype=probabilities.dtype
    )
    intersections.scatter_add_(0, safe_targets[valid], matched_probabilities[valid])
    target_counts = torch.bincount(
        targets[valid], minlength=num_classes
    ).to(probabilities.dtype)

    dice = (2 * intersections + smooth) / (
        predicted_sums + target_counts + smooth
    )
    return 1 - dice.mean()


def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    predictions = logits.argmax(dim=1)
    valid = targets != ignore_index
    encoded = targets[valid] * num_classes + predictions[valid]
    confusion += torch.bincount(
        encoded, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, object]:
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    actual = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    false_negative = actual - true_positive
    false_positive = predicted - true_positive
    total = matrix.sum()
    true_negative = total - true_positive - false_negative - false_positive

    iou_denominator = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    sensitivity_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive

    def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(numerator, torch.nan)
        valid = denominator > 0
        result[valid] = numerator[valid] / denominator[valid]
        return result

    per_class_iou = safe_ratio(true_positive, iou_denominator)
    per_class_dice = safe_ratio(2 * true_positive, dice_denominator)
    per_class_sensitivity = safe_ratio(true_positive, sensitivity_denominator)
    per_class_specificity = safe_ratio(true_negative, specificity_denominator)

    def mean_present(values: torch.Tensor) -> float:
        valid = ~torch.isnan(values)
        return values[valid].mean().item() if torch.any(valid) else float("nan")

    return {
        "pixel_accuracy": (
            (true_positive.sum() / total).item() if total > 0 else float("nan")
        ),
        "mean_pixel_accuracy": mean_present(
            safe_ratio(true_positive, actual)
        ),
        "miou": mean_present(per_class_iou),
        "mean_dice": mean_present(per_class_dice),
        "mean_sensitivity": mean_present(per_class_sensitivity),
        "mean_specificity": mean_present(per_class_specificity),
        "per_class_iou": per_class_iou.tolist(),
        "per_class_dice": per_class_dice.tolist(),
        "per_class_sensitivity": per_class_sensitivity.tolist(),
        "per_class_specificity": per_class_specificity.tolist(),
        "confusion_matrix": confusion.cpu().tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    cross_entropy = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.int64, device=device
    )
    total_loss = 0.0
    total_batches = 0
    context = torch.enable_grad if training else torch.inference_mode

    with context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            ce_loss = cross_entropy(logits, targets)
            dice_loss = multiclass_dice_loss(logits, targets, num_classes)
            loss = ce_loss + dice_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training/validation loss detected.")

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            update_confusion_matrix(confusion, logits.detach(), targets, num_classes)

    if total_batches == 0:
        raise ValueError("DataLoader produced no batches.")

    results = metrics_from_confusion(confusion)
    results["loss"] = total_loss / total_batches
    return results


def update_confusion_from_prediction(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    valid = targets != ignore_index
    encoded = targets[valid] * num_classes + predictions[valid]
    confusion += torch.bincount(
        encoded, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
