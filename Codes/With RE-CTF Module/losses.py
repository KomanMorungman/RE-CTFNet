"""Detection and restoration losses used to train RE-HOG."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _pairwise_iou_xywh(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.maximum(
        boxes_a[:, None, :2] - boxes_a[:, None, 2:] / 2,
        boxes_b[None, :, :2] - boxes_b[None, :, 2:] / 2,
    )
    bottom_right = torch.minimum(
        boxes_a[:, None, :2] + boxes_a[:, None, 2:] / 2,
        boxes_b[None, :, :2] + boxes_b[None, :, 2:] / 2,
    )
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area_a = boxes_a[:, 2:].clamp(min=0).prod(dim=-1)
    area_b = boxes_b[:, 2:].clamp(min=0).prod(dim=-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-8)


class YOLODetectionLoss(nn.Module):
    """RDMNet's YOLOX-style IoU, focal objectness, and dynamic-k class loss."""

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int] = (8, 16, 32),
        center_radius: float = 2.5,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.strides = tuple(strides)
        self.center_radius = center_radius

    def _assign(
        self,
        prediction: Tensor,
        labels: Tensor,
        grid_centers: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if labels.numel() == 0:
            empty_indices = prediction.new_zeros((0,), dtype=torch.long)
            matched_indices = prediction.new_full(
                (prediction.shape[0],), -1, dtype=torch.long)
            foreground = prediction.new_zeros(
                (prediction.shape[0],), dtype=torch.bool)
            return empty_indices, matched_indices, prediction.new_zeros((0,)), foreground

        gt_boxes = labels[:, :4]
        gt_classes = labels[:, 4].long()
        left_top = gt_boxes[:, None, :2] - gt_boxes[:, None, 2:] / 2
        right_bottom = gt_boxes[:, None, :2] + gt_boxes[:, None, 2:] / 2
        centers = grid_centers.unsqueeze(0)
        in_boxes = (
            (centers[..., 0] > left_top[..., 0])
            & (centers[..., 0] < right_bottom[..., 0])
            & (centers[..., 1] > left_top[..., 1])
            & (centers[..., 1] < right_bottom[..., 1])
        )
        center_delta = (centers - gt_boxes[:, None, :2]).abs()
        in_centers = (
            (center_delta[..., 0] < self.center_radius * strides[None, :])
            & (center_delta[..., 1] < self.center_radius * strides[None, :])
        )
        candidate = (in_boxes | in_centers).any(dim=0)
        if not candidate.any():
            distance = torch.cdist(gt_boxes[:, :2], grid_centers)
            candidate[distance.min(dim=0).values.argmin()] = True

        candidate_indices = torch.where(candidate)[0]
        candidate_predictions = prediction[candidate]
        pairwise_iou = _pairwise_iou_xywh(
            gt_boxes, candidate_predictions[:, :4])
        class_logits = candidate_predictions[:, 5: 5 + self.num_classes]
        object_logits = candidate_predictions[:, 4:5]
        one_hot = F.one_hot(gt_classes, self.num_classes).float()
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            probabilities = (
                class_logits.float().sigmoid() * object_logits.float().sigmoid()
            ).sqrt()
            class_cost = F.binary_cross_entropy(
                probabilities.unsqueeze(0).expand(len(labels), -1, -1),
                one_hot[:, None, :].expand(-1, len(candidate_indices), -1),
                reduction="none",
            ).sum(dim=-1)
        box_cost = -torch.log(pairwise_iou.clamp(min=1e-8))
        valid_pair = (in_boxes | in_centers)[:, candidate]
        cost = class_cost + 3.0 * box_cost + (~valid_pair).float() * 100000.0

        matching = torch.zeros_like(cost)
        for gt_index in range(len(labels)):
            top_ious = pairwise_iou[gt_index].topk(
                min(10, pairwise_iou.shape[1])).values
            dynamic_k = max(int(top_ious.sum().item()), 1)
            chosen = torch.topk(cost[gt_index], min(
                dynamic_k, cost.shape[1]), largest=False).indices
            matching[gt_index, chosen] = 1.0
        multiple = matching.sum(dim=0) > 1
        if multiple.any():
            best_gt = cost[:, multiple].argmin(dim=0)
            matching[:, multiple] = 0.0
            matching[best_gt, torch.where(multiple)[0]] = 1.0

        foreground_candidate = matching.sum(dim=0) > 0
        foreground = torch.zeros_like(candidate)
        foreground[candidate_indices[foreground_candidate]] = True
        matched_gt = matching[:, foreground_candidate].argmax(dim=0)
        matched_classes = gt_classes[matched_gt]
        matched_ious = (
            matching * pairwise_iou).sum(dim=0)[foreground_candidate]
        matched_indices = torch.full_like(foreground, -1, dtype=torch.long)
        matched_indices[candidate_indices[foreground_candidate]] = matched_gt
        return matched_classes, matched_indices, matched_ious, foreground

    def forward(self, outputs: Sequence[Tensor], targets: Sequence[Tensor]) -> Tensor:
        # Run the whole loss in fp32: fp16 loses sub-pixel precision on box coordinates.
        outputs = [output.float() for output in outputs]
        with torch.autocast(device_type=outputs[0].device.type, enabled=False):
            return self._forward(outputs, targets)

    def _forward(self, outputs: Sequence[Tensor], targets: Sequence[Tensor]) -> Tensor:
        if len(outputs) != len(self.strides):
            raise ValueError(
                "Detection output count must match the configured stride count.")
        batch_size = outputs[0].shape[0]
        decoded_levels: list[Tensor] = []
        centers_levels: list[Tensor] = []
        strides_levels: list[Tensor] = []
        for output, stride in zip(outputs, self.strides):
            _, _, height, width = output.shape
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, device=output.device, dtype=output.dtype),
                torch.arange(width, device=output.device, dtype=output.dtype),
                indexing="ij",
            )
            grid = torch.stack((x_grid, y_grid), dim=-1).reshape(1, -1, 2)
            prediction = output.flatten(2).transpose(1, 2)
            decoded = torch.cat(
                (
                    (prediction[..., :2] + grid) * stride,
                    prediction[..., 2:4].clamp(max=10).exp() * stride,
                    prediction[..., 4:],
                ),
                dim=-1,
            )
            decoded_levels.append(decoded)
            centers_levels.append(
                (grid + 0.5).expand(batch_size, -1, -1).reshape(batch_size, -1, 2) * stride)
            strides_levels.append(output.new_full((height * width,), stride))

        predictions = torch.cat(decoded_levels, dim=1)
        grid_centers = torch.cat(centers_levels, dim=1)
        expanded_strides = torch.cat(strides_levels, dim=0)
        foreground_masks: list[Tensor] = []
        class_targets: list[Tensor] = []
        box_targets: list[Tensor] = []
        iou_targets: list[Tensor] = []
        for batch_index in range(batch_size):
            labels = targets[batch_index].to(
                device=predictions.device, dtype=predictions.dtype)
            if labels.numel() and (
                labels.shape[-1] != 5
                or torch.any(labels[:, 4] < 0)
                or torch.any(labels[:, 4] >= self.num_classes)
            ):
                raise ValueError(
                    "Targets must contain cx,cy,w,h,class_id with valid class IDs.")
            classes, matched, matched_iou, foreground = self._assign(
                predictions[batch_index], labels, grid_centers[batch_index], expanded_strides
            )
            foreground_masks.append(foreground)
            if foreground.any():
                class_targets.append(
                    F.one_hot(classes, self.num_classes).to(predictions.dtype)
                    * matched_iou.unsqueeze(-1)
                )
                box_targets.append(labels[matched[foreground], :4])
                iou_targets.append(matched_iou)

        foreground = torch.stack(foreground_masks)
        object_targets = foreground.unsqueeze(-1).to(predictions.dtype)
        object_logits = torch.cat(
            [output[:, 4:5].flatten(2).transpose(1, 2) for output in outputs], dim=1
        )
        object_probabilities = object_logits.sigmoid()
        positive_loss = (
            torch.log(object_probabilities.clamp(min=1e-5))
            * (1 - object_probabilities).square()
            * object_targets
            * 0.75
        )
        negative_loss = (
            torch.log((1 - object_probabilities).clamp(min=1e-5))
            * object_probabilities.square()
            * (1 - object_targets)
            * 0.25
        )
        object_loss = -(positive_loss + negative_loss).sum()

        num_foreground = max(int(foreground.sum().item()), 1)
        if foreground.any():
            foreground_predictions = predictions[foreground]
            target_boxes = torch.cat(box_targets)
            target_classes = torch.cat(class_targets)
            target_ious = torch.cat(iou_targets)
            pred_xyxy = torch.cat(
                (
                    foreground_predictions[:, :2] -
                    foreground_predictions[:, 2:4] / 2,
                    foreground_predictions[:, :2] +
                    foreground_predictions[:, 2:4] / 2,
                ),
                dim=1,
            )
            target_xyxy = torch.cat(
                (target_boxes[:, :2] - target_boxes[:, 2:4] / 2,
                 target_boxes[:, :2] + target_boxes[:, 2:4] / 2),
                dim=1,
            )
            intersection = (
                torch.minimum(pred_xyxy[:, 2:], target_xyxy[:, 2:])
                - torch.maximum(pred_xyxy[:, :2], target_xyxy[:, :2])
            ).clamp(min=0).prod(dim=1)
            pred_area = (pred_xyxy[:, 2:] - pred_xyxy[:,
                         :2]).clamp(min=0).prod(dim=1)
            target_area = (
                target_xyxy[:, 2:] - target_xyxy[:, :2]).clamp(min=0).prod(dim=1)
            iou = intersection / \
                (pred_area + target_area - intersection + 1e-8)
            iou_loss = (1 - iou.square()).sum()
            class_logits = torch.cat(
                [output[:, 5:].flatten(2).transpose(1, 2) for output in outputs], dim=1
            )[foreground]
            class_loss = F.binary_cross_entropy_with_logits(
                class_logits, target_classes, reduction="sum"
            )
            del target_ious
        else:
            iou_loss = predictions.sum() * 0
            class_loss = predictions.sum() * 0
        return (5.0 * iou_loss + object_loss + class_loss) / num_foreground


class HOGLayer(nn.Module):
    def __init__(self, bins: int = 9, cell_size: int = 8) -> None:
        super().__init__()
        self.bins = bins
        self.cell_size = cell_size
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            .view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            .view(1, 1, 3, 3),
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.shape[1] == 3:
            image = (
                0.299 * image[:, 0:1]
                + 0.587 * image[:, 1:2]
                + 0.114 * image[:, 2:3]
            )
        height = image.shape[-2] // self.cell_size * self.cell_size
        width = image.shape[-1] // self.cell_size * self.cell_size
        if height == 0 or width == 0:
            raise ValueError("HOG input dimensions must be at least one cell.")
        image = image[..., :height, :width]
        gx = F.conv2d(image, self.sobel_x, padding=1)
        gy = F.conv2d(image, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        orientation = torch.atan2(gy, gx).remainder(math.pi)
        bin_index = (orientation * (self.bins / math.pi)
                     ).long().remainder(self.bins)
        histogram = image.new_zeros((image.shape[0], self.bins, height, width))
        histogram.scatter_add_(1, bin_index, magnitude)
        histogram = F.avg_pool2d(histogram, self.cell_size, self.cell_size)
        return F.normalize(histogram, p=2, dim=1, eps=1e-6)


class HOGLoss(nn.Module):
    def __init__(self, bins: int = 9, cell_size: int = 8) -> None:
        super().__init__()
        self.hog = HOGLayer(bins=bins, cell_size=cell_size)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Restoration prediction and target must have identical shapes.")
        return F.mse_loss(self.hog(prediction), self.hog(target))


class RestorationLoss(nn.Module):
    """HOGformer restoration objective: pixel L1 + Pearson + HOG descriptor loss."""

    def __init__(self) -> None:
        super().__init__()
        self.hog_loss = HOGLoss()

    def forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        # fp32 and no autocast: HOG/Pearson terms are numerically fragile in fp16.
        prediction = prediction.float()
        target = target.float()
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return self._forward(prediction, target)

    def _forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                "Restoration prediction and target must have identical shapes.")
        pixel = F.l1_loss(prediction, target)
        prediction_flat = prediction.flatten(1)
        target_flat = target.flatten(1)
        prediction_centered = prediction_flat - \
            prediction_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        numerator = (prediction_centered * target_centered).sum(dim=1)
        denominator = (
            prediction_centered.square().sum(dim=1).sqrt()
            * target_centered.square().sum(dim=1).sqrt()
        ).clamp(min=1e-8)
        pearson = ((1 - numerator / denominator) / 2).mean()
        hog = self.hog_loss(prediction, target)
        return {"total": pixel + pearson + hog, "pixel": pixel, "pearson": pearson, "hog": hog}


def restoration_loss_weight(
    epoch: int,
    decay_interval: int = 20,
    weights: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6),
) -> float:
    """RDMNet schedule: reduce restoration weight by 0.1 every 20 epochs."""
    if epoch < 0 or decay_interval < 1 or not weights:
        raise ValueError(
            "epoch, decay_interval, and weights must define a valid schedule.")
    schedule_index = min(epoch // decay_interval, len(weights) - 1)
    return float(weights[schedule_index])
