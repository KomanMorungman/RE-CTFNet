"""VOC-style detection evaluation using the RDMNet mAP@0.5 protocol."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_architecture import REHOGModel, decode_predictions
from preprocessing import parse_annotation_line, preprocess_image


class DetectionEvaluationDataset(Dataset[dict[str, Any]]):
    """Annotation-backed images; clean restoration targets are not required."""

    def __init__(
        self,
        annotation_file: str | os.PathLike[str],
        image_root: str | os.PathLike[str],
        image_size: tuple[int, int],
        num_classes: int,
    ) -> None:
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.num_classes = num_classes
        with open(annotation_file, encoding="utf-8") as file:
            self.lines = [line.strip() for line in file if line.strip()]
        if not self.lines:
            raise ValueError(
                f"No evaluation annotations found in {annotation_file}.")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_field, _, boxes = parse_annotation_line(self.lines[index])
        image_path = Path(image_field)
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Evaluation image does not exist: {image_path}")
        if len(boxes):
            classes = boxes[:, 4]
            if (
                np.any(classes < 0)
                or not np.all(classes == classes.astype(np.int64))
                or np.any(classes >= self.num_classes)
            ):
                raise ValueError(
                    f"Invalid class ID in evaluation annotation for {image_path}.")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        tensor, transform = preprocess_image(image, self.image_size)
        return {
            "image": tensor,
            "boxes": torch.from_numpy(boxes.copy()),
            "transform": transform,
            "path": str(image_path),
        }


def _evaluation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "transform": [item["transform"] for item in batch],
        "path": [item["path"] for item in batch],
    }


def build_evaluation_loader(
    annotation_file: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    image_size: tuple[int, int],
    num_classes: int,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be positive.")
    dataset = DetectionEvaluationDataset(
        annotation_file, image_root, image_size, num_classes
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_evaluation_collate,
    )


def _undo_letterbox(detections: Tensor, transform: dict[str, float | int]) -> Tensor:
    boxes = detections[:, :4].clone()
    scale = float(transform["scale"])
    pad_x = float(transform["pad_x"])
    pad_y = float(transform["pad_y"])
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
        0, int(transform["original_width"]))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
        0, int(transform["original_height"]))
    return boxes


def _inclusive_iou(box: Tensor, boxes: Tensor) -> Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left + 1).clamp(min=0).prod(dim=1)
    area_box = (box[2:] - box[:2] + 1).clamp(min=0).prod()
    area_boxes = (boxes[:, 2:] - boxes[:, :2] + 1).clamp(min=0).prod(dim=1)
    return intersection / (area_box + area_boxes - intersection).clamp(min=1e-8)


def _voc_average_precision(recall: list[float], precision: list[float]) -> float:
    recall_curve = [0.0, *recall, 1.0]
    precision_curve = [0.0, *precision, 0.0]
    for index in range(len(precision_curve) - 2, -1, -1):
        precision_curve[index] = max(
            precision_curve[index], precision_curve[index + 1])
    return sum(
        (recall_curve[index] - recall_curve[index - 1]) *
        precision_curve[index]
        for index in range(1, len(recall_curve))
        if recall_curve[index] != recall_curve[index - 1]
    )


def evaluate_map(
    model: REHOGModel,
    loader: DataLoader[dict[str, Any]],
    class_names: Sequence[str],
    device: torch.device,
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = 0.65,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute VOC integral AP/mAP, matching RDMNet's default IoU threshold."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if not 0 <= nms_iou_threshold <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("IoU thresholds must be between 0 and 1.")
    class_count = len(class_names)
    if class_count != model.num_classes:
        raise ValueError("class_names count must match model.num_classes.")

    ground_truth: dict[int, list[Tensor]] = {
        class_id: [] for class_id in range(class_count)}
    detections: dict[int, list[tuple[float, int, Tensor]]] = {
        class_id: [] for class_id in range(class_count)
    }
    image_index = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc="mAP eval", dynamic_ncols=True, mininterval=1.0):
                images = batch["image"].to(device, non_blocking=True)
                outputs = model(images)
                batch_detections = decode_predictions(
                    outputs["detections"],
                    class_count,
                    strides=model.detection_strides,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=nms_iou_threshold,
                )
                for batch_index, predicted in enumerate(batch_detections):
                    target = batch["boxes"][batch_index]
                    target_classes = target[:, 4].long()
                    for class_id in range(class_count):
                        ground_truth[class_id].append(
                            target[target_classes == class_id, :4].to(device)
                        )

                    if predicted.numel():
                        original_boxes = _undo_letterbox(
                            predicted, batch["transform"][batch_index]
                        )
                        for row, box in zip(predicted, original_boxes):
                            class_id = int(row[5].item())
                            detections[class_id].append(
                                (float(row[4].item()), image_index, box)
                            )
                    image_index += 1
    finally:
        model.train(was_training)

    per_class_ap: dict[str, float] = {}
    precision_at_05: dict[str, float] = {}
    recall_at_05: dict[str, float] = {}
    f1_at_05: dict[str, float] = {}
    evaluated_classes = 0
    for class_id, class_name in enumerate(class_names):
        total_ground_truth = sum(len(boxes)
                                 for boxes in ground_truth[class_id])
        if total_ground_truth == 0:
            continue
        evaluated_classes += 1
        ordered = sorted(detections[class_id],
                         key=lambda item: item[0], reverse=True)
        used = [torch.zeros(len(boxes), dtype=torch.bool, device=device)
                for boxes in ground_truth[class_id]]
        true_positive: list[float] = []
        false_positive: list[float] = []
        scores: list[float] = []
        for score, index, box in ordered:
            scores.append(score)
            image_targets = ground_truth[class_id][index]
            if not len(image_targets):
                true_positive.append(0.0)
                false_positive.append(1.0)
                continue
            overlaps = _inclusive_iou(box, image_targets)
            best_index = int(overlaps.argmax().item())
            if float(overlaps[best_index].item()) >= iou_threshold and not used[index][best_index]:
                true_positive.append(1.0)
                false_positive.append(0.0)
                used[index][best_index] = True
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)

        tp_cumulative = np.cumsum(true_positive)
        fp_cumulative = np.cumsum(false_positive)
        recall_curve = (tp_cumulative / total_ground_truth).tolist()
        precision_curve = (
            tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
        ).tolist()
        per_class_ap[class_name] = _voc_average_precision(
            recall_curve, precision_curve)

        above_threshold = [index for index,
                           score in enumerate(scores) if score >= 0.5]
        if above_threshold:
            last = above_threshold[-1]
            precision_at_05[class_name] = float(precision_curve[last])
            recall_at_05[class_name] = float(recall_curve[last])
        else:
            precision_at_05[class_name] = 0.0
            recall_at_05[class_name] = 0.0
        precision_value = precision_at_05[class_name]
        recall_value = recall_at_05[class_name]
        f1_at_05[class_name] = (
            2 * precision_value * recall_value /
            (precision_value + recall_value)
            if precision_value + recall_value
            else 0.0
        )

    if evaluated_classes == 0:
        raise ValueError(
            "No ground-truth objects for the configured classes were found.")
    return {
        "mAP@0.5": float(sum(per_class_ap.values()) / evaluated_classes),
        "AP@0.5": per_class_ap,
        "precision@confidence_0.5": precision_at_05,
        "recall@confidence_0.5": recall_at_05,
        "F1@confidence_0.5": f1_at_05,
        "evaluated_images": image_index,
        "ground_truth_objects": sum(
            len(boxes)
            for class_boxes in ground_truth.values()
            for boxes in class_boxes
        ),
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "iou_threshold": iou_threshold,
    }
