"""Evaluate a trained RE-HOG checkpoint.

* Saves detection images (boxes, class names, confidence scores) and the matching
  restored images for N images.
* Computes per-class AP and mAP@0.5 on the whole annotation list and prints/saves
  them as a table.

Output layout (works on Google Drive or any local disk path):
    <output-dir>/detections/   images with boxes, class name and confidence
    <output-dir>/restored/     restored (de-hazed / de-rained / de-snowed) images
    <output-dir>/metrics/      per_class_ap.csv, per_class_ap.txt, evaluation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluation import build_evaluation_loader, evaluate_map
from model_architecture import (
    REHOGModel,
    decode_predictions,
    load_compatible_model_state_dict,
)
from preprocessing import list_images, preprocess_image


_PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 190, 190), (240, 50, 230), (160, 130, 20),
]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _restore_coordinates(
    detections: torch.Tensor, transform: dict[str, float | int]
) -> torch.Tensor:
    result = detections.clone()
    scale = float(transform["scale"])
    pad_x = float(transform["pad_x"])
    pad_y = float(transform["pad_y"])
    result[:, [0, 2]] = (result[:, [0, 2]] - pad_x) / scale
    result[:, [1, 3]] = (result[:, [1, 3]] - pad_y) / scale
    result[:, [0, 2]] = result[:, [0, 2]].clamp(0, int(transform["original_width"]))
    result[:, [1, 3]] = result[:, [1, 3]].clamp(0, int(transform["original_height"]))
    return result


def _load_class_names(path: str | None, saved_names: list[str] | None, count: int) -> list[str]:
    if path:
        with open(path, encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]
    elif saved_names:
        names = saved_names
    else:
        names = [f"class_{index}" for index in range(count)]
    if len(names) != count:
        raise ValueError(f"Expected {count} class names, found {len(names)}.")
    return names


def _annotation_image_paths(annotation_file: str, image_root: str) -> list[str]:
    paths: list[str] = []
    with open(annotation_file, encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if not fields:
                continue
            image_path = Path(fields[0])
            if not image_path.is_absolute():
                image_path = Path(image_root) / image_path
            paths.append(str(image_path))
    return paths


def _select_images(args: argparse.Namespace) -> list[str]:
    if args.num_images == 0:
        return []
    if args.input:
        paths = list_images(args.input)
    elif args.annotations:
        paths = _annotation_image_paths(args.annotations, args.image_root)
    else:
        raise ValueError("Provide --annotations or --input to choose images to visualise.")
    if not paths:
        raise ValueError("No images found to visualise.")
    if args.num_images < 0 or args.num_images >= len(paths):
        return paths
    if args.random_images:
        return random.Random(args.seed).sample(paths, args.num_images)
    return paths[: args.num_images]


def _draw_detections(image: np.ndarray, detections: np.ndarray, class_names: list[str]) -> None:
    height, width = image.shape[:2]
    thickness = max(2, int(round(min(height, width) / 300)))
    font_scale = max(0.5, min(height, width) / 800)
    text_thickness = max(1, thickness - 1)
    for x1, y1, x2, y2, score, class_index in detections:
        class_id = int(class_index)
        color = _PALETTE[class_id % len(_PALETTE)]
        top_left = (int(round(x1)), int(round(y1)))
        bottom_right = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
        label = f"{class_names[class_id]} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, text_thickness)
        label_top = max(top_left[1] - text_h - baseline - 2, 0)
        cv2.rectangle(
            image,
            (top_left[0], label_top),
            (min(top_left[0] + text_w + 4, width - 1), label_top + text_h + baseline + 2),
            color,
            -1,
        )
        cv2.putText(
            image, label, (top_left[0] + 2, label_top + text_h + 1),
            _FONT, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
        )


@torch.inference_mode()
def _visualise(
    model: REHOGModel,
    image_paths: list[str],
    class_names: list[str],
    image_size: int,
    device: torch.device,
    confidence: float,
    iou: float,
    detections_dir: Path,
    restored_dir: Path,
) -> None:
    detection_total = 0
    for image_path in tqdm(image_paths, desc="Saving detections + restored", dynamic_ncols=True):
        with Image.open(image_path) as source:
            original = source.convert("RGB")
        image_tensor, transform = preprocess_image(original, (image_size, image_size))
        prediction = model(image_tensor.unsqueeze(0).to(device))
        detections = decode_predictions(
            prediction["detections"],
            model.num_classes,
            strides=model.detection_strides,
            confidence_threshold=confidence,
            iou_threshold=iou,
        )[0]
        restored = prediction["restored"][0].clamp(0, 1).float().cpu().numpy()

        original_w = int(transform["original_width"])
        original_h = int(transform["original_height"])
        pad_x = int(transform["pad_x"])
        pad_y = int(transform["pad_y"])
        scaled_w = max(1, int(round(original_w * float(transform["scale"]))))
        scaled_h = max(1, int(round(original_h * float(transform["scale"]))))
        restored = restored[:, pad_y: pad_y + scaled_h, pad_x: pad_x + scaled_w]
        restored = np.clip(restored.transpose(1, 2, 0) * 255, 0, 255).round().astype(np.uint8)
        file_name = Path(image_path).name
        Image.fromarray(restored).resize(
            (original_w, original_h), Image.Resampling.BICUBIC
        ).save(restored_dir / file_name, quality=95)

        canvas = np.asarray(original).copy()
        detections = _restore_coordinates(detections, transform).cpu().numpy()
        _draw_detections(canvas, detections, class_names)
        Image.fromarray(canvas).save(detections_dir / file_name, quality=95)
        detection_total += len(detections)
    print(
        f"Saved {len(image_paths)} detection images -> {detections_dir}\n"
        f"Saved {len(image_paths)} restored images  -> {restored_dir}\n"
        f"Total detections drawn (confidence >= {confidence}): {detection_total}"
    )


def _metrics_rows(metrics: dict, class_names: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for name in class_names:
        if name in metrics["AP@0.5"]:
            rows.append([
                name,
                f"{metrics['AP@0.5'][name] * 100:.2f}",
                f"{metrics['precision@confidence_0.5'][name] * 100:.2f}",
                f"{metrics['recall@confidence_0.5'][name] * 100:.2f}",
                f"{metrics['F1@confidence_0.5'][name] * 100:.2f}",
            ])
        else:
            rows.append([name, "n/a", "n/a", "n/a", "n/a"])
    rows.append(["mAP@0.5", f"{metrics['mAP@0.5'] * 100:.2f}", "", "", ""])
    return rows


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(
            str(cell).ljust(width) if index == 0 else str(cell).rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths))
        ) + " |"

    out = [separator, line(headers), separator]
    out.extend(line(row) for row in rows[:-1])
    out.extend([separator, line(rows[-1]), separator])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth / last.pth.")
    parser.add_argument("--annotations", help="RDMNet-format test list; used for mAP and (by default) to pick images.")
    parser.add_argument("--image-root", default=".", help="Root directory for annotation image paths.")
    parser.add_argument("--input", help="Optional image file or directory to visualise instead of the annotation images.")
    parser.add_argument("--num-images", "-n", type=int, default=10,
                        help="How many images to save with boxes + restored versions (0 = none, -1 = all).")
    parser.add_argument("--random-images", action="store_true",
                        help="Pick the N images randomly instead of the first N.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output-dir", default="results",
                        help="Any Google Drive or local path; detections/, restored/ and metrics/ are created inside.")
    parser.add_argument("--classes", help="Optional class-name file.")
    parser.add_argument("--image-size", type=int, help="Override saved square inference size.")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.25, help="Score threshold for drawn boxes.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU for drawn boxes.")
    parser.add_argument("--map-confidence", type=float, default=0.001)
    parser.add_argument("--map-nms-iou", type=float, default=0.65)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive.")
    if args.num_images != 0 and not (args.annotations or args.input):
        raise ValueError("Provide --annotations and/or --input (or use --num-images 0).")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError("Checkpoint is missing model_config; use a checkpoint saved by train.py.")
    model = REHOGModel(**model_config).to(device)
    skipped_fusion_keys = load_compatible_model_state_dict(model, checkpoint["model"])
    if skipped_fusion_keys:
        print(
            "Loaded legacy full-CTF checkpoint for the projection-add model; "
            f"skipped {len(skipped_fusion_keys)} removed fusion weights."
        )
    model.eval()
    class_names = _load_class_names(
        args.classes, checkpoint.get("class_names"), int(model_config["num_classes"])
    )
    image_size = args.image_size or int(checkpoint.get("image_size", 640))
    required_divisor = math.lcm(
        math.prod(model.restoration_downsample_factors),
        math.prod(model.detection_downsample_factors),
    )
    if image_size < required_divisor or image_size % required_divisor:
        raise ValueError(
            f"Inference image size must be at least and divisible by {required_divisor}."
        )
    epoch = checkpoint.get("epoch")
    print(
        f"Device: {device} | checkpoint: {args.checkpoint}"
        + (f" (epoch {int(epoch) + 1})" if epoch is not None else "")
    )

    output_root = Path(args.output_dir)
    detections_dir = output_root / "detections"
    restored_dir = output_root / "restored"
    metrics_dir = output_root / "metrics"
    for directory in (detections_dir, restored_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_paths = _select_images(args)
    if image_paths:
        _visualise(
            model, image_paths, class_names, image_size, device,
            args.confidence, args.iou, detections_dir, restored_dir,
        )

    if args.annotations:
        evaluation_loader = build_evaluation_loader(
            args.annotations,
            args.image_root,
            (image_size, image_size),
            model.num_classes,
            batch_size=args.eval_batch_size,
            num_workers=args.workers,
        )
        metrics = evaluate_map(
            model,
            evaluation_loader,
            class_names,
            device,
            confidence_threshold=args.map_confidence,
            nms_iou_threshold=args.map_nms_iou,
            iou_threshold=0.5,
        )
        headers = ["Class", "AP@0.5 (%)", "Precision@0.5 (%)", "Recall@0.5 (%)", "F1@0.5 (%)"]
        rows = _metrics_rows(metrics, class_names)
        table = _format_table(headers, rows)
        summary = (
            f"Evaluated images: {metrics['evaluated_images']} | "
            f"ground-truth objects: {metrics['ground_truth_objects']}\n"
            f"mAP NMS IoU {args.map_nms_iou}, score threshold {args.map_confidence}; "
            f"Precision/Recall/F1 use detections with score >= 0.5\n"
        )
        print("\n" + summary + table)
        (metrics_dir / "per_class_ap.txt").write_text(summary + table + "\n", encoding="utf-8")
        with open(metrics_dir / "per_class_ap.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        with open(metrics_dir / "evaluation.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        print(f"Metrics saved -> {metrics_dir}")


if __name__ == "__main__":
    main()
