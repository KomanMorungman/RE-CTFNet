"""Paired image preprocessing and RDMNet-style detection augmentation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _resolve_path(path: str, root: str | os.PathLike[str] | None) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(root or ".") / candidate)


def parse_annotation_line(line: str) -> tuple[str, str | None, np.ndarray]:
    """Parse ``image [clean_image] x1,y1,x2,y2,class ...`` annotation rows."""
    fields = line.strip().split()
    if not fields:
        raise ValueError("Annotation line is empty.")

    image_path = fields[0]
    clean_path: str | None = None
    annotation_fields = fields[1:]
    if annotation_fields and "," not in annotation_fields[0]:
        clean_path = annotation_fields.pop(0)

    boxes: list[list[float]] = []
    for field in annotation_fields:
        values = field.split(",")
        if len(values) != 5:
            raise ValueError(
                f"Invalid box {field!r}; expected x1,y1,x2,y2,class_id."
            )
        try:
            x1, y1, x2, y2 = (float(value) for value in values[:4])
            class_id = float(values[4])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric box values: {field!r}.") from exc
        boxes.append([x1, y1, x2, y2, class_id])

    return image_path, clean_path, np.asarray(boxes, dtype=np.float32).reshape(-1, 5)


def _read_rgb(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _letterbox(
    image: Image.Image, size: tuple[int, int]
) -> tuple[Image.Image, float, int, int]:
    target_h, target_w = size
    width, height = image.size
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def preprocess_image(
    image: Image.Image, size: tuple[int, int]
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Letterbox an RGB image and return its NCHW-ready tensor and inverse map."""
    original_w, original_h = image.size
    boxed, scale, pad_x, pad_y = _letterbox(image, size)
    array = np.asarray(boxed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor, {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "original_width": original_w,
        "original_height": original_h,
    }


def _apply_hsv(image: Image.Image, gains: tuple[float, float, float]) -> Image.Image:
    array = np.asarray(image, dtype=np.uint8)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    hue_gain, saturation_gain, value_gain = gains
    dtype = array.dtype
    lookup = np.arange(256, dtype=np.float32)
    hue_lut = ((lookup * hue_gain) % 180).astype(dtype)
    sat_lut = np.clip(lookup * saturation_gain, 0, 255).astype(dtype)
    val_lut = np.clip(lookup * value_gain, 0, 255).astype(dtype)
    adjusted = cv2.merge(
        (cv2.LUT(hue, hue_lut), cv2.LUT(saturation, sat_lut), cv2.LUT(value, val_lut))
    )
    return Image.fromarray(cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB))


def _training_transform(
    degraded: Image.Image,
    clean: Image.Image,
    boxes: np.ndarray,
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    target_h, target_w = size
    original_w, original_h = degraded.size

    aspect = (
        original_w
        / original_h
        * random.uniform(0.7, 1.3)
        / random.uniform(0.7, 1.3)
    )
    scale = random.uniform(0.25, 2.0)
    if aspect < 1:
        resized_h = max(1, int(scale * target_h))
        resized_w = max(1, int(resized_h * aspect))
    else:
        resized_w = max(1, int(scale * target_w))
        resized_h = max(1, int(resized_w / aspect))

    degraded = degraded.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    clean = clean.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    offset_x = int(random.uniform(0, target_w - resized_w))
    offset_y = int(random.uniform(0, target_h - resized_h))
    degraded_canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    clean_canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    degraded_canvas.paste(degraded, (offset_x, offset_y))
    clean_canvas.paste(clean, (offset_x, offset_y))

    transformed = boxes.copy()
    if len(transformed):
        transformed[:, [0, 2]] *= resized_w / original_w
        transformed[:, [1, 3]] *= resized_h / original_h
        transformed[:, [0, 2]] += offset_x
        transformed[:, [1, 3]] += offset_y

    if random.random() < 0.5:
        degraded_canvas = degraded_canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        clean_canvas = clean_canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if len(transformed):
            old_x1 = transformed[:, 0].copy()
            transformed[:, 0] = target_w - transformed[:, 2]
            transformed[:, 2] = target_w - old_x1

    gains = tuple(
        float(value) for value in (1 + np.random.uniform(-1, 1, 3) * [0.1, 0.7, 0.4])
    )
    degraded_canvas = _apply_hsv(degraded_canvas, gains)
    clean_canvas = _apply_hsv(clean_canvas, gains)

    if len(transformed):
        transformed[:, [0, 2]] = transformed[:, [0, 2]].clip(0, target_w)
        transformed[:, [1, 3]] = transformed[:, [1, 3]].clip(0, target_h)
        valid = (transformed[:, 2] - transformed[:, 0] > 1) & (
            transformed[:, 3] - transformed[:, 1] > 1
        )
        transformed = transformed[valid]
    return degraded_canvas, clean_canvas, transformed


def _evaluation_transform(
    degraded: Image.Image,
    clean: Image.Image,
    boxes: np.ndarray,
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    original_w, original_h = degraded.size
    degraded_boxed, scale, pad_x, pad_y = _letterbox(degraded, size)
    clean_boxed, _, _, _ = _letterbox(clean, size)
    transformed = boxes.copy()
    if len(transformed):
        transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_x
        transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_y
        transformed[:, [0, 2]] = transformed[:, [0, 2]].clip(0, size[1])
        transformed[:, [1, 3]] = transformed[:, [1, 3]].clip(0, size[0])
        valid = (transformed[:, 2] > transformed[:, 0]) & (
            transformed[:, 3] > transformed[:, 1]
        )
        transformed = transformed[valid]
    if clean.size != (original_w, original_h):
        raise ValueError("Paired degraded and clean images must have identical dimensions.")
    return degraded_boxed, clean_boxed, transformed


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class PairedDetectionDataset(Dataset[dict[str, Any]]):
    """Dataset for RDMNet-style annotation rows and paired clean images."""

    def __init__(
        self,
        annotation_file: str | os.PathLike[str],
        image_root: str | os.PathLike[str],
        clean_root: str | os.PathLike[str] | None,
        image_size: tuple[int, int] = (640, 640),
        training: bool = True,
        num_classes: int | None = None,
    ) -> None:
        self.image_root = str(image_root)
        self.clean_root = str(clean_root) if clean_root is not None else None
        self.image_size = image_size
        self.training = training
        self.num_classes = num_classes
        self.augment = True  # train.py switches this off for the final no-augmentation epochs
        with open(annotation_file, encoding="utf-8") as file:
            self.lines = [line.strip() for line in file if line.strip()]
        if not self.lines:
            raise ValueError(f"No annotations found in {annotation_file}.")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_field, clean_field, boxes = parse_annotation_line(self.lines[index])
        image_path = _resolve_path(image_field, self.image_root)
        if clean_field is not None:
            clean_path = _resolve_path(clean_field, self.image_root)
        elif self.clean_root is not None:
            clean_path = str(Path(self.clean_root) / Path(image_field).name)
        else:
            raise ValueError(
                "Each annotation must include a clean-image path or the dataset "
                "must be constructed with clean_root."
            )

        degraded = _read_rgb(image_path)
        clean = _read_rgb(clean_path)
        if degraded.size != clean.size:
            raise ValueError(
                f"Paired images must have matching dimensions: {image_path} and {clean_path}."
            )
        if len(boxes):
            if np.any(boxes[:, 4] < 0) or not np.all(boxes[:, 4] == boxes[:, 4].astype(int)):
                raise ValueError(f"Class IDs must be non-negative integers in {image_path}.")
            if self.num_classes is not None and np.any(boxes[:, 4] >= self.num_classes):
                raise ValueError(f"Class ID is out of range in {image_path}.")

        transform = (
            _training_transform if (self.training and self.augment) else _evaluation_transform
        )
        degraded, clean, boxes = transform(degraded, clean, boxes, self.image_size)
        if len(boxes):
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            boxes[:, :2] += np.stack((widths, heights), axis=1) / 2
            boxes[:, 2:4] = np.stack((widths, heights), axis=1)

        return {
            "image": _to_tensor(degraded),
            "clean": _to_tensor(clean),
            "boxes": torch.from_numpy(boxes.copy()),
            "path": image_path,
        }


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "clean": torch.stack([item["clean"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "path": [item["path"] for item in batch],
    }


def list_images(path: str | os.PathLike[str]) -> list[str]:
    target = Path(path)
    if target.is_file():
        if target.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {target}")
        return [str(target)]
    if not target.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    return [
        str(image)
        for image in sorted(target.iterdir())
        if image.is_file() and image.suffix.lower() in _IMAGE_EXTENSIONS
    ]
