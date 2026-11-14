
"""Run RE-CTF detection with full cross-task fusion and display inline.

Compatible with Google Colab and Jupyter Notebook.
Does not save the prediction image.
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from IPython.display import display

from model_architecture import (
    REHOGModel,
    decode_predictions,
)
from preprocessing import preprocess_image


_PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 190, 190),
    (240, 50, 230),
    (160, 130, 20),
]


def _class_names(
    checkpoint: dict, classes_path: str | None, num_classes: int
) -> list[str]:

    if classes_path:
        with open(classes_path, encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]
    else:
        names = checkpoint.get("class_names")

        if not names:
            names = [
                f"class_{index}"
                for index in range(num_classes)
            ]

    if len(names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, found {len(names)}."
        )

    return names


def _restore_coordinates(
    detections: torch.Tensor,
    transform: dict[str, float | int]
) -> torch.Tensor:

    result = detections.clone()

    scale = float(transform["scale"])
    pad_x = float(transform["pad_x"])
    pad_y = float(transform["pad_y"])

    result[:, [0, 2]] = (
        result[:, [0, 2]] - pad_x
    ) / scale

    result[:, [1, 3]] = (
        result[:, [1, 3]] - pad_y
    ) / scale

    result[:, [0, 2]] = result[:, [0, 2]].clamp(
        0, int(transform["original_width"])
    )

    result[:, [1, 3]] = result[:, [1, 3]].clamp(
        0, int(transform["original_height"])
    )

    return result


def _draw_detections(
    image: np.ndarray,
    detections: np.ndarray,
    class_names: list[str]
) -> None:

    height, width = image.shape[:2]

    thickness = max(
        2, int(round(min(height, width) / 300))
    )

    font_scale = max(
        0.5, min(height, width) / 800
    )

    for x1, y1, x2, y2, score, class_index in detections:

        class_id = int(class_index)

        color = _PALETTE[
            class_id % len(_PALETTE)
        ]

        top_left = (
            int(round(x1)),
            int(round(y1))
        )

        bottom_right = (
            int(round(x2)),
            int(round(y2))
        )

        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            color,
            thickness
        )

        label = (
            f"{class_names[class_id]} {score:.2f}"
        )

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )
        )

        label_top = max(
            top_left[1] - text_height - baseline - 2,
            0
        )

        cv2.rectangle(
            image,
            (
                top_left[0],
                label_top
            ),
            (
                min(top_left[0] + text_width + 4, width - 1),
                label_top + text_height + baseline + 2
            ),
            color,
            -1
        )

        cv2.putText(
            image,
            label,
            (
                top_left[0] + 2,
                label_top + text_height + 1
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA
        )


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:

    # --------------------------------------------------
    # 1. Device and checkpoint loading
    # --------------------------------------------------

    device = torch.device(args.device)

    print(f"Using device: {device}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    if (
        not isinstance(checkpoint, dict)
        or "model_config" not in checkpoint
    ):
        raise ValueError(
            "Checkpoint must contain model_config and model state; "
            "use a checkpoint saved by train.py."
        )

    if "model" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain model weights."
        )

    # --------------------------------------------------
    # 2. Initialize RE-CTF model
    # --------------------------------------------------

    model_config = checkpoint["model_config"]

    model = REHOGModel(**model_config).to(device)

    try:
        model.load_state_dict(
            checkpoint["model"],
            strict=True
        )

    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights are incompatible with the current "
            "model_architecture.py. Use a checkpoint trained with "
            "this same architecture and model configuration."
        ) from exc

    model.eval()

    print(
        "Inference architecture: full RE-CTF cross-task fusion."
    )

    # --------------------------------------------------
    # 3. Class names and image size
    # --------------------------------------------------

    class_names = _class_names(
        checkpoint,
        args.classes,
        int(model_config["num_classes"])
    )

    image_size = (
        args.image_size
        or int(checkpoint.get("image_size", 640))
    )

    required_divisor = math.lcm(
        math.prod(model.restoration_downsample_factors),
        math.prod(model.detection_downsample_factors),
    )

    if (
        image_size < required_divisor
        or image_size % required_divisor
    ):
        raise ValueError(
            f"Inference image size must be at least and "
            f"divisible by {required_divisor}."
        )

    # --------------------------------------------------
    # 4. Load and preprocess input image
    # --------------------------------------------------

    with PILImage.open(args.image) as source:
        original = source.convert("RGB")

    image_tensor, transform = preprocess_image(
        original,
        (image_size, image_size)
    )

    image_tensor = image_tensor.unsqueeze(0).to(device)

    # --------------------------------------------------
    # 5. Model inference
    # --------------------------------------------------

    outputs = model(image_tensor)

    # --------------------------------------------------
    # 6. Decode predictions
    # --------------------------------------------------

    detections = decode_predictions(
        outputs["detections"],
        model.num_classes,
        strides=model.detection_strides,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
    )[0]

    # --------------------------------------------------
    # 7. Restore original image coordinates
    # --------------------------------------------------

    detections = _restore_coordinates(
        detections,
        transform
    ).cpu().numpy()

    # --------------------------------------------------
    # 8. Draw bounding boxes
    # --------------------------------------------------

    canvas = np.asarray(original).copy()

    _draw_detections(
        canvas,
        detections,
        class_names
    )

    # --------------------------------------------------
    # 9. Display predictions inline in Colab
    # --------------------------------------------------

    print(f"Found {len(detections)} detections.")

    if len(detections) > 0:
        for x1, y1, x2, y2, score, class_index in detections:
            print(
                f"{class_names[int(class_index)]}: "
                f"Confidence = {score:.4f}"
            )
    else:
        print("No detections found.")

    display(PILImage.fromarray(canvas))


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained RE-CTF .pth checkpoint."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to one input image."
    )

    parser.add_argument(
        "--classes",
        help="Optional text file with one class name per line."
    )

    parser.add_argument(
        "--image-size",
        type=int,
        help="Override the saved inference size."
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    predict(args)


if __name__ == "__main__":
    main()
