"""Generate RDMNet-style annotation rows for the VOC weather datasets."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parent
CLASSES = ("person", "bicycle", "car", "motorbike", "bus")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DATASETS = {
    "Clean": ("VOC_Clean", "CleanImages"),
    "Haze": ("VOC_FOG", "HazyImages"),
    "Rain": ("VOC_Rain", "RainyImages"),
    "Snow": ("VOC_Snow", "SnowyImages"),
}
MIXED_DATASET_ALIASES = {
    "VOC_Clean": "Clean",
    "VOC_Haze": "Haze",
    "VOC_FOG": "Haze",
    "VOC_Rain": "Rain",
    "VOC_Snow": "Snow",
}


def _find_image(image_dir: Path, image_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No supported image found for {image_id!r} in {image_dir}.")


def _annotation_boxes(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    class_ids = {name: index for index, name in enumerate(CLASSES)}
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        difficult = int(float(obj.findtext("difficult") or "0"))
        if name not in class_ids or difficult == 1:
            continue

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Object has no bounding box in {xml_path}.")
        coordinates = [
            int(float(box.findtext(coordinate)))
            for coordinate in ("xmin", "ymin", "xmax", "ymax")
        ]
        boxes.append(",".join(str(value) for value in (*coordinates, class_ids[name])))
    return boxes


def _relative_path(path: Path, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).as_posix()


def _make_row(
    dataset_root: Path,
    dataset_key: str,
    split: str,
    image_id: str,
    boxes: list[str],
) -> str:
    dataset_folder, image_folder = DATASETS[dataset_key]
    degraded = _find_image(
        dataset_root / dataset_folder / split / image_folder, image_id
    )
    clean = _find_image(
        dataset_root / "VOC_Clean" / split / "CleanImages", image_id
    )
    fields = [_relative_path(degraded, dataset_root), _relative_path(clean, dataset_root)]
    return " ".join((*fields, *boxes))


def make_annotation_row(
    dataset_root: Path,
    dataset_key: str,
    split: str,
    image_id: str,
) -> str:
    source_split = "train" if split == "val" else split
    dataset_folder, _ = DATASETS[dataset_key]
    xml_path = (
        dataset_root
        / dataset_folder
        / source_split
        / "Annotations"
        / f"{image_id}.xml"
    )
    boxes = _annotation_boxes(xml_path)
    return _make_row(dataset_root, dataset_key, source_split, image_id, boxes)


def _write_split(
    dataset_root: Path,
    output_dir: Path,
    dataset_key: str,
    split: str,
) -> int:
    dataset_folder, _ = DATASETS[dataset_key]
    ids_path = dataset_root / dataset_folder / "ImageSets" / "Main" / f"{split}.txt"
    image_ids = ids_path.read_text(encoding="utf-8").split()
    output_path = output_dir / f"{split}_{dataset_key}.txt"
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for image_id in image_ids:
            output.write(make_annotation_row(dataset_root, dataset_key, split, image_id))
            output.write("\n")
    return len(image_ids)


def _write_mixed_train(
    dataset_root: Path,
    mixed_list: Path,
    output_dir: Path,
) -> int:
    output_path = output_dir / "train_Mixed.txt"
    count = 0
    train_ids = {
        dataset_key: set(
            (
                dataset_root
                / dataset_folder
                / "ImageSets"
                / "Main"
                / "train.txt"
            )
            .read_text(encoding="utf-8")
            .split()
        )
        for dataset_key, (dataset_folder, _) in DATASETS.items()
    }
    val_ids = {
        dataset_key: set(
            (
                dataset_root
                / dataset_folder
                / "ImageSets"
                / "Main"
                / "val.txt"
            )
            .read_text(encoding="utf-8")
            .split()
        )
        if (
            dataset_root
            / dataset_folder
            / "ImageSets"
            / "Main"
            / "val.txt"
        ).is_file()
        else set()
        for dataset_key, (dataset_folder, _) in DATASETS.items()
    }
    with mixed_list.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            source_path = PurePosixPath(fields[0].replace("\\", "/"))
            parts = source_path.parts
            if len(parts) < 4 or parts[1] != "train":
                raise ValueError(
                    f"Expected a VOC training image path on line {line_number} "
                    f"of {mixed_list}, got {fields[0]!r}."
                )
            dataset_key = MIXED_DATASET_ALIASES.get(parts[0])
            if dataset_key is None:
                raise ValueError(
                    f"Unsupported dataset {parts[0]!r} on line {line_number} "
                    f"of {mixed_list}."
                )
            expected_image_folder = DATASETS[dataset_key][1]
            if parts[2] != expected_image_folder:
                raise ValueError(
                    f"Unexpected image folder {parts[2]!r} on line {line_number} "
                    f"of {mixed_list}; expected {expected_image_folder!r}."
                )
            image_id = Path(parts[-1]).stem
            if image_id not in train_ids[dataset_key]:
                if image_id in val_ids[dataset_key]:
                    continue
                raise ValueError(
                    f"Image {image_id!r} from line {line_number} of {mixed_list} "
                    "is not listed in the VOC training split."
                )
            output.write(make_annotation_row(dataset_root, dataset_key, "train", image_id))
            output.write("\n")
            count += 1
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "Dataset",
        help="VOC dataset directory (default: this project's Dataset folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Annotation_rows",
        help="Directory for generated annotation files.",
    )
    parser.add_argument(
        "--mixed-list",
        type=Path,
        default=PROJECT_ROOT / "train_Mixed.txt",
        help="RDMNet-style mixed training list used to select mixed samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    if not args.mixed_list.is_file():
        raise FileNotFoundError(f"Mixed training list does not exist: {args.mixed_list}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in DATASETS:
        for split in ("train", "val", "test"):
            dataset_folder, _ = DATASETS[dataset_key]
            ids_path = (
                dataset_root
                / dataset_folder
                / "ImageSets"
                / "Main"
                / f"{split}.txt"
            )
            if not ids_path.is_file():
                continue
            count = _write_split(dataset_root, output_dir, dataset_key, split)
            print(f"Created {split}_{dataset_key}.txt ({count} images)")

    mixed_count = _write_mixed_train(dataset_root, args.mixed_list, output_dir)
    print(f"Created train_Mixed.txt ({mixed_count} images)")


if __name__ == "__main__":
    main()
