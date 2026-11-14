"""Create a stratified validation split from each VOC weather training set."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from generate_annotation_rows import CLASSES, DATASETS, make_annotation_row


PROJECT_ROOT = Path(__file__).resolve().parent


def _read_ids(path: Path) -> list[str]:
    ids = path.read_text(encoding="utf-8").split()
    if not ids:
        raise ValueError(f"No image IDs found in {path}.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs found in {path}.")
    return ids


def _class_ids(row: str) -> frozenset[int]:
    class_ids = set()
    for box in row.split()[2:]:
        try:
            class_id = int(box.rsplit(",", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid annotation box in generated row: {box!r}") from exc
        if class_id < 0 or class_id >= len(CLASSES):
            raise ValueError(f"Class ID is out of range in generated row: {box!r}")
        class_ids.add(class_id)
    return frozenset(class_ids)


def _stratified_validation_ids(
    image_ids: list[str],
    labels_by_id: dict[str, frozenset[int]],
    validation_fraction: float,
    seed: int,
) -> set[str]:
    validation_count = round(len(image_ids) * validation_fraction)
    validation_count = min(max(validation_count, 1), len(image_ids) - 1)

    label_ids = {
        image_id: labels_by_id[image_id] or frozenset({-1})
        for image_id in image_ids
    }
    label_totals = Counter(
        label for labels in label_ids.values() for label in labels
    )
    target_counts = {
        label: min(
            label_totals[label],
            int(label_totals[label] * validation_fraction + 0.5),
        )
        for label in label_totals
    }
    selected_counts: Counter[int] = Counter()
    remaining = set(image_ids)
    randomizer = random.Random(seed)
    tie_breakers = {image_id: randomizer.random() for image_id in image_ids}
    selected: set[str] = set()

    def marginal_cost(image_id: str) -> float:
        return sum(
            (
                (selected_counts[label] + 1 - target_counts[label]) ** 2
                - (selected_counts[label] - target_counts[label]) ** 2
            )
            / max(target_counts[label], 1)
            for label in label_ids[image_id]
        )

    while len(selected) < validation_count:
        chosen = min(
            remaining,
            key=lambda image_id: (
                marginal_cost(image_id),
                tie_breakers[image_id],
            ),
        )
        selected.add(chosen)
        remaining.remove(chosen)
        selected_counts.update(label_ids[chosen])

    selected_ids = sorted(selected)
    remaining_ids = sorted(remaining)
    for _ in range(300_000):
        selected_index = randomizer.randrange(len(selected_ids))
        remaining_index = randomizer.randrange(len(remaining_ids))
        outgoing = selected_ids[selected_index]
        incoming = remaining_ids[remaining_index]
        changes = Counter(label_ids[incoming])
        changes.subtract(label_ids[outgoing])
        cost_change = sum(
            (
                (selected_counts[label] + change - target_counts[label]) ** 2
                - (selected_counts[label] - target_counts[label]) ** 2
            )
            / max(target_counts[label], 1)
            for label, change in changes.items()
        )
        if cost_change < -1e-12:
            selected_counts.update(changes)
            selected.remove(outgoing)
            selected.add(incoming)
            selected_ids[selected_index] = incoming
            remaining_ids[remaining_index] = outgoing

    return selected


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


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
        help="Directory for generated annotation rows.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of training images reserved for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratification (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    split_paths = {
        dataset_key: (
            dataset_root / dataset_folder / "ImageSets" / "Main" / "train.txt",
            dataset_root / dataset_folder / "ImageSets" / "Main" / "train_all.txt",
            dataset_root / dataset_folder / "ImageSets" / "Main" / "val.txt",
        )
        for dataset_key, (dataset_folder, _) in DATASETS.items()
    }
    backup_flags = [paths[1].is_file() for paths in split_paths.values()]
    if any(backup_flags) and not all(backup_flags):
        raise ValueError(
            "Some but not all train_all.txt backups exist; resolve the incomplete "
            "split before running this script."
        )
    has_backups = all(backup_flags)
    if not has_backups and any(paths[2].is_file() for paths in split_paths.values()):
        raise FileExistsError(
            "A val.txt already exists without train_all.txt backups. "
            "Refusing to overwrite an unknown existing split."
        )

    source_ids: dict[str, list[str]] = {}
    for dataset_key, (train_path, backup_path, val_path) in split_paths.items():
        source_path = backup_path if has_backups else train_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Training ID list does not exist: {source_path}")
        source_ids[dataset_key] = _read_ids(source_path)
        if val_path.is_file():
            current_train = set(_read_ids(train_path))
            current_val = set(_read_ids(val_path))
            if current_train & current_val:
                raise ValueError(f"Training and validation IDs overlap for {dataset_key}.")
            if current_train | current_val != set(source_ids[dataset_key]):
                raise ValueError(
                    f"Current train/val lists do not match {backup_path}; "
                    "refusing to overwrite the split."
                )

    reference_ids = source_ids["Clean"]
    reference_set = set(reference_ids)
    for dataset_key, image_ids in source_ids.items():
        if set(image_ids) != reference_set:
            raise ValueError(
                f"{dataset_key} training IDs do not match VOC_Clean; "
                "one shared split cannot preserve paired-image alignment."
            )

    all_rows: dict[str, dict[str, str]] = {}
    labels_by_id: dict[str, frozenset[int]] = {}
    for dataset_key in DATASETS:
        rows = {
            image_id: make_annotation_row(
                dataset_root, dataset_key, "train", image_id
            )
            for image_id in reference_ids
        }
        all_rows[dataset_key] = rows
        if dataset_key == "Clean":
            labels_by_id = {
                image_id: _class_ids(row) for image_id, row in rows.items()
            }
        else:
            for image_id, row in rows.items():
                if _class_ids(row) != labels_by_id[image_id]:
                    raise ValueError(
                        f"Class annotations differ for {image_id!r} in "
                        f"{dataset_key}; a shared stratified split is unsafe."
                    )

    validation_ids = _stratified_validation_ids(
        reference_ids, labels_by_id, args.validation_fraction, args.seed
    )
    train_ids = [image_id for image_id in reference_ids if image_id not in validation_ids]
    val_ids = [image_id for image_id in reference_ids if image_id in validation_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_key, (train_path, backup_path, val_path) in split_paths.items():
        if not has_backups:
            _write_lines(backup_path, reference_ids)
        _write_lines(train_path, train_ids)
        _write_lines(val_path, val_ids)
        _write_lines(
            output_dir / f"train_{dataset_key}.txt",
            [all_rows[dataset_key][image_id] for image_id in train_ids],
        )
        _write_lines(
            output_dir / f"val_{dataset_key}.txt",
            [all_rows[dataset_key][image_id] for image_id in val_ids],
        )

    total_by_class = Counter(
        class_id for labels in labels_by_id.values() for class_id in labels
    )
    val_by_class = Counter(
        class_id
        for image_id in val_ids
        for class_id in labels_by_id[image_id]
    )
    print(
        f"Created {len(val_ids)} validation and {len(train_ids)} training "
        f"images per weather type ({len(val_ids) / len(reference_ids):.1%} validation)."
    )
    for class_id, class_name in enumerate(CLASSES):
        print(
            f"{class_name}: validation images with class "
            f"{val_by_class[class_id]}/{total_by_class[class_id]}"
        )
    print(f"Validation ID lists: Dataset/<weather>/ImageSets/Main/val.txt")
    print(f"Annotation rows: {output_dir}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
