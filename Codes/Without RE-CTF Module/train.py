"""Train RE-HOG from paired images and RDMNet-format object annotations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import build_evaluation_loader, evaluate_map
from losses import RestorationLoss, YOLODetectionLoss, restoration_loss_weight
from model_architecture import REHOGModel, load_compatible_model_state_dict
from preprocessing import PairedDetectionDataset, detection_collate


def _read_class_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {path}.")
    return names


def _move_targets(targets: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [target.to(device, non_blocking=True) for target in targets]


def _parse_integer_list(value: str, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(
            f"{option} must be a comma-separated list of integers.") from exc
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{option} must contain positive integers.")
    return values


def _rdmnet_epoch_learning_rate(
    epoch: int, total_epochs: int, learning_rate: float, minimum_learning_rate: float
) -> float:
    warmup_epochs = min(max(0.1 * total_epochs, 1), 3)
    no_augmentation_epochs = min(max(0.3 * total_epochs, 1), 15)
    warmup_start = max(0.1 * learning_rate, 1e-6)
    if epoch <= warmup_epochs:
        return (learning_rate - warmup_start) * (epoch / warmup_epochs) ** 2 + warmup_start
    if epoch >= total_epochs - no_augmentation_epochs:
        return minimum_learning_rate
    cosine_progress = (epoch - warmup_epochs) / (
        total_epochs - warmup_epochs - no_augmentation_epochs
    )
    return minimum_learning_rate + 0.5 * (learning_rate - minimum_learning_rate) * (
        1 + math.cos(math.pi * cosine_progress)
    )


def _build_rdmnet_optimizer(
    model: nn.Module,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    momentum: float,  # unused, kept so the call site doesn't change
) -> tuple[torch.optim.AdamW, float, float]:
    base_lr = learning_rate
    min_lr = base_lr * 0.01
    buckets: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_restoration = name.startswith(
            ("restoration_encoder", "restoration_decoder")
        ) or "fusion_" in name
        lr_scale = 0.6 if is_restoration else 1.0
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        buckets.setdefault((lr_scale, no_decay), []).append(parameter)
    groups = [
        {
            "params": params,
            "lr": base_lr * lr_scale,
            "lr_scale": lr_scale,
            "weight_decay": 0.0 if no_decay else weight_decay,
        }
        for (lr_scale, no_decay), params in buckets.items()
    ]
    optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999))
    return optimizer, base_lr, min_lr


_YOLOX_S_DETECTION_PREFIX_MAP = (
    ("backbone.backbone.stem.conv", "detection_encoder.focus.conv"),
    ("backbone.backbone.dark2.0", "detection_encoder.dark2_down"),
    ("backbone.backbone.dark2.1.conv1", "detection_encoder.dark2.branch1"),
    ("backbone.backbone.dark2.1.conv2", "detection_encoder.dark2.branch2"),
    ("backbone.backbone.dark2.1.conv3", "detection_encoder.dark2.merge"),
    ("backbone.backbone.dark2.1.m", "detection_encoder.dark2.blocks"),
    ("backbone.backbone.dark3.0", "detection_encoder.dark3_down"),
    ("backbone.backbone.dark3.1.conv1", "detection_encoder.dark3.branch1"),
    ("backbone.backbone.dark3.1.conv2", "detection_encoder.dark3.branch2"),
    ("backbone.backbone.dark3.1.conv3", "detection_encoder.dark3.merge"),
    ("backbone.backbone.dark3.1.m", "detection_encoder.dark3.blocks"),
    ("backbone.backbone.dark4.0", "detection_encoder.dark4_down"),
    ("backbone.backbone.dark4.1.conv1", "detection_encoder.dark4.branch1"),
    ("backbone.backbone.dark4.1.conv2", "detection_encoder.dark4.branch2"),
    ("backbone.backbone.dark4.1.conv3", "detection_encoder.dark4.merge"),
    ("backbone.backbone.dark4.1.m", "detection_encoder.dark4.blocks"),
    ("backbone.backbone.dark5.0", "detection_encoder.dark5_down"),
    ("backbone.backbone.dark5.1.conv1", "detection_encoder.spp.conv1"),
    ("backbone.backbone.dark5.1.conv2", "detection_encoder.spp.conv2"),
    ("backbone.backbone.dark5.2.conv1", "detection_encoder.dark5.branch1"),
    ("backbone.backbone.dark5.2.conv2", "detection_encoder.dark5.branch2"),
    ("backbone.backbone.dark5.2.conv3", "detection_encoder.dark5.merge"),
    ("backbone.backbone.dark5.2.m", "detection_encoder.dark5.blocks"),
    ("backbone.lateral_conv0", "detection_neck.lateral5"),
    ("backbone.C3_p4.conv1", "detection_neck.top4.branch1"),
    ("backbone.C3_p4.conv2", "detection_neck.top4.branch2"),
    ("backbone.C3_p4.conv3", "detection_neck.top4.merge"),
    ("backbone.C3_p4.m", "detection_neck.top4.blocks"),
    ("backbone.reduce_conv1", "detection_neck.reduce4"),
    ("backbone.C3_p3.conv1", "detection_neck.top3.branch1"),
    ("backbone.C3_p3.conv2", "detection_neck.top3.branch2"),
    ("backbone.C3_p3.conv3", "detection_neck.top3.merge"),
    ("backbone.C3_p3.m", "detection_neck.top3.blocks"),
    ("backbone.bu_conv2", "detection_neck.down3"),
    ("backbone.C3_n3.conv1", "detection_neck.bottom4.branch1"),
    ("backbone.C3_n3.conv2", "detection_neck.bottom4.branch2"),
    ("backbone.C3_n3.conv3", "detection_neck.bottom4.merge"),
    ("backbone.C3_n3.m", "detection_neck.bottom4.blocks"),
    ("backbone.bu_conv1", "detection_neck.down4"),
    ("backbone.C3_n4.conv1", "detection_neck.bottom5.branch1"),
    ("backbone.C3_n4.conv2", "detection_neck.bottom5.branch2"),
    ("backbone.C3_n4.conv3", "detection_neck.bottom5.merge"),
    ("backbone.C3_n4.m", "detection_neck.bottom5.blocks"),
    ("head.stems", "detection_head.stems"),
    ("head.cls_convs", "detection_head.classification"),
    ("head.reg_convs", "detection_head.regression"),
    ("head.reg_preds", "detection_head.box_output"),
    ("head.obj_preds", "detection_head.objectness_output"),
)


def _load_yolox_s_detection_weights(
    model: REHOGModel, checkpoint_path: Path
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"YOLOX-S pretrained checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("YOLOX-S checkpoint must contain a state dictionary.")
    source_state = checkpoint.get(
        "state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(source_state, dict) or not source_state:
        raise ValueError(
            "YOLOX-S checkpoint contains no model state dictionary.")

    destination_state = model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    loaded: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    class_output_compatibility: list[dict[str, Any]] = []
    for source_name, source_tensor in source_state.items():
        if not isinstance(source_name, str) or not isinstance(source_tensor, torch.Tensor):
            skipped.append({
                "source": str(source_name),
                "reason": "Checkpoint entry is not a named tensor.",
            })
            continue
        if source_name.startswith("head.cls_preds."):
            destination_name = (
                "detection_head.classification_output."
                + source_name[len("head.cls_preds."):]
            )
            destination_tensor = destination_state.get(destination_name)
            if destination_tensor is None:
                reason = (
                    "No corresponding RE-CTF class-specific output tensor exists."
                )
            elif source_tensor.shape != destination_tensor.shape:
                reason = (
                    f"Class-output shape mismatch: source "
                    f"{tuple(source_tensor.shape)}, destination "
                    f"{tuple(destination_tensor.shape)}."
                )
            else:
                reason = (
                    f"Shape matches {tuple(source_tensor.shape)}, but transfer is "
                    "skipped because checkpoint class names/order are unavailable."
                )
            if source_name.endswith(".weight"):
                class_output_compatibility.append({
                    "source": source_name,
                    "destination": destination_name,
                    "source_class_channels": (
                        int(source_tensor.shape[0]
                            ) if source_tensor.ndim else None
                    ),
                    "destination_class_channels": (
                        int(destination_tensor.shape[0])
                        if destination_tensor is not None and destination_tensor.ndim
                        else None
                    ),
                    "shape_matches": (
                        destination_tensor is not None
                        and source_tensor.shape == destination_tensor.shape
                    ),
                    "transferred": False,
                })
            skipped.append({
                "source": source_name,
                "reason": reason,
            })
            continue

        mapping = next(
            (
                (source_prefix, destination_prefix)
                for source_prefix, destination_prefix in _YOLOX_S_DETECTION_PREFIX_MAP
                if source_name == source_prefix
                or source_name.startswith(source_prefix + ".")
            ),
            None,
        )
        if mapping is None:
            if ".m." in source_name:
                reason = (
                    "CSP bottleneck implementation differs; RE-CTF uses a "
                    "different inner channel layout."
                )
            elif source_name.startswith("backbone.sc"):
                reason = (
                    "RDMNet SCBottleneck is not operation-equivalent to "
                    "RE-CTF's custom neck SCBottleneck."
                )
            else:
                reason = "No verified YOLOX-S-to-RE-CTF operation mapping."
            skipped.append({"source": source_name, "reason": reason})
            continue

        source_prefix, destination_prefix = mapping
        destination_name = destination_prefix + \
            source_name[len(source_prefix):]
        destination_tensor = destination_state.get(destination_name)
        if destination_tensor is None:
            skipped.append({
                "source": source_name,
                "reason": f"Mapped destination tensor is absent: {destination_name}.",
            })
            continue
        if source_tensor.shape != destination_tensor.shape:
            skipped.append({
                "source": source_name,
                "reason": (
                    f"Shape mismatch for {destination_name}: source "
                    f"{tuple(source_tensor.shape)}, destination "
                    f"{tuple(destination_tensor.shape)}."
                ),
            })
            continue
        if destination_name in copied:
            raise ValueError(
                f"Multiple YOLOX-S tensors map to {destination_name}."
            )
        copied[destination_name] = source_tensor
        loaded.append({"source": source_name, "destination": destination_name})

    required_destinations = {
        "detection_encoder.focus.conv.conv.weight",
        "detection_encoder.dark2_down.conv.weight",
        "detection_neck.lateral5.conv.weight",
        "detection_head.stems.0.conv.weight",
    }
    missing_required = required_destinations.difference(copied)
    if missing_required:
        raise ValueError(
            "YOLOX-S checkpoint did not initialize required RE-CTF detection "
            f"layers: {', '.join(sorted(missing_required))}. Checkpoint may not "
            "be YOLOX-S or the configured detector width/depth may be incompatible."
        )

    destination_state.update(copied)
    model.load_state_dict(destination_state)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "loaded_count": len(loaded),
        "skipped_count": len(skipped),
        "loaded": loaded,
        "skipped": skipped,
        "class_output_compatibility": class_output_compatibility,
    }

    loaded_by_mapping: dict[str, int] = {}
    for entry in loaded:
        source_prefix, destination_prefix = next(
            pair
            for pair in _YOLOX_S_DETECTION_PREFIX_MAP
            if entry["source"] == pair[0]
            or entry["source"].startswith(pair[0] + ".")
        )
        label = f"{source_prefix} -> {destination_prefix}"
        loaded_by_mapping[label] = loaded_by_mapping.get(label, 0) + 1

    skipped_by_reason: dict[str, int] = {}
    for entry in skipped:
        reason = entry["reason"]
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    print(
        f"YOLOX-S detection initialization: loaded {len(loaded)} tensors; "
        f"skipped {len(skipped)}."
    )
    print("Loaded mappings:")
    for label, count in sorted(loaded_by_mapping.items()):
        print(f"  {label}: {count} tensors")
    print("Skipped tensors:")
    for reason, count in sorted(skipped_by_reason.items()):
        print(f"  {count}: {reason}")
    return report


def _run_epoch(
    model: REHOGModel,
    loader: DataLoader[dict[str, Any]],
    detection_loss: YOLODetectionLoss,
    restoration_loss: RestorationLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    alpha: float,
    beta: float,
    scaler: torch.amp.GradScaler,
    phase: str,
    total_epochs: int,
    lr_schedule=None,
    accum_steps: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "detection": 0.0, "restoration": 0.0}
    batches = 0
    total_batches = len(loader)
    optimizer_steps = 0
    skipped_steps = 0
    last_grad_norm = float("nan")
    progress = tqdm(
        loader,
        desc=f"{phase} {epoch + 1}/{total_epochs}",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        targets = _move_targets(batch["boxes"], device)
        if training:
            if lr_schedule is not None:
                current_lr = lr_schedule(epoch + batches / total_batches)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr * group.get("lr_scale", 1.0)
            if batches % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp and device.type == "cuda",
            ):
                result = model(images)
                detection = detection_loss(result["detections"], targets)
                restoration = restoration_loss(result["restored"], clean)
                restoration_scale = restoration_loss_weight(epoch)
                total = alpha * detection + beta * \
                    restoration_scale * restoration["total"]
            if training:
                scaler.scale(total / accum_steps).backward()
                if (batches + 1) % accum_steps == 0 or batches + 1 == total_batches:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=10.0)
                    last_grad_norm = float(grad_norm)
                    optimizer_steps += 1
                    if not math.isfinite(last_grad_norm):
                        skipped_steps += 1  # GradScaler skips the update on inf/NaN gradients
                    scaler.step(optimizer)
                    scaler.update()
        sums["total"] += float(total.detach())
        sums["detection"] += float(detection.detach())
        sums["restoration"] += float(restoration["total"].detach())
        batches += 1
        progress.set_postfix(
            total=f"{sums['total'] / batches:.4f}",
            det=f"{sums['detection'] / batches:.4f}",
            res=f"{sums['restoration'] / batches:.4f}",
            **(
                {
                    "gnorm": f"{last_grad_norm:.2f}",
                    "skipped": f"{skipped_steps}/{optimizer_steps}",
                }
                if training
                else {}
            ),
        )
    progress.close()
    if training:
        print(
            f"{phase} {epoch + 1}: skipped optimizer steps (non-finite gradients) "
            f"{skipped_steps}/{optimizer_steps}",
            flush=True,
        )
    if batches == 0:
        raise ValueError("A data loader produced no complete batches.")
    return {name: value / batches for name, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", required=True,
                        help="RDMNet-format training annotation file.")
    parser.add_argument("--val-list", required=True,
                        help="RDMNet-format validation annotation file.")
    parser.add_argument("--classes", required=True,
                        help="Text file containing one class name per line.")
    parser.add_argument("--image-root", default=".",
                        help="Root directory for degraded image paths.")
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained-detection",
        type=Path,
        default=Path(__file__).resolve().parent
        / "yolo"
        / "yolox_s.pth",
        help="YOLOX-S weights for verified detection-layer initialization.",
    )
    pretrained_group.add_argument(
        "--no-pretrained-detection",
        action="store_true",
        help="Train detection layers from their default initialization.",
    )
    parser.add_argument(
        "--clean-root",
        help="Root directory for paired clean images (matched by basename when not explicit in a row).",
    )
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Square input size compatible with encoder strides.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch-size x accum-steps).")
    parser.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16",
                        help="Mixed precision mode. Use 'off' to test whether fp16 causes non-finite gradients.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Recompute HOGformer blocks in backward to cut GPU memory.")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Base AdamW learning rate (detection group).")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Detection loss coefficient.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Restoration loss coefficient.")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hog-dim", type=int, default=36)
    parser.add_argument("--depth", type=float, default=0.33)
    parser.add_argument("--block-depths", default="4,6,6,6,8")
    parser.add_argument("--heads", default="1,2,4,8,16")
    parser.add_argument("--restoration-downsample-factors",
                        default="2,2,2,2,2")
    parser.add_argument("--detection-downsample-factors", default="2,2,2,2,2")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--resume", help="Optional checkpoint to resume.")
    args = parser.parse_args()

    block_depths = _parse_integer_list(args.block_depths, "--block-depths")
    heads = _parse_integer_list(args.heads, "--heads")
    restoration_factors = _parse_integer_list(
        args.restoration_downsample_factors, "--restoration-downsample-factors"
    )
    detection_factors = _parse_integer_list(
        args.detection_downsample_factors, "--detection-downsample-factors"
    )
    if len(block_depths) != 5 or len(heads) != 5:
        raise ValueError("--block-depths and --heads each require 5 values.")
    if len(restoration_factors) != 5 or len(detection_factors) != 5:
        raise ValueError(
            "--restoration-downsample-factors requires 5 values and "
            "--detection-downsample-factors requires 5 values."
        )
    required_divisor = math.lcm(
        math.prod(restoration_factors), math.prod(detection_factors))
    if args.image_size < required_divisor or args.image_size % required_divisor:
        raise ValueError(
            f"--image-size must be at least and divisible by {required_divisor} "
            "for the configured encoder downsampling factors."
        )
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1:
        raise ValueError(
            "--epochs, --batch-size and --accum-steps must be positive.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    class_names = _read_class_names(args.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    image_size = (args.image_size, args.image_size)

    train_dataset = PairedDetectionDataset(
        args.train_list, args.image_root, args.clean_root, image_size, True, len(
            class_names)
    )
    val_dataset = PairedDetectionDataset(
        args.val_list, args.image_root, args.clean_root, image_size, False, len(
            class_names)
    )
    evaluation_loader = build_evaluation_loader(
        args.val_list,
        args.image_root,
        image_size,
        len(class_names),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=detection_collate,
    )
    model_config = {
        "num_classes": len(class_names),
        "base_channels": args.base_channels,
        "hog_dim": args.hog_dim,
        "depth": args.depth,
        "block_depths": block_depths,
        "heads": heads,
        "restoration_downsample_factors": restoration_factors,
        "detection_downsample_factors": detection_factors,
    }
    model = REHOGModel(**model_config).to(device)
    if args.grad_checkpoint:
        model.set_gradient_checkpointing(True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer, scaled_learning_rate, minimum_learning_rate = _build_rdmnet_optimizer(
        model, args.batch_size, args.learning_rate, args.weight_decay, args.momentum
    )
    start_epoch = 0
    best_loss = math.inf
    best_map = -math.inf
    if args.resume:
        if not args.no_pretrained_detection:
            print(
                "Skipping YOLOX-S initialization because --resume restores the full model.")
        checkpoint = torch.load(args.resume, map_location=device)
        skipped_fusion_keys = load_compatible_model_state_dict(
            model, checkpoint["model"]
        )
        if skipped_fusion_keys:
            print(
                "Loaded legacy full-CTF model weights for the projection-add "
                "ablation; removed fusion weights were skipped. Optimizer state "
                "and epoch counters are reset for this ablation run."
            )
        else:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_loss = float(checkpoint.get("best_val_loss", best_loss))
            best_map = float(checkpoint.get("best_val_map", best_map))
    elif not args.no_pretrained_detection:
        load_report = _load_yolox_s_detection_weights(
            model, args.pretrained_detection
        )
        report_path = output_dir / "pretrained_detection_load_report.json"
        report_path.write_text(
            json.dumps(load_report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Pretrained transfer report saved to {report_path}")
    else:
        print("YOLOX-S detection initialization disabled.")

    detection_loss = YOLODetectionLoss(
        len(class_names), strides=model.detection_strides
    )
    restoration_loss = RestorationLoss().to(device)
    use_amp = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    print(f"Mixed precision: {args.amp if use_amp else 'off'}", flush=True)
    best_checkpoint_path = output_dir / "best.pth"

    for epoch in range(start_epoch, args.epochs):
        learning_rate = _rdmnet_epoch_learning_rate(
            epoch, args.epochs, scaled_learning_rate, minimum_learning_rate
        )
        no_augmentation_epochs = min(max(0.3 * args.epochs, 1), 15)
        train_dataset.augment = epoch < args.epochs - no_augmentation_epochs
        train_metrics = _run_epoch(
            model,
            train_loader,
            detection_loss,
            restoration_loss,
            optimizer,
            device,
            epoch,
            args.alpha,
            args.beta,
            scaler,
            phase="Train",
            total_epochs=args.epochs,
            lr_schedule=lambda e: _rdmnet_epoch_learning_rate(
                e, args.epochs, scaled_learning_rate, minimum_learning_rate
            ),
            accum_steps=args.accum_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                detection_loss,
                restoration_loss,
                None,
                device,
                epoch,
                args.alpha,
                args.beta,
                scaler,
                phase="Validation",
                total_epochs=args.epochs,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        map_metrics = evaluate_map(
            model,
            evaluation_loader,
            class_names,
            device,
            confidence_threshold=0.001,
            nms_iou_threshold=0.65,
            iou_threshold=0.5,
        )
        restoration_scale = restoration_loss_weight(epoch)
        best_loss = min(best_loss, val_metrics["total"])
        current_map = map_metrics["mAP@0.5"]
        best_checkpoint_updated = current_map > best_map
        best_map = max(best_map, current_map)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "fusion_variant": "restoration_projection_add",
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_loss,
            "best_val_map": best_map,
            "model_config": model_config,
            "class_names": class_names,
            "image_size": args.image_size,
            "alpha": args.alpha,
            "beta": args.beta,
            "validation_metrics": map_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if best_checkpoint_updated:
            torch.save(checkpoint, best_checkpoint_path)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] complete "
            f"train={train_metrics['total']:.4f} "
            f"val={val_metrics['total']:.4f} "
            f"det={val_metrics['detection']:.4f} "
            f"res={val_metrics['restoration']:.4f} "
            f"res_weight={restoration_scale:.2f} "
            f"mAP@0.5={current_map * 100:.2f}% "
            f"lr={learning_rate:.6g}"
        )
        print(
            f"Best checkpoint: {best_checkpoint_path} "
            f"(validation mAP@0.5={best_map * 100:.2f}%)"
            f"{' [updated this epoch]' if best_checkpoint_updated else ''}"
        )
        if epoch + 1 == 45:
            print("Stopping after epoch 45.")
            break


if __name__ == "__main__":
    main()
