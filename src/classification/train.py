"""Training entry point for the crop classifier.

Reads every hyperparameter from ``configs/classifier.yaml``, trains the backbone
from :mod:`classification.model` on the crops served by
:mod:`classification.dataset`, logs train loss and validation accuracy per epoch,
and writes a checkpoint to the configured (gitignored) path.

Usage::

    python src/classification/train.py
    python src/classification/train.py --config configs/classifier.yaml --epochs 2
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    # Direct script run (`python src/classification/train.py`): put `src` on the path
    # and declare the package so the relative imports below resolve (PEP 366).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "classification"

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_device
from .dataset import create_dataloaders, index_to_class
from .model import build_model, count_trainable_parameters


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, name: str, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create the optimizer named in the config.

    Args:
        model: Model whose parameters are optimised.
        name: ``"adamw"`` or ``"adam"`` (case-insensitive).
        learning_rate: Initial learning rate.
        weight_decay: L2 / decoupled weight decay.

    Returns:
        The configured optimizer.

    Raises:
        ValueError: If ``name`` is not a supported optimizer.
    """
    optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}
    key = name.lower()
    if key not in optimizers:
        raise ValueError(f"Unsupported optimizer '{name}'. Choose from: {sorted(optimizers)}")
    return optimizers[key](model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Returns:
        Mean training loss over all samples in the epoch.
    """
    model.train()
    running_loss = 0.0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate on the validation split.

    Returns:
        ``(mean_loss, accuracy)`` where accuracy is in ``[0, 1]``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running_loss += criterion(logits, labels).item() * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    classes: list[str],
    backbone: str,
    image_size: int,
    epoch: int,
    validation_accuracy: float,
    preserve_aspect: bool = True,
) -> Path:
    """Write the model weights plus everything inference needs to rebuild it.

    The class list, backbone name and image size travel with the weights so
    inference never has to trust that the config still matches the checkpoint.

    Args:
        model: Trained model.
        path: Destination file (parent directories are created).
        classes: Class names ordered by label index.
        backbone: Backbone key used to build the model.
        image_size: Crop size the model was trained on.
        epoch: 1-based epoch this checkpoint came from.
        validation_accuracy: Validation accuracy at that epoch.
        preserve_aspect: Whether crops were letterboxed rather than stretched.
            Inference must match this or every crop is preprocessed differently
            from training.

    Returns:
        The path written.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "backbone": backbone,
            "image_size": image_size,
            "preserve_aspect": preserve_aspect,
            "epoch": epoch,
            "validation_accuracy": validation_accuracy,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train(config: dict[str, Any]) -> Path:
    """Run the full training loop described by ``config``.

    Args:
        config: Parsed ``configs/classifier.yaml`` contents.

    Returns:
        Path to the saved checkpoint (best validation accuracy seen).
    """
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    classes = index_to_class(config["classes"])
    image_size = int(data_cfg.get("image_size", 224))
    backbone = str(model_cfg.get("backbone", "mobilenet_v3_small"))
    epochs = int(train_cfg.get("epochs", 3))
    preserve_aspect = bool(data_cfg.get("preserve_aspect", True))

    set_seed(int(train_cfg.get("seed", 42)))
    device = resolve_device(str(train_cfg.get("device", "auto")))

    train_loader, validation_loader = create_dataloaders(
        dataset_root=data_cfg.get("dataset_root", "data/classifier"),
        classes=classes,
        image_size=image_size,
        batch_size=int(train_cfg.get("batch_size", 16)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        train_split=str(data_cfg.get("train_split", "train")),
        validation_split=str(data_cfg.get("validation_split", "validation")),
        preserve_aspect=preserve_aspect,
        min_crop_pixels=int(data_cfg.get("min_crop_pixels", 32)),
    )

    model = build_model(
        num_classes=len(classes),
        backbone=backbone,
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    optimizer = build_optimizer(
        model,
        name=str(train_cfg.get("optimizer", "adamw")),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    print(
        f"device={device} backbone={backbone} classes={classes} "
        f"params={count_trainable_parameters(model):,} preserve_aspect={preserve_aspect}\n"
        f"train={len(train_loader.dataset)} crops {train_loader.dataset.class_counts()}\n"
        f"validation={len(validation_loader.dataset)} crops {validation_loader.dataset.class_counts()}"
    )

    # Nadir urban crops are overwhelmingly `other` (pavement, dirt, shadow), so an
    # unweighted loss rewards always predicting it. "auto" counteracts that with
    # inverse-frequency weights; "none" disables it.
    weighting = str(train_cfg.get("class_weights", "auto")).lower()
    weights = train_loader.dataset.class_weights().to(device) if weighting == "auto" else None
    if weights is not None:
        print("class weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(classes, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights)

    checkpoint_path = Path(config.get("checkpoint", {}).get("path", "checkpoints/classifier.pt"))
    # Stop once validation accuracy has not improved for this many epochs. Lets the
    # epoch budget be set generously without overfitting the tail of the run; 0 runs
    # every epoch regardless.
    patience = int(train_cfg.get("early_stopping_patience", 0))
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={validation_loss:.4f}  val_acc={validation_accuracy:.2%}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                checkpoint_path,
                classes,
                backbone,
                image_size,
                epoch,
                validation_accuracy,
                preserve_aspect=preserve_aspect,
            )
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"early stop: no validation improvement for {patience} epochs")
                break

    print(
        f"saved checkpoint -> {checkpoint_path}  "
        f"(best val_acc={best_accuracy:.2%} at epoch {best_epoch})"
    )
    return checkpoint_path


def main() -> None:
    """CLI entry point; command-line flags override the config file."""
    parser = argparse.ArgumentParser(description="Train the drone crop classifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to classifier.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--device", type=str, default=None, help="Override training.device (auto|cpu|cuda)")
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {"epochs": args.epochs, "batch_size": args.batch_size, "device": args.device}
    config.setdefault("training", {}).update({k: v for k, v in overrides.items() if v is not None})

    train(config)


if __name__ == "__main__":
    main()
