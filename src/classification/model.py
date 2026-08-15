"""Model construction for the crop classifier.

Backbone choice: **MobileNetV3-Small**. SAM emits hundreds of crops per drone
frame, so the classifier is called far more often than the segmenter and has to
stay cheap; MobileNetV3-Small is ~2.5M parameters against ResNet18's ~11M, runs
comfortably on CPU, and its ImageNet features transfer well to the small,
low-detail crops this stage receives. ResNet18 stays available through the same
factory for accuracy comparisons on the real labelled set.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision import models

# Backbone name -> (torchvision constructor, default weights enum).
SUPPORTED_BACKBONES: dict[str, tuple[object, object]] = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
}

DEFAULT_BACKBONE = "mobilenet_v3_small"


def build_model(
    num_classes: int,
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = True,
) -> nn.Module:
    """Build a classification model whose head outputs ``num_classes`` logits.

    The pretrained ImageNet head is replaced by a fresh linear layer sized to the
    target classes; every other layer keeps its pretrained weights and is
    fine-tuned during training.

    Args:
        num_classes: Number of target classes (4 for building/tree/car/other).
        backbone: Key from :data:`SUPPORTED_BACKBONES`.
        pretrained: Load ImageNet weights. Falls back to random initialisation
            with a warning if the weights cannot be downloaded (offline runs),
            so a smoke test still completes.

    Returns:
        An :class:`torch.nn.Module` in eval-agnostic state (caller sets the mode)
        producing logits of shape ``(batch, num_classes)``.

    Raises:
        ValueError: If ``num_classes`` is not positive or ``backbone`` is unknown.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from: {sorted(SUPPORTED_BACKBONES)}")

    constructor, default_weights = SUPPORTED_BACKBONES[backbone]
    weights = default_weights if pretrained else None

    try:
        model = constructor(weights=weights)  # type: ignore[operator]
    except Exception as error:  # network/cache failure only — bad args raise above
        if not pretrained:
            raise
        warnings.warn(
            f"Could not load pretrained weights for '{backbone}' ({error}). "
            f"Falling back to random initialisation; accuracy will be poor.",
            RuntimeWarning,
            stacklevel=2,
        )
        model = constructor(weights=None)  # type: ignore[operator]

    _replace_classifier_head(model, backbone, num_classes)
    return model


def _replace_classifier_head(model: nn.Module, backbone: str, num_classes: int) -> None:
    """Swap the backbone's ImageNet head for a linear layer of ``num_classes``."""
    if backbone.startswith("mobilenet"):
        # MobileNetV3's head is a Sequential; the last Linear is the classifier.
        in_features = model.classifier[-1].in_features  # type: ignore[index]
        model.classifier[-1] = nn.Linear(in_features, num_classes)  # type: ignore[index]
    else:  # resnet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # type: ignore[union-attr]


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters, for logging model size."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":  # quick manual check: shapes only, no data required
    net = build_model(num_classes=4)
    logits = net(torch.randn(2, 3, 224, 224))
    print(f"backbone={DEFAULT_BACKBONE} params={count_trainable_parameters(net):,} logits={tuple(logits.shape)}")
