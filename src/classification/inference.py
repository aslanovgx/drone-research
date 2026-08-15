"""Inference entry point for the crop classifier.

Classifies a single crop produced by the SAM stage
(``outputs/crops/segment_<id>.png``) and returns::

    {"segment_id": 17, "class": "building", "confidence": 0.92}

``segment_id`` is parsed from the filename, ``class`` is the argmax class name
and ``confidence`` is the softmax probability of that class, rounded to the
number of decimals set in ``configs/classifier.yaml``.

Usage::

    python src/classification/inference.py outputs/crops/segment_17.png
    python src/classification/inference.py <crop> --checkpoint checkpoints/classifier.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, TypedDict

import torch
from PIL import Image

if __package__ in (None, ""):  # allow `python src/classification/inference.py` without PYTHONPATH
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.config import DEFAULT_CONFIG_PATH, load_config, resolve_device
from classification.dataset import build_transforms, index_to_class
from classification.model import build_model

# Matches segment_17.png and prefixed variants such as scene3_segment_17.png.
SEGMENT_ID_PATTERN = re.compile(r"segment_(\d+)", re.IGNORECASE)


# One classified crop: {"segment_id": 17, "class": "building", "confidence": 0.92}.
# Declared functionally because `class` is a Python keyword and cannot be a field name.
# `segment_id` is None when the filename does not follow the segment_<id> convention.
Prediction = TypedDict("Prediction", {"segment_id": Optional[int], "class": str, "confidence": float})


def parse_segment_id(crop_path: str | Path) -> int | None:
    """Extract the SAM segment id from a crop filename.

    Args:
        crop_path: Path whose filename contains ``segment_<id>``.

    Returns:
        The integer id, or ``None`` if the filename does not follow the
        convention (the crop is still classified; only the id is unknown).
    """
    match = SEGMENT_ID_PATTERN.search(Path(crop_path).stem)
    return int(match.group(1)) if match else None


def load_classifier(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_classes: list[str] | None = None,
    fallback_backbone: str = "mobilenet_v3_small",
    fallback_image_size: int = 224,
) -> tuple[torch.nn.Module, list[str], int]:
    """Load a trained classifier from a checkpoint written by ``train.py``.

    The checkpoint's own ``classes`` / ``backbone`` / ``image_size`` are
    authoritative; the fallbacks (taken from the config) only apply to older
    checkpoints that stored weights alone.

    Args:
        checkpoint_path: Path to the ``.pt`` file.
        device: Device to load the model onto.
        fallback_classes: Class names to assume if the checkpoint lacks them.
        fallback_backbone: Backbone to assume if the checkpoint lacks it.
        fallback_image_size: Crop size to assume if the checkpoint lacks it.

    Returns:
        ``(model_in_eval_mode, classes_by_index, image_size)``.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the class list can be recovered from neither source.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train the classifier first (src/classification/train.py).")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    classes = checkpoint.get("classes") or fallback_classes
    if not classes:
        raise ValueError(f"Checkpoint {path} has no class list and no fallback was provided.")
    classes = index_to_class(classes)

    backbone = checkpoint.get("backbone", fallback_backbone)
    image_size = int(checkpoint.get("image_size", fallback_image_size))

    # Weights come from the checkpoint, so skip the pretrained download entirely.
    model = build_model(num_classes=len(classes), backbone=backbone, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, classes, image_size


@torch.no_grad()
def classify_crop(
    crop_path: str | Path,
    model: torch.nn.Module,
    classes: list[str],
    image_size: int,
    device: torch.device,
    confidence_decimals: int = 2,
) -> Prediction:
    """Classify one crop with an already-loaded model.

    Args:
        crop_path: Path to the crop image.
        model: Model in eval mode, as returned by :func:`load_classifier`.
        classes: Class names ordered by label index.
        image_size: Crop size the model was trained on.
        device: Device the model lives on.
        confidence_decimals: Decimal places for the reported confidence.

    Returns:
        A :class:`Prediction` with ``segment_id``, ``class`` and ``confidence``.

    Raises:
        FileNotFoundError: If the crop does not exist.
    """
    path = Path(crop_path)
    if not path.is_file():
        raise FileNotFoundError(f"Crop not found: {path}")

    # Validation transforms — no augmentation at inference time.
    transform = build_transforms(image_size, train=False)
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    index = int(probabilities.argmax().item())

    return {
        "segment_id": parse_segment_id(path),
        "class": classes[index],
        "confidence": round(float(probabilities[index].item()), confidence_decimals),
    }


def predict(
    crop_path: str | Path,
    config: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> Prediction:
    """Classify one crop, loading the model from the configured checkpoint.

    Convenience wrapper for one-off calls; batch callers should use
    :func:`load_classifier` once and then :func:`classify_crop` per crop.

    Args:
        crop_path: Path to the crop image.
        config: Parsed config; loaded from the default path when omitted.
        checkpoint_path: Overrides ``checkpoint.path`` from the config.

    Returns:
        A :class:`Prediction` for the crop.
    """
    config = config if config is not None else load_config()
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))

    model, classes, image_size = load_classifier(
        checkpoint_path or config.get("checkpoint", {}).get("path", "checkpoints/classifier.pt"),
        device=device,
        fallback_classes=list(config.get("classes", [])),
        fallback_backbone=str(config.get("model", {}).get("backbone", "mobilenet_v3_small")),
        fallback_image_size=int(config.get("data", {}).get("image_size", 224)),
    )
    return classify_crop(
        crop_path,
        model=model,
        classes=classes,
        image_size=image_size,
        device=device,
        confidence_decimals=int(config.get("inference", {}).get("confidence_decimals", 2)),
    )


def main() -> None:
    """CLI entry point: print the prediction for one crop as JSON."""
    parser = argparse.ArgumentParser(description="Classify a single SAM crop.")
    parser.add_argument("crop", type=Path, help="Path to a crop, e.g. outputs/crops/segment_17.png")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to classifier.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint.path from the config")
    args = parser.parse_args()

    prediction = predict(args.crop, config=load_config(args.config), checkpoint_path=args.checkpoint)
    print(json.dumps(prediction))


if __name__ == "__main__":
    main()
