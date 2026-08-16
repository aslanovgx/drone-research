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
from typing import Any, NamedTuple, Optional, Sequence, TypedDict

import torch

if __package__ in (None, ""):
    # Direct script run (`python src/classification/inference.py`): put `src` on the
    # path and declare the package so the relative imports below resolve (PEP 366).
    # Relative imports keep the module importable as `classification.inference` and
    # as `src.classification.inference`, whichever layout the pipeline settles on.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "classification"

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_device
from .dataset import build_transforms, index_to_class, load_crop_image
from .model import build_model

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


class LoadedClassifier(NamedTuple):
    """A ready-to-use model plus the preprocessing settings it was trained with."""

    model: torch.nn.Module
    classes: list[str]
    image_size: int
    preserve_aspect: bool


def load_classifier(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_classes: list[str] | None = None,
    fallback_backbone: str = "mobilenet_v3_small",
    fallback_image_size: int = 224,
    fallback_preserve_aspect: bool = True,
) -> LoadedClassifier:
    """Load a trained classifier from a checkpoint written by ``train.py``.

    The checkpoint's own ``classes`` / ``backbone`` / ``image_size`` /
    ``preserve_aspect`` are authoritative; the fallbacks (taken from the config)
    only apply to older checkpoints that stored weights alone.

    Args:
        checkpoint_path: Path to the ``.pt`` file.
        device: Device to load the model onto.
        fallback_classes: Class names to assume if the checkpoint lacks them.
        fallback_backbone: Backbone to assume if the checkpoint lacks it.
        fallback_image_size: Crop size to assume if the checkpoint lacks it.
        fallback_preserve_aspect: Letterbox setting to assume if absent.

    Returns:
        A :class:`LoadedClassifier` — the model in eval mode plus the class list,
        crop size and letterbox flag it expects.

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
    preserve_aspect = bool(checkpoint.get("preserve_aspect", fallback_preserve_aspect))

    # Weights come from the checkpoint, so skip the pretrained download entirely.
    model = build_model(num_classes=len(classes), backbone=backbone, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return LoadedClassifier(model, classes, image_size, preserve_aspect)


def _load_crop_tensor(crop_path: Path, transform: Any) -> torch.Tensor:
    """Read one crop and apply the inference transforms.

    Raises:
        FileNotFoundError: If the crop does not exist.
    """
    if not crop_path.is_file():
        raise FileNotFoundError(f"Crop not found: {crop_path}")
    return transform(load_crop_image(crop_path))


def classify_crop(
    crop_path: str | Path,
    model: torch.nn.Module,
    classes: list[str],
    image_size: int,
    device: torch.device,
    confidence_decimals: int = 2,
    segment_id: int | None = None,
    preserve_aspect: bool = True,
) -> Prediction:
    """Classify one crop with an already-loaded model.

    Args:
        crop_path: Path to the crop image.
        model: Model in eval mode, as returned by :func:`load_classifier`.
        classes: Class names ordered by label index.
        image_size: Crop size the model was trained on.
        device: Device the model lives on.
        confidence_decimals: Decimal places for the reported confidence.
        segment_id: Explicit id, used in place of parsing the filename. Callers
            that already hold the SAM metadata should pass it rather than relying
            on the ``segment_<id>`` naming convention.
        preserve_aspect: Must match the value the model was trained with; take it
            from :attr:`LoadedClassifier.preserve_aspect`.

    Returns:
        A :class:`Prediction` with ``segment_id``, ``class`` and ``confidence``.

    Raises:
        FileNotFoundError: If the crop does not exist.
    """
    return classify_crops(
        [crop_path],
        model=model,
        classes=classes,
        image_size=image_size,
        device=device,
        confidence_decimals=confidence_decimals,
        segment_ids=None if segment_id is None else [segment_id],
        preserve_aspect=preserve_aspect,
    )[0]


@torch.no_grad()
def classify_crops(
    crop_paths: Sequence[str | Path],
    model: torch.nn.Module,
    classes: list[str],
    image_size: int,
    device: torch.device,
    confidence_decimals: int = 2,
    segment_ids: Sequence[int | None] | None = None,
    batch_size: int = 32,
    preserve_aspect: bool = True,
) -> list[Prediction]:
    """Classify many crops with one loaded model, batching the forward passes.

    This is the entry point for the pipeline: SAM emits hundreds of crops per
    frame, and classifying them one at a time wastes both the model load and the
    batch dimension.

    Args:
        crop_paths: Paths to the crops, in the order results should be returned.
        model: Model in eval mode, as returned by :func:`load_classifier`.
        classes: Class names ordered by label index.
        image_size: Crop size the model was trained on.
        device: Device the model lives on.
        confidence_decimals: Decimal places for the reported confidence.
        segment_ids: Explicit ids parallel to ``crop_paths``. When omitted, each
            id is parsed from its filename.
        batch_size: Number of crops per forward pass.
        preserve_aspect: Must match the value the model was trained with; take it
            from :attr:`LoadedClassifier.preserve_aspect`.

    Returns:
        One :class:`Prediction` per input path, in input order. An empty input
        returns an empty list.

    Raises:
        FileNotFoundError: If any crop does not exist.
        ValueError: If ``segment_ids`` is given but its length differs from
            ``crop_paths``.
    """
    paths = [Path(path) for path in crop_paths]
    if not paths:
        return []
    if segment_ids is not None and len(segment_ids) != len(paths):
        raise ValueError(f"segment_ids has {len(segment_ids)} entries but {len(paths)} crops were given.")

    # Validation transforms — no augmentation at inference time.
    transform = build_transforms(image_size, train=False, preserve_aspect=preserve_aspect)
    predictions: list[Prediction] = []

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        batch = torch.stack([_load_crop_tensor(path, transform) for path in chunk]).to(device)
        probabilities = torch.softmax(model(batch), dim=1)

        for offset, path in enumerate(chunk):
            index = int(probabilities[offset].argmax().item())
            explicit_id = segment_ids[start + offset] if segment_ids is not None else None
            predictions.append(
                {
                    "segment_id": explicit_id if explicit_id is not None else parse_segment_id(path),
                    "class": classes[index],
                    "confidence": round(float(probabilities[offset, index].item()), confidence_decimals),
                }
            )

    return predictions


def predict(
    crop_path: str | Path,
    config: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    segment_id: int | None = None,
) -> Prediction:
    """Classify one crop, loading the model from the configured checkpoint.

    Convenience wrapper for one-off calls. It reloads the checkpoint on every
    call, so callers classifying more than a handful of crops should use
    :func:`predict_many`, or :func:`load_classifier` plus :func:`classify_crops`.

    Args:
        crop_path: Path to the crop image.
        config: Parsed config; loaded from the default path when omitted.
        checkpoint_path: Overrides ``checkpoint.path`` from the config.
        segment_id: Explicit id, used in place of parsing the filename.

    Returns:
        A :class:`Prediction` for the crop.
    """
    return predict_many(
        [crop_path],
        config=config,
        checkpoint_path=checkpoint_path,
        segment_ids=None if segment_id is None else [segment_id],
    )[0]


def predict_many(
    crop_paths: Sequence[str | Path],
    config: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    segment_ids: Sequence[int | None] | None = None,
) -> list[Prediction]:
    """Classify a batch of crops, loading the checkpoint exactly once.

    The intended entry point for the pipeline: hand it every crop SAM produced
    for one frame and get one prediction per crop back, in input order.

    Args:
        crop_paths: Paths to the crops.
        config: Parsed config; loaded from the default path when omitted.
        checkpoint_path: Overrides ``checkpoint.path`` from the config.
        segment_ids: Explicit ids parallel to ``crop_paths``; parsed from the
            filenames when omitted.

    Returns:
        One :class:`Prediction` per input path. An empty input returns an empty
        list without loading the model.
    """
    if not crop_paths:
        return []

    config = config if config is not None else load_config()
    data_cfg = config.get("data", {})
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))

    bundle = load_classifier(
        checkpoint_path or config.get("checkpoint", {}).get("path", "checkpoints/classifier.pt"),
        device=device,
        fallback_classes=list(config.get("classes", [])),
        fallback_backbone=str(config.get("model", {}).get("backbone", "mobilenet_v3_small")),
        fallback_image_size=int(data_cfg.get("image_size", 224)),
        fallback_preserve_aspect=bool(data_cfg.get("preserve_aspect", True)),
    )
    return classify_crops(
        crop_paths,
        model=bundle.model,
        classes=bundle.classes,
        image_size=bundle.image_size,
        device=device,
        confidence_decimals=int(config.get("inference", {}).get("confidence_decimals", 2)),
        segment_ids=segment_ids,
        preserve_aspect=bundle.preserve_aspect,
    )


def main() -> None:
    """CLI entry point: print the prediction for one crop as JSON."""
    parser = argparse.ArgumentParser(description="Classify a single SAM crop.")
    parser.add_argument("crops", type=Path, nargs="+", help="Crop paths, e.g. outputs/crops/segment_17.png")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to classifier.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint.path from the config")
    parser.add_argument("--segment-id", type=int, default=None, help="Explicit id; only valid for a single crop")
    args = parser.parse_args()

    if args.segment_id is not None and len(args.crops) > 1:
        parser.error("--segment-id can only be used with a single crop.")

    predictions = predict_many(
        args.crops,
        config=load_config(args.config),
        checkpoint_path=args.checkpoint,
        segment_ids=None if args.segment_id is None else [args.segment_id],
    )
    # One JSON object for a single crop, a JSON array for several.
    print(json.dumps(predictions[0] if len(args.crops) == 1 else predictions))


if __name__ == "__main__":
    main()
