"""Dataset and DataLoader construction for the crop classifier.

Reads labelled crops from the layout documented in
``docs/annotation_guidelines.md``::

    <dataset_root>/
    ├── train/{building,tree,car,other}/
    └── validation/{building,tree,car,other}/

Class-to-index mapping
----------------------
Indices come from the **sorted** class list, so they are stable no matter what
order classes appear in ``configs/classifier.yaml``. For the four target classes
that is::

    building -> 0
    car      -> 1
    other    -> 2
    tree     -> 3

Use :func:`class_to_index` / :func:`index_to_class` rather than hardcoding the
mapping; training and inference must agree on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Crops are RGB photographs, so the pretrained backbone's ImageNet statistics apply.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def class_to_index(classes: Sequence[str]) -> dict[str, int]:
    """Map class name to label index using the sorted class order.

    Args:
        classes: Class names, in any order.

    Returns:
        ``{class_name: index}`` with indices assigned alphabetically.
    """
    return {name: index for index, name in enumerate(sorted(classes))}


def index_to_class(classes: Sequence[str]) -> list[str]:
    """Return class names ordered by label index (i.e. sorted alphabetically)."""
    return sorted(classes)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Build the preprocessing pipeline for one split.

    Both splits resize to a square ``image_size`` and normalise with ImageNet
    statistics. Augmentation (flips, small rotations, colour jitter) is applied
    to the training split only, so validation accuracy stays comparable across
    epochs.

    Args:
        image_size: Target width and height in pixels.
        train: Whether to include augmentation.

    Returns:
        A composed torchvision transform producing a normalised ``float32``
        tensor of shape ``(3, image_size, image_size)``.
    """
    if train:
        stages: list[transforms.Compose | object] = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # drone crops have no canonical up
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        stages = [transforms.Resize((image_size, image_size))]

    stages += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(stages)  # type: ignore[arg-type]


class CropDataset(Dataset):
    """Labelled SAM crops for one split, filed one directory per class.

    Args:
        root: Split directory, e.g. ``data/classifier/train``.
        classes: Class names to load. Every class must have a directory under
            ``root``; empty class directories are allowed.
        image_size: Target crop size in pixels.
        train: Whether to apply training augmentation.

    Raises:
        FileNotFoundError: If ``root`` or any class directory is missing.
    """

    def __init__(self, root: str | Path, classes: Sequence[str], image_size: int, train: bool) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.root}")

        self.classes = index_to_class(classes)
        self.class_to_idx = class_to_index(classes)
        self.transform = build_transforms(image_size, train=train)
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}. Expected layout is "
                    f"<dataset_root>/<split>/<class>/ — see docs/annotation_guidelines.md"
                )
            label = self.class_to_idx[class_name]
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

    def __len__(self) -> int:
        """Number of crops in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return ``(image_tensor, label_index)`` for one crop.

        The image is converted to RGB (dropping any alpha channel left by the
        SAM crop stage) before transforms are applied.
        """
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label

    def class_counts(self) -> dict[str, int]:
        """Return the number of crops per class, useful for spotting imbalance."""
        counts = {name: 0 for name in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


def create_dataloaders(
    dataset_root: str | Path,
    classes: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    train_split: str = "train",
    validation_split: str = "validation",
) -> tuple[DataLoader, DataLoader]:
    """Build the training and validation dataloaders.

    Args:
        dataset_root: Root containing the split directories, e.g. ``data/classifier``.
        classes: Class names for the classifier.
        image_size: Target crop size in pixels.
        batch_size: Batch size for both loaders.
        num_workers: Worker processes per loader. Defaults to 0, which is the
            safe choice on Windows and for small datasets.
        train_split: Name of the training split directory.
        validation_split: Name of the validation split directory.

    Returns:
        ``(train_loader, validation_loader)``. The training loader shuffles and
        drops nothing; the validation loader preserves order.

    Raises:
        ValueError: If either split contains no images.
    """
    root = Path(dataset_root)
    train_dataset = CropDataset(root / train_split, classes, image_size, train=True)
    validation_dataset = CropDataset(root / validation_split, classes, image_size, train=False)

    for name, dataset in ((train_split, train_dataset), (validation_split, validation_dataset)):
        if len(dataset) == 0:
            raise ValueError(
                f"No images found in the '{name}' split under {root / name}. "
                f"Run scripts/generate_sample_crops.py to create placeholder crops."
            )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, validation_loader
