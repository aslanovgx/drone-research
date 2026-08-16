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
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Crops are RGB photographs, so the pretrained backbone's ImageNet statistics apply.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Padding colour = ImageNet mean in 8-bit, so padded borders normalise to ~0 and
# contribute nothing to the first convolution.
PAD_FILL: tuple[int, int, int] = tuple(round(channel * 255) for channel in IMAGENET_MEAN)  # type: ignore[assignment]

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Crops smaller than this on their short side are unreliable to label or classify.
# At the Packing House District's ~1.4 cm/px, 32 px is roughly 45 cm on the ground.
DEFAULT_MIN_CROP_PIXELS = 32


class LetterboxResize:
    """Resize to a square, preserving aspect ratio and padding the remainder.

    SAM crops are strongly non-square — a car at this dataset's ground sampling
    distance is about 330x150 px, i.e. 2.2:1. Squashing that into a square warps
    it toward the proportions of a roof, throwing away a shape cue that
    distinguishes the classes. Letterboxing keeps the aspect ratio and pads
    instead.

    Args:
        size: Output width and height in pixels.
        fill: RGB padding colour.
    """

    def __init__(self, size: int, fill: tuple[int, int, int] = PAD_FILL) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return a ``size`` x ``size`` image with the original aspect preserved."""
        width, height = image.size
        scale = self.size / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.resize(new_size, Image.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - new_size[0]) // 2, (self.size - new_size[1]) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size}, fill={self.fill})"


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


def build_transforms(image_size: int, train: bool, preserve_aspect: bool = True) -> transforms.Compose:
    """Build the preprocessing pipeline for one split.

    Both splits resize to a square ``image_size`` and normalise with ImageNet
    statistics. Augmentation (flips, small rotations, colour jitter) is applied
    to the training split only, so validation accuracy stays comparable across
    epochs.

    Args:
        image_size: Target width and height in pixels.
        train: Whether to include augmentation.
        preserve_aspect: Letterbox rather than stretch. Recommended for SAM crops,
            whose aspect ratios carry class information. Training and inference
            must use the same setting, which is why it travels in the checkpoint.

    Returns:
        A composed torchvision transform producing a normalised ``float32``
        tensor of shape ``(3, image_size, image_size)``.
    """
    resize: object = (
        LetterboxResize(image_size) if preserve_aspect else transforms.Resize((image_size, image_size))
    )

    if train:
        stages: list[object] = [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # drone crops have no canonical up
            # Rotation fill matches the letterbox padding so corners stay neutral.
            transforms.RandomRotation(degrees=15, fill=list(PAD_FILL)),
            # Afternoon capture means strong, high-contrast shadows; jitter keeps the
            # model from keying on scene brightness.
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        ]
    else:
        stages = [resize]

    stages += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(stages)  # type: ignore[arg-type]


def load_crop_image(path: str | Path) -> Image.Image:
    """Open a crop as RGB, honouring any EXIF orientation.

    Source frames are MPO-tagged JPEGs; PIL reads them normally. ``convert("RGB")``
    also flattens the alpha channel that masked crops may carry.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class CropDataset(Dataset):
    """Labelled SAM crops for one split, filed one directory per class.

    Args:
        root: Split directory, e.g. ``data/classifier/train``.
        classes: Class names to load. Every class must have a directory under
            ``root``; empty class directories are allowed.
        image_size: Target crop size in pixels.
        train: Whether to apply training augmentation.
        preserve_aspect: Letterbox instead of stretching to a square.
        min_crop_pixels: Crops whose short side is below this are skipped as
            unreliable. Set to 0 to keep everything.

    Raises:
        FileNotFoundError: If ``root`` or any class directory is missing.
    """

    def __init__(
        self,
        root: str | Path,
        classes: Sequence[str],
        image_size: int,
        train: bool,
        preserve_aspect: bool = True,
        min_crop_pixels: int = DEFAULT_MIN_CROP_PIXELS,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.root}")

        self.classes = index_to_class(classes)
        self.class_to_idx = class_to_index(classes)
        self.transform = build_transforms(image_size, train=train, preserve_aspect=preserve_aspect)
        self.samples: list[tuple[Path, int]] = []
        self.skipped_small = 0
        self.skipped_unreadable = 0

        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}. Expected layout is "
                    f"<dataset_root>/<split>/<class>/ — see docs/annotation_guidelines.md"
                )
            label = self.class_to_idx[class_name]
            for path in sorted(class_dir.iterdir()):
                if not (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                    continue
                if min_crop_pixels > 0 and not self._is_large_enough(path, min_crop_pixels):
                    continue
                self.samples.append((path, label))

    def _is_large_enough(self, path: Path, min_crop_pixels: int) -> bool:
        """Check a crop's short side without decoding pixel data."""
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            self.skipped_unreadable += 1
            return False
        if min(width, height) < min_crop_pixels:
            self.skipped_small += 1
            return False
        return True

    def __len__(self) -> int:
        """Number of crops in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return ``(image_tensor, label_index)`` for one crop.

        The image is converted to RGB (dropping any alpha channel left by the
        SAM crop stage) before transforms are applied.
        """
        path, label = self.samples[index]
        return self.transform(load_crop_image(path)), label

    def class_counts(self) -> dict[str, int]:
        """Return the number of crops per class, useful for spotting imbalance."""
        counts = {name: 0 for name in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency loss weights, ordered by label index.

        Nadir urban scenes are dominated by pavement, dirt and shadow, so `other`
        will outnumber the target classes heavily. Without weighting, a model that
        predicts `other` for everything scores well while being useless. Empty
        classes get weight 0 so they cannot produce a division by zero.
        """
        counts = self.class_counts()
        total = sum(counts.values())
        num_classes = len(self.classes)
        weights = [
            total / (num_classes * counts[name]) if counts[name] else 0.0 for name in self.classes
        ]
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    dataset_root: str | Path,
    classes: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    train_split: str = "train",
    validation_split: str = "validation",
    preserve_aspect: bool = True,
    min_crop_pixels: int = DEFAULT_MIN_CROP_PIXELS,
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
        preserve_aspect: Letterbox instead of stretching to a square.
        min_crop_pixels: Skip crops whose short side is below this.

    Returns:
        ``(train_loader, validation_loader)``. The training loader shuffles and
        drops nothing; the validation loader preserves order.

    Raises:
        ValueError: If either split contains no images.
    """
    root = Path(dataset_root)
    shared = {"preserve_aspect": preserve_aspect, "min_crop_pixels": min_crop_pixels}
    train_dataset = CropDataset(root / train_split, classes, image_size, train=True, **shared)
    validation_dataset = CropDataset(root / validation_split, classes, image_size, train=False, **shared)

    for name, dataset in ((train_split, train_dataset), (validation_split, validation_dataset)):
        if len(dataset) == 0:
            raise ValueError(
                f"No images found in the '{name}' split under {root / name}. "
                f"Run scripts/generate_sample_crops.py to create placeholder crops."
            )
        if dataset.skipped_small or dataset.skipped_unreadable:
            print(
                f"[{name}] skipped {dataset.skipped_small} crops under {min_crop_pixels}px "
                f"and {dataset.skipped_unreadable} unreadable files"
            )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, validation_loader
