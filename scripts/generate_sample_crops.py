"""Generate synthetic placeholder crops for the classifier dataset.

The real crops come from the SAM stage (``outputs/crops/segment_<id>.png``) and
are not labelled yet. These placeholders exist purely to exercise the dataset
loader, training loop and inference path end to end; they are simple shapes,
colours and noise and carry no semantic meaning.

Everything written under ``data/classifier/`` is gitignored — see .gitignore.

Usage::

    python scripts/generate_sample_crops.py
    python scripts/generate_sample_crops.py --config configs/classifier.yaml
    python scripts/generate_sample_crops.py --per-class 10 --seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

# Fallbacks used when no config file is supplied (keeps the script standalone).
DEFAULT_CLASSES: tuple[str, ...] = ("building", "tree", "car", "other")
DEFAULT_DATASET_ROOT = Path("data/classifier")
SPLITS: tuple[str, ...] = ("train", "validation")

# Per-class base colour (RGB), only so the four classes are trivially separable
# and a smoke-test run shows the loss actually moving.
CLASS_PALETTE: dict[str, tuple[int, int, int]] = {
    "building": (150, 150, 160),
    "tree": (40, 110, 55),
    "car": (60, 90, 180),
    "other": (140, 120, 90),
}

# Typical crop shape per class, as (short_side_px, aspect_ratio), measured against
# the Packing House District frames at ~1.4 cm/px: a car is roughly 330x150 px,
# roofs and canopies are far larger and closer to square, and `other` is whatever
# fragment SAM produced. Real aspect ratios matter because the loader letterboxes
# rather than stretching, so square placeholders would not exercise that path.
CLASS_SHAPE: dict[str, tuple[int, float]] = {
    "building": (420, 1.25),
    "tree": (300, 1.10),
    "car": (150, 2.20),
    "other": (120, 1.80),
}


def _jitter(colour: Sequence[int], rng: random.Random, amount: int = 25) -> tuple[int, int, int]:
    """Randomly perturb an RGB colour, clamped to the valid range."""
    return tuple(int(np.clip(c + rng.randint(-amount, amount), 0, 255)) for c in colour)  # type: ignore[return-value]


def _crop_dimensions(class_name: str, scale: float, rng: random.Random) -> tuple[int, int]:
    """Pick a plausible (width, height) for a crop of this class."""
    short_side, aspect = CLASS_SHAPE.get(class_name, (150, 1.5))
    short = max(16, round(short_side * scale * rng.uniform(0.7, 1.3)))
    long = max(16, round(short * aspect * rng.uniform(0.85, 1.15)))
    return (long, short) if rng.random() < 0.5 else (short, long)  # nadir crops have no fixed orientation


def make_synthetic_crop(class_name: str, size: int, rng: random.Random) -> Image.Image:
    """Build one synthetic crop for ``class_name``.

    The crop is **not** square: its dimensions follow the per-class shapes in
    :data:`CLASS_SHAPE`, so the letterboxing path in the dataset loader is
    exercised the same way real SAM crops will exercise it.

    Args:
        class_name: One of the classifier classes; selects the base palette,
            shape and typical proportions.
        size: Nominal crop size; used to scale the per-class dimensions.
        rng: Seeded RNG so runs are reproducible.

    Returns:
        An RGB :class:`PIL.Image.Image` with class-appropriate proportions.
    """
    base = CLASS_PALETTE.get(class_name, (128, 128, 128))
    width, height = _crop_dimensions(class_name, size / 224, rng)

    noise = np.random.default_rng(rng.randrange(2**32)).integers(-18, 18, (height, width, 3))
    canvas = np.clip(np.array(base, dtype=np.int16) + noise, 0, 255).astype(np.uint8)

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    fill = _jitter(base, rng, amount=60)
    m = max(2, min(width, height) // 8)  # margin

    if class_name == "building":
        draw.rectangle([m, m, width - m, height - m], fill=fill, outline=(30, 30, 30), width=2)
    elif class_name == "tree":
        draw.ellipse([m, m, width - m, height - m], fill=fill, outline=(20, 60, 25), width=2)
    elif class_name == "car":
        # Fills the crop, so the drawn object keeps the crop's elongated proportions.
        draw.rounded_rectangle(
            [m, m, width - m, height - m], radius=m, fill=fill, outline=(20, 20, 40), width=2
        )
    else:  # "other" and any future class: scattered fragments, deliberately messy
        for _ in range(rng.randint(3, 7)):
            x0, y0 = rng.randint(0, max(1, width - m)), rng.randint(0, max(1, height - m))
            draw.rectangle([x0, y0, x0 + rng.randint(m, 3 * m), y0 + rng.randint(m, 3 * m)], fill=_jitter(base, rng, 70))

    return image


def generate(
    dataset_root: Path,
    classes: Sequence[str],
    per_class: int,
    size: int,
    seed: int,
) -> int:
    """Write synthetic crops for every split/class pair.

    Args:
        dataset_root: Root of the classifier dataset (``data/classifier``).
        classes: Class names; one directory is created per class per split.
        per_class: Number of crops per class in the ``train`` split. The
            ``validation`` split gets roughly a third of that (minimum 3).
        size: Crop width/height in pixels.
        seed: RNG seed for reproducibility.

    Returns:
        Total number of files written.
    """
    rng = random.Random(seed)
    written = 0
    for split in SPLITS:
        count = per_class if split == "train" else max(3, per_class // 3)
        for class_name in classes:
            out_dir = dataset_root / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                # Mirror the SAM naming convention so downstream id parsing is exercised.
                path = out_dir / f"segment_{index}.png"
                make_synthetic_crop(class_name, size, rng).save(path)
                written += 1
    return written


def _load_config(config_path: Path | None) -> tuple[Path, Sequence[str], int]:
    """Read dataset root, classes and image size from the YAML config if given."""
    if config_path is None:
        return DEFAULT_DATASET_ROOT, DEFAULT_CLASSES, 64

    import yaml  # imported lazily so the script runs without a config

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config.get("data", {})
    return (
        Path(data.get("dataset_root", DEFAULT_DATASET_ROOT)),
        tuple(config.get("classes", DEFAULT_CLASSES)),
        int(data.get("image_size", 64)),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="Optional configs/classifier.yaml to read paths from")
    parser.add_argument("--per-class", type=int, default=8, help="Train crops per class (default: 8)")
    parser.add_argument("--size", type=int, default=None, help="Crop size in px (default: config image size or 64)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = parser.parse_args()

    dataset_root, classes, config_size = _load_config(args.config)
    size = args.size or config_size

    total = generate(dataset_root, classes, args.per_class, size, args.seed)
    print(f"Wrote {total} synthetic crops ({size}x{size}) under {dataset_root}/{{{','.join(SPLITS)}}}/<class>/")
    print("These are placeholders only — delete them before training on real SAM crops.")


if __name__ == "__main__":
    main()
