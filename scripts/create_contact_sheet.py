"""Create a labeled contact sheet from exported SAM crops."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw


SEGMENT_PATTERN = re.compile(r"segment_(\d+)")


def segment_sort_key(path: Path) -> int:
    match = SEGMENT_PATTERN.search(path.stem)
    return int(match.group(1)) if match else 0


def create_contact_sheet(
    input_dir: Path,
    output_path: Path,
    columns: int = 7,
    thumbnail_size: int = 200,
) -> Path:
    crop_paths = sorted(
        input_dir.glob("*.png"),
        key=segment_sort_key,
    )

    if not crop_paths:
        raise ValueError(
            f"No PNG crops found in: {input_dir}"
        )

    rows = math.ceil(len(crop_paths) / columns)
    cell_width = thumbnail_size + 24
    cell_height = thumbnail_size + 48

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        color=(245, 245, 245),
    )

    draw = ImageDraw.Draw(sheet)

    for index, crop_path in enumerate(crop_paths):
        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        with Image.open(crop_path) as crop:
            crop = crop.convert("RGB")
            crop.thumbnail(
                (thumbnail_size, thumbnail_size),
                Image.Resampling.LANCZOS,
            )

            image_x = (
                x + (cell_width - crop.width) // 2
            )
            image_y = y + 8

            sheet.paste(
                crop,
                (image_x, image_y),
            )

        draw.text(
            (x + 8, y + thumbnail_size + 16),
            crop_path.name,
            fill=(0, 0, 0),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(
        output_path,
        quality=95,
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a contact sheet from SAM crops."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
    )
    parser.add_argument(
        "output",
        type=Path,
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=7,
    )

    arguments = parser.parse_args()

    output = create_contact_sheet(
        input_dir=arguments.input_dir,
        output_path=arguments.output,
        columns=arguments.columns,
    )

    print(f"Contact sheet saved: {output}")


if __name__ == "__main__":
    main()