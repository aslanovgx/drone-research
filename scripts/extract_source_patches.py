from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


REQUIRED_COLUMNS = {
    "patch_id",
    "source_filename",
    "split",
    "x",
    "y",
    "width",
    "height",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract square source patches defined in a CSV manifest. "
            "Runs as a dry-run unless --execute is supplied."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="ArcGIS dataset root containing the Images directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/patch_selection.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/labeling/source_patches"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write patch images and the exported manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing patch images.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
    )

    return parser


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest contains no patches: {path}")

    return rows


def parse_positive_int(
    row: dict[str, str],
    field: str,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field!r} for patch {row.get('patch_id')!r}"
        ) from error

    minimum = 0 if allow_zero else 1

    if value < minimum:
        raise ValueError(
            f"{field} must be >= {minimum} for "
            f"patch {row.get('patch_id')!r}"
        )

    return value


def main() -> None:
    args = build_parser().parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    image_dir = dataset_root / "Images"
    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset Images directory not found: {image_dir}"
        )

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    rows = read_manifest(manifest_path)
    exported_rows: list[dict[str, str | int]] = []

    mode = "EXECUTE" if args.execute else "DRY RUN"

    print(f"Dataset root: {dataset_root}")
    print(f"Manifest    : {manifest_path}")
    print(f"Output root : {output_root}")
    print(f"Mode        : {mode}")
    print()

    written = 0
    skipped = 0

    for row in rows:
        patch_id = row["patch_id"].strip()
        split = row["split"].strip()
        source_filename = row["source_filename"].strip()

        if not patch_id:
            raise ValueError("patch_id cannot be empty.")

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unsupported split {split!r} for {patch_id!r}"
            )

        x = parse_positive_int(row, "x", allow_zero=True)
        y = parse_positive_int(row, "y", allow_zero=True)
        width = parse_positive_int(row, "width")
        height = parse_positive_int(row, "height")

        source_path = image_dir / source_filename
        patch_directory = output_root / split / patch_id
        patch_path = patch_directory / "source.jpg"

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source image not found: {source_path}"
            )

        with Image.open(source_path) as source:
            source.seek(0)
            source_width, source_height = source.size

            right = x + width
            bottom = y + height

            if right > source_width or bottom > source_height:
                raise ValueError(
                    f"Patch {patch_id!r} is outside source bounds: "
                    f"crop=({x}, {y}, {right}, {bottom}), "
                    f"source={source.size}"
                )

            if args.execute:
                if patch_path.exists() and not args.overwrite:
                    status = "skipped"
                    skipped += 1
                else:
                    patch = source.crop(
                        (x, y, right, bottom)
                    ).convert("RGB")

                    patch_directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    patch.save(
                        patch_path,
                        format="JPEG",
                        quality=args.jpeg_quality,
                    )

                    status = "written"
                    written += 1
            else:
                status = "planned"

        print(
            f"[{split}] {patch_id}: "
            f"{source_filename} "
            f"({x}, {y}, {width}, {height}) "
            f"-> {patch_path} [{status}]"
        )

        exported_rows.append(
            {
                **row,
                "patch_path": patch_path.as_posix(),
                "status": status,
            }
        )

    if args.execute:
        exported_manifest = output_root / "extracted_manifest.csv"
        exported_manifest.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(exported_rows[0].keys())

        with exported_manifest.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(exported_rows)

        print()
        print(f"Written : {written}")
        print(f"Skipped : {skipped}")
        print(f"Manifest: {exported_manifest}")
    else:
        print()
        print(
            f"Dry run complete: {len(rows)} patches planned. "
            "No files were written."
        )


if __name__ == "__main__":
    main()