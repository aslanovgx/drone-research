from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.bbox_extractor import export_segments
from src.segmentation.sam_model import (
    generate_masks,
    load_config,
    load_sam_model,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SAM segments and crops for extracted "
            "classifier-labeling patches."
        )
    )

    parser.add_argument(
        "--patch-root",
        type=Path,
        default=Path("outputs/labeling/source_patches"),
    )
    parser.add_argument(
        "--sam-config",
        type=Path,
        default=Path("configs/sam.yaml"),
    )
    parser.add_argument(
        "--patch-id",
        action="append",
        default=[],
        help="Process only this patch ID. May be supplied repeatedly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def draw_overlay(
    image,
    segments: list[dict],
    output_path: Path,
) -> None:
    overlay = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    for segment in segments:
        bbox = segment["bbox"]

        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])

        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 255, 255),
            thickness=3,
        )

        cv2.putText(
            overlay,
            str(segment["segment_id"]),
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(
            f"Overlay could not be saved: {output_path}"
        )


def discover_patch_directories(
    patch_root: Path,
    selected_ids: set[str],
) -> list[Path]:
    directories = []

    for source_path in sorted(
        patch_root.glob("*/*/source.jpg")
    ):
        patch_directory = source_path.parent

        if (
            selected_ids
            and patch_directory.name not in selected_ids
        ):
            continue

        directories.append(patch_directory)

    return directories


def main() -> None:
    args = build_parser().parse_args()

    patch_root = args.patch_root.resolve()
    selected_ids = set(args.patch_id)

    patch_directories = discover_patch_directories(
        patch_root,
        selected_ids,
    )

    if not patch_directories:
        raise RuntimeError(
            f"No matching source patches found in {patch_root}"
        )

    config = load_config(str(args.sam_config))
    device = resolve_device(config)

    print(f"Patch root : {patch_root}")
    print(f"Patch count: {len(patch_directories)}")
    print(f"Device      : {device}")
    print("Loading SAM2 model...")

    mask_generator = load_sam_model(config)

    print("SAM2 model loaded.")
    print()

    summary_rows = []

    for position, patch_directory in enumerate(
        patch_directories,
        start=1,
    ):
        patch_id = patch_directory.name
        split = patch_directory.parent.name

        source_path = patch_directory / "source.jpg"
        crops_dir = patch_directory / "crops"
        segments_path = patch_directory / "segments.json"
        overlay_path = patch_directory / "overlay.jpg"

        if segments_path.exists() and not args.overwrite:
            print(
                f"[{position}/{len(patch_directories)}] "
                f"{patch_id}: skipped (already processed)"
            )

            summary_rows.append(
                {
                    "patch_id": patch_id,
                    "split": split,
                    "status": "skipped",
                    "mask_count": "",
                    "segment_count": "",
                    "seconds": "",
                }
            )
            continue

        if args.overwrite and crops_dir.is_dir():
            for crop_path in crops_dir.glob(
                "segment_*.png"
            ):
                crop_path.unlink()

        print(
            f"[{position}/{len(patch_directories)}] "
            f"{patch_id}: generating masks..."
        )

        started = time.perf_counter()

        processed_image, masks = generate_masks(
            mask_generator=mask_generator,
            image_path=str(source_path),
            config=config,
        )

        segments = export_segments(
            image=processed_image,
            masks=masks,
            crops_dir=str(crops_dir),
            json_path=str(segments_path),
        )

        draw_overlay(
            image=processed_image,
            segments=segments,
            output_path=overlay_path,
        )

        elapsed = time.perf_counter() - started

        print(
            f"  masks={len(masks)} "
            f"segments={len(segments)} "
            f"time={elapsed:.1f}s"
        )

        summary_rows.append(
            {
                "patch_id": patch_id,
                "split": split,
                "status": "written",
                "mask_count": len(masks),
                "segment_count": len(segments),
                "seconds": round(elapsed, 2),
            }
        )

    summary_path = patch_root / "sam_batch_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patch_id",
                "split",
                "status",
                "mask_count",
                "segment_count",
                "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()