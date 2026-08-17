import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from PIL import ExifTags  # noqa: E402


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Read-only analysis of the drone image dataset. Never modifies the dataset."
    )
    parser.add_argument(
        "--dataset-root",
        help=f"Path to the dataset root directory (overrides {common.DEFAULT_DATASET_ENV}).",
    )
    parser.add_argument(
        "--config",
        help="Path to configs/data.yaml (default: <repo>/configs/data.yaml).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write a JSON report. Must be outside the repository and the dataset.",
    )
    parser.add_argument(
        "--full-decode",
        action="store_true",
        help="Fully decode every image to check pixel-level corruption (slow, reads the whole dataset).",
    )
    return parser


def analyze_files(files, full_decode=False):
    ext_counter = Counter()
    dim_counter = Counter()
    mode_counter = Counter()
    format_counter = Counter()
    aspect_counter = Counter()
    frame_count_counter = Counter()
    exif_field_counter = Counter()
    orientation_counter = Counter()
    gps_count = 0
    sizes = {}
    open_errors = {}
    corrupted = []

    for path in files:
        ext_counter[path.suffix.lower()] += 1
        sizes[path.name] = path.stat().st_size
        try:
            with common.open_frame(path, 0) as im:
                w, h = im.size
                dim_counter[(w, h)] += 1
                mode_counter[im.mode] += 1
                format_counter[im.format] += 1
                aspect_counter[round(w / h, 3)] += 1
                frame_count_counter[getattr(im, "n_frames", 1)] += 1

                exif = im.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_field_counter[tag_name] += 1
                        if tag_name == "Orientation":
                            orientation_counter[value] += 1
                    gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else None
                    if gps_ifd:
                        gps_count += 1

                if full_decode:
                    im.load()
        except Exception as exc:
            open_errors[path.name] = str(exc)

    for path in files:
        try:
            with common.open_frame(path, 0) as im:
                im.verify()
        except Exception as exc:
            corrupted.append({"file": path.name, "error": str(exc)})

    by_size = defaultdict(list)
    for name, size in sizes.items():
        by_size[size].append(name)
    size_duplicates = {size: names for size, names in by_size.items() if len(names) > 1}

    name_counts = Counter(p.name for p in files)
    filename_duplicates = {name: c for name, c in name_counts.items() if c > 1}

    return {
        "image_count": len(files),
        "extension_counts": dict(ext_counter),
        "total_size_bytes": sum(sizes.values()),
        "per_image_size_bytes": sizes,
        "dimension_distribution": {f"{w}x{h}": c for (w, h), c in dim_counter.most_common()},
        "aspect_ratio_distribution": {str(k): v for k, v in aspect_counter.most_common()},
        "mode_distribution": dict(mode_counter),
        "format_distribution": dict(format_counter),
        "mpo_frame_count_distribution": {str(k): v for k, v in frame_count_counter.items()},
        "exif_field_coverage": dict(exif_field_counter.most_common()),
        "gps_coverage_count": gps_count,
        "orientation_distribution": {str(k): v for k, v in orientation_counter.items()},
        "open_errors": open_errors,
        "corrupted_files": corrupted,
        "filename_duplicates": filename_duplicates,
        "size_duplicates": size_duplicates,
        "full_decode_used": full_decode,
    }


def print_summary(report, dataset_root, image_dir):
    print(f"Dataset root : {dataset_root}")
    print(f"Image dir    : {image_dir}")
    print(f"Image count  : {report['image_count']}")
    print(f"Extensions   : {report['extension_counts']}")
    print(f"Total size   : {report['total_size_bytes'] / (1024 ** 3):.2f} GB")
    print(f"Dimensions   : {report['dimension_distribution']}")
    print(f"Aspect ratios: {report['aspect_ratio_distribution']}")
    print(f"Modes        : {report['mode_distribution']}")
    print(f"Formats      : {report['format_distribution']}")
    print(f"MPO frames   : {report['mpo_frame_count_distribution']}")
    print(f"GPS coverage : {report['gps_coverage_count']}/{report['image_count']}")
    print(f"Orientation  : {report['orientation_distribution']}")
    print(f"Open errors  : {len(report['open_errors'])}")
    print(f"Corrupted    : {len(report['corrupted_files'])} (header-level check, full_decode={report['full_decode_used']})")
    print(f"Filename dupes: {len(report['filename_duplicates'])}")
    print(f"Size dupes   : {len(report['size_duplicates'])}")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    config_path = Path(args.config) if args.config else common.repo_root() / "configs" / "data.yaml"
    config = common.load_config(config_path)

    dataset_root = common.resolve_root(args.dataset_root, common.DEFAULT_DATASET_ENV, "Dataset root")
    if not dataset_root.is_dir():
        raise common.DatasetError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    image_dir = common.resolve_safe_subdir(dataset_root, config["dataset"]["image_subdir"], "dataset.image_subdir")
    accepted_ext = config["dataset"]["accepted_extensions"]
    excluded = config["dataset"].get("excluded_files", [])

    files = common.list_dataset_images(image_dir, accepted_ext, excluded)
    if not files:
        raise common.DatasetError(f"No images with extensions {accepted_ext} found in {image_dir}")

    report = analyze_files(files, full_decode=args.full_decode)
    print_summary(report, dataset_root, image_dir)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        common.check_safe_output_root(output_path.parent, dataset_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except common.DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
