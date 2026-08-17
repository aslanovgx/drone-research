import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

RESAMPLE = Image.Resampling.LANCZOS


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Select and preprocess a subset of drone images. Runs in dry-run mode "
            "(no files written) unless --execute is given. Never modifies raw images."
        )
    )
    parser.add_argument(
        "--dataset-root",
        help=f"Path to the dataset root directory (overrides {common.DEFAULT_DATASET_ENV}).",
    )
    parser.add_argument(
        "--output-root",
        help=(
            f"Path to write generated outputs (overrides {common.DEFAULT_OUTPUT_ENV}). "
            "Must be outside the repository and outside the raw dataset."
        ),
    )
    parser.add_argument("--config", help="Path to configs/data.yaml (default: <repo>/configs/data.yaml).")
    parser.add_argument("--execute", action="store_true", help="Actually write processed images and the manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (overrides config).")
    parser.add_argument("--selection-method", choices=["stratified", "random"], help="Override selection.method.")
    parser.add_argument("--count", type=int, help="Override selection.count.")
    parser.add_argument("--seed", type=int, help="Override selection.seed.")
    parser.add_argument(
        "--preprocess-method",
        choices=["letterbox", "center_crop", "stretch", "none"],
        help="Override preprocessing.method.",
    )
    parser.add_argument("--manifest-format", choices=["json", "csv"], default="json", help="Manifest file format.")
    return parser


def transform_letterbox(im, width, height, bg_color):
    src_w, src_h = im.size
    scale = min(width / src_w, height / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = im.resize((new_w, new_h), RESAMPLE)
    canvas = Image.new("RGB", (width, height), tuple(bg_color))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas.paste(resized, (left, top))
    info = {
        "scale": scale,
        "resized_width": new_w,
        "resized_height": new_h,
        "pad_left": left,
        "pad_top": top,
        "pad_right": width - new_w - left,
        "pad_bottom": height - new_h - top,
    }
    return canvas, info


def transform_center_crop(im, width, height):
    src_w, src_h = im.size
    target_ratio = width / height
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        crop_h = src_h
        crop_w = round(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = round(crop_w / target_ratio)
    left = (src_w - crop_w) // 2
    top = (src_h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    cropped = im.crop((left, top, right, bottom))
    resized = cropped.resize((width, height), RESAMPLE)
    info = {
        "crop_box": [left, top, right, bottom],
        "scale": width / crop_w,
    }
    return resized, info


def transform_stretch(im, width, height):
    src_w, src_h = im.size
    resized = im.resize((width, height), RESAMPLE)
    info = {"scale_x": width / src_w, "scale_y": height / src_h}
    return resized, info


def transform_none(im):
    return im.copy(), {"scale": 1.0}


def apply_preprocessing(im, method, width, height, bg_color):
    if method == "letterbox":
        return transform_letterbox(im, width, height, bg_color)
    if method == "center_crop":
        return transform_center_crop(im, width, height)
    if method == "stretch":
        return transform_stretch(im, width, height)
    if method == "none":
        return transform_none(im)
    raise common.DatasetError(f"Unknown preprocessing method: {method!r}.")


SUPPORTED_OUTPUT_FORMATS = {"jpg", "jpeg"}
OUTPUT_FORMAT_LABELS = {"jpg": "JPEG", "jpeg": "JPEG"}


def validate_output_format(output_format):
    normalized = str(output_format).strip().lower()
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise common.DatasetError(
            f"preprocessing.output_format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}, "
            f"got {output_format!r}. This component's contract is standard JPEG output only."
        )
    return normalized


def describe_output_encoding(source_format, exported_frame_index, output_format):
    fmt_label = OUTPUT_FORMAT_LABELS.get(output_format, output_format.upper())
    source_label = source_format or "unknown"
    return f"standard single-frame {fmt_label} re-encoded from source {source_label} frame {exported_frame_index}"


def plan_output_filenames(selected, output_format):
    seen = {}
    out_names = []
    for item in selected:
        src_path = item["file"]
        out_name = f"{src_path.stem}.{output_format}"
        key = out_name.lower()
        if key in seen:
            raise common.DatasetError(
                f"Output filename collision: '{seen[key]}' and '{src_path.name}' would both "
                f"produce output filename '{out_name}' (case-insensitive match on this "
                f"filesystem: '{key}'). Refusing to proceed; resolve the collision (e.g. rename "
                "or exclude one of the source files) before running preprocessing."
            )
        seen[key] = src_path.name
        out_names.append(out_name)
    return out_names


def build_manifest_entries(selected, dataset_root, output_root, config, execute, overwrite, seed):
    preprocessing_cfg = config["preprocessing"]
    method = preprocessing_cfg["method"]
    width = preprocessing_cfg["width"]
    height = preprocessing_cfg["height"]
    bg_color = preprocessing_cfg.get("background_color", [0, 0, 0])
    output_format = validate_output_format(preprocessing_cfg.get("output_format", "jpg"))
    quality = preprocessing_cfg.get("jpeg_quality", 95)
    mpo_frame = preprocessing_cfg.get("mpo_frame", 0)
    output_subdir = preprocessing_cfg.get("output_subdir", "selected_images")
    selection_method_name = config["selection"]["method_used"]

    processed_dir = common.resolve_safe_subdir(
        output_root, output_subdir, "preprocessing.output_subdir", allow_root=False
    )
    out_names = plan_output_filenames(selected, output_format)
    entries = []

    for item, out_name in zip(selected, out_names):
        src_path = item["file"]
        rel_source = src_path.resolve().relative_to(dataset_root.resolve()).as_posix()
        out_path = processed_dir / out_name
        rel_output = (processed_dir / out_name).relative_to(output_root.resolve()).as_posix()

        try:
            with common.open_frame(src_path, mpo_frame) as raw_im:
                raw_format = raw_im.format
                raw_frame_count = getattr(raw_im, "n_frames", 1)
                oriented = ImageOps.exif_transpose(raw_im).convert("RGB")
                source_w, source_h = oriented.size

                if method == "none":
                    out_im, transform_info = transform_none(oriented)
                else:
                    out_im, transform_info = apply_preprocessing(oriented, method, width, height, bg_color)
                out_w, out_h = out_im.size
        except common.DatasetError:
            raise
        except Exception as exc:
            raise common.DatasetError(f"Could not read or transform image {rel_source}: {exc}") from exc

        entry = {
            "source_relative_path": rel_source,
            "source_filename": src_path.name,
            "output_relative_path": rel_output,
            "selection_index": item["index"],
            "selection_bin": item["bin"],
            "selection_method": selection_method_name,
            "selection_seed": seed,
            "source_width": source_w,
            "source_height": source_h,
            "output_width": out_w,
            "output_height": out_h,
            "source_format": raw_format,
            "source_frame_count": raw_frame_count,
            "exported_frame_index": mpo_frame,
            "output_format": output_format,
            "preprocessing_method": method,
            "output_encoding": describe_output_encoding(raw_format, mpo_frame, output_format),
            "transform": transform_info,
            "status": "planned",
        }

        if execute:
            processed_dir.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                entry["status"] = "skipped_existing"
            else:
                save_kwargs = {"quality": quality} if output_format in ("jpg", "jpeg") else {}
                out_im.save(out_path, **save_kwargs)
                entry["status"] = "written"

        entries.append(entry)

    return entries


def write_manifest(entries, output_root, fmt):
    output_root.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        manifest_path = output_root / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    elif fmt == "csv":
        manifest_path = output_root / "manifest.csv"
        fieldnames = [
            "source_relative_path", "source_filename", "output_relative_path",
            "selection_index", "selection_bin", "selection_method", "selection_seed",
            "source_width", "source_height", "output_width", "output_height",
            "source_format", "source_frame_count", "exported_frame_index", "output_format",
            "preprocessing_method", "output_encoding", "status", "transform",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                row = dict(e)
                row["transform"] = json.dumps(e["transform"])
                writer.writerow(row)
    else:
        raise common.DatasetError(f"Unknown manifest format: {fmt!r}.")
    return manifest_path


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    config_path = Path(args.config) if args.config else common.repo_root() / "configs" / "data.yaml"
    config = common.load_config(config_path)

    dataset_root = common.resolve_root(args.dataset_root, common.DEFAULT_DATASET_ENV, "Dataset root")
    if not dataset_root.is_dir():
        raise common.DatasetError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    output_root = common.resolve_root(args.output_root, common.DEFAULT_OUTPUT_ENV, "Output root")
    common.check_safe_output_root(output_root, dataset_root)

    image_dir = common.resolve_safe_subdir(dataset_root, config["dataset"]["image_subdir"], "dataset.image_subdir")
    accepted_ext = config["dataset"]["accepted_extensions"]
    excluded = config["dataset"].get("excluded_files", [])
    files = common.list_dataset_images(image_dir, accepted_ext, excluded)
    if not files:
        raise common.DatasetError(f"No images with extensions {accepted_ext} found in {image_dir}")

    selection_cfg = config["selection"]
    method = args.selection_method or selection_cfg["method"]
    count = args.count if args.count is not None else selection_cfg["count"]
    seed = args.seed if args.seed is not None else selection_cfg["seed"]
    selection_cfg["method_used"] = method

    if args.preprocess_method:
        config["preprocessing"]["method"] = args.preprocess_method
    overwrite = args.overwrite or config["preprocessing"].get("overwrite", False)

    selected = common.select_images(files, method, count, seed)

    print(f"Dataset root : {dataset_root}")
    print(f"Image dir    : {image_dir}")
    print(f"Output root  : {output_root}")
    print(f"Selection    : method={method} count={count} seed={seed}")
    preprocess_method = config["preprocessing"]["method"]
    if preprocess_method == "none":
        print("Preprocessing: method=none (original resolution preserved, width/height ignored)")
    else:
        print(
            f"Preprocessing: method={preprocess_method} "
            f"size={config['preprocessing']['width']}x{config['preprocessing']['height']}"
        )
    print(f"Mode         : {'EXECUTE' if args.execute else 'DRY RUN (no files will be written)'}")
    print()

    entries = build_manifest_entries(
        selected, dataset_root, output_root, config, execute=args.execute, overwrite=overwrite, seed=seed
    )

    for e in entries:
        print(
            f"  [{e['selection_method']} idx={e['selection_index']} bin={e['selection_bin']}] "
            f"{e['source_relative_path']} ({e['source_width']}x{e['source_height']}) -> "
            f"{e['output_relative_path']} ({e['output_width']}x{e['output_height']}) [{e['status']}]"
        )

    if args.execute:
        manifest_path = write_manifest(entries, output_root, args.manifest_format)
        written = sum(1 for e in entries if e["status"] == "written")
        skipped = sum(1 for e in entries if e["status"] == "skipped_existing")
        print(f"\nManifest written to: {manifest_path}")
        print(f"Written: {written}, Skipped (existing): {skipped}")
    else:
        print("\nDry run complete. No output images or manifest were written. Re-run with --execute to write outputs.")

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
