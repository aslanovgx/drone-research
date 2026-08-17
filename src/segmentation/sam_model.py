import cv2
import torch
import yaml

from .tiling import generate_masks_tiled


def load_config(path="configs/sam.yaml"):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(config):
    requested_device = (
        config.get("model", {})
        .get("device", "auto")
        .lower()
    )

    valid_devices = {"auto", "cuda", "mps", "cpu"}

    if requested_device not in valid_devices:
        raise ValueError(
            f"Unsupported device: {requested_device}. "
            f"Expected one of: {sorted(valid_devices)}"
        )

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but it is not available."
            )
        return "cuda"

    if requested_device == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if not mps_available:
            raise RuntimeError(
                "MPS was requested, but it is not available."
            )
        return "mps"

    if requested_device == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def load_sam_model(config):
    try:
        from sam2.automatic_mask_generator import (
            SAM2AutomaticMaskGenerator,
        )
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError(
            "SAM 2 is not installed. Clone the official SAM 2 "
            "repository into sam2_repo/ and run: "
            "python -m pip install -e sam2_repo"
        ) from exc

    device = resolve_device(config)

    sam2_model = build_sam2(
        config["model"]["config_path"],
        config["model"]["checkpoint_path"],
        device=device,
    )

    return SAM2AutomaticMaskGenerator(sam2_model)


def generate_masks_from_image(
    mask_generator,
    image,
    config=None,
):
    """Generate masks from an already loaded RGB image."""

    strategy = (
        (config or {})
        .get("preprocessing", {})
        .get("strategy", "resize")
    )

    if strategy == "tiling":
        tile_cfg = config["preprocessing"].get("tiling", {})
        filter_cfg = config.get("mask_filter", {})
        spatial_cfg = config.get("spatial_index", {})

        masks = generate_masks_tiled(
            mask_generator,
            image,
            tile_size=tile_cfg.get("tile_size", 1536),
            overlap=tile_cfg.get("overlap", 256),
            iou_threshold=tile_cfg.get(
                "iou_threshold",
                0.7,
            ),
            min_area=filter_cfg.get("min_area", 1500),
            min_stability_score=filter_cfg.get(
                "min_stability_score",
                0.9,
            ),
            min_predicted_iou=filter_cfg.get(
                "min_predicted_iou",
                0.85,
            ),
            reject_tile_edge=filter_cfg.get(
                "reject_tile_edge",
                False,
            ),
            edge_tolerance=filter_cfg.get(
                "edge_tolerance",
                2,
            ),
            cell_size=spatial_cfg.get(
                "cell_size",
                1536,
            ),
        )

        return image, masks

    max_size = (
        (config or {})
        .get("preprocessing", {})
        .get("resize_max_dim", 1536)
    )

    height, width = image.shape[:2]
    scale = min(max_size / width, max_size / height)

    if scale < 1:
        image = cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    masks = mask_generator.generate(image)
    return image, masks


def generate_masks(
    mask_generator,
    image_path,
    config=None,
):
    """Load an image and generate masks."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image could not be loaded: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return generate_masks_from_image(
        mask_generator,
        image,
        config,
    )