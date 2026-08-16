import cv2
import yaml
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segmentation.tiling import generate_masks_tiled


def load_config(path="configs/sam.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_sam_model(config):
    sam2_model = build_sam2(
        config["model"]["config_path"],
        config["model"]["checkpoint_path"],
        device="cuda"
    )
    return SAM2AutomaticMaskGenerator(sam2_model)


def generate_masks_from_image(mask_generator, image, config=None):
    """Core function: takes an already-loaded RGB image array."""
    strategy = (config or {}).get("preprocessing", {}).get("strategy", "resize")

    if strategy == "tiling":
        tile_cfg = config["preprocessing"].get("tiling", {})
        filter_cfg = config.get("mask_filter", {})
        spatial_cfg = config.get("spatial_index", {})
        masks = generate_masks_tiled(
            mask_generator,
            image,
            tile_size=tile_cfg.get("tile_size", 1536),
            overlap=tile_cfg.get("overlap", 256),
            iou_threshold=tile_cfg.get("iou_threshold", 0.7),
            min_area=filter_cfg.get("min_area", 1500),
            min_stability_score=filter_cfg.get("min_stability_score", 0.9),
            min_predicted_iou=filter_cfg.get("min_predicted_iou", 0.85),
            reject_tile_edge=filter_cfg.get("reject_tile_edge", False),
            edge_tolerance=filter_cfg.get("edge_tolerance", 2),
            cell_size=spatial_cfg.get("cell_size", 1536),
        )
        return image, masks

    # resize strategy (default, used for video frames)
    max_size = (config or {}).get("preprocessing", {}).get("resize_max_dim", 1536)
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    masks = mask_generator.generate(image)
    return image, masks


def generate_masks(mask_generator, image_path, config=None):
    """Convenience wrapper: loads image from disk, then calls generate_masks_from_image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return generate_masks_from_image(mask_generator, image, config)