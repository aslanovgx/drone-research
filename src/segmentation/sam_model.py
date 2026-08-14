# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# import cv2
# import yaml
#
# def load_config(path="configs/sam.yaml"):
#     with open(path) as f:
#         return yaml.safe_load(f)
#
# def load_sam_model(checkpoint_path, model_type="vit_b"):
#     sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
#     return SamAutomaticMaskGenerator(sam)
#
# def generate_masks(mask_generator, image_path):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     masks = mask_generator.generate(image)
#     return image, masks
# import cv2
# import yaml
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from segmentation.tiling import generate_masks_tiled
#
#
# def load_config(path="configs/sam.yaml"):
#     with open(path) as f:
#         return yaml.safe_load(f)
#
#
# def load_sam_model(config):
#     sam2_model = build_sam2(
#         config["model"]["config_path"],
#         config["model"]["checkpoint_path"],
#         device="cuda"
#     )
#     return SAM2AutomaticMaskGenerator(sam2_model)
#
#
# def generate_masks(mask_generator, image_path, config=None):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     strategy = (config or {}).get("preprocessing", {}).get("strategy", "resize")
#
#     if strategy == "tiling":
#         tile_cfg = config["preprocessing"].get("tiling", {})
#         masks = generate_masks_tiled(
#             mask_generator,
#             image,
#             tile_size=tile_cfg.get("tile_size", 1536),
#             overlap=tile_cfg.get("overlap", 256),
#             iou_threshold=tile_cfg.get("iou_threshold", 0.7),
#         )
#         return image, masks
#
#     # default: resize strategy
#     max_size = (config or {}).get("preprocessing", {}).get("resize_max_dim", 1536)
#     h, w = image.shape[:2]
#     scale = min(max_size / w, max_size / h)
#     if scale < 1:
#         image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
#
#     masks = mask_generator.generate(image)
#     return image, masks

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
        masks = generate_masks_tiled(
            mask_generator,
            image,
            tile_size=tile_cfg.get("tile_size", 1536),
            overlap=tile_cfg.get("overlap", 256),
            iou_threshold=tile_cfg.get("iou_threshold", 0.7),
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