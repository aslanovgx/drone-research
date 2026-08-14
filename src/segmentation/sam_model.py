from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import yaml

def load_config(path="configs/sam.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_sam_model(checkpoint_path, model_type="vit_b"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    return SamAutomaticMaskGenerator(sam)

def generate_masks(mask_generator, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return image, masks