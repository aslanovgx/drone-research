import sys
sys.path.append("src")

from segmentation.sam_model import load_config, load_sam_model, generate_masks

config = load_config()
mask_generator = load_sam_model(config)

image, masks = generate_masks(mask_generator, "data/samples/sample_02.jpg", config=config)

print(f"Tapılan mask sayı: {len(masks)}")
print("İlk mask açarları:", masks[0].keys())
print("İlk mask sahəsi:", masks[0]["area"])