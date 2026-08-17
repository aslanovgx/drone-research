import torch
from sam2.build_sam import build_sam2

checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2_model = build_sam2(
    model_cfg,
    checkpoint,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("Model successfully loaded:", type(sam2_model))