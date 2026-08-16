# SAM 2.1 Drone Object Segmentation & Temporal Tracking

A high-performance computer vision pipeline for automated object segmentation and temporal tracking on high-resolution drone orthomosaics and aerial video using **Meta SAM 2.1 (Segment Anything Model 2.1)**.

---

## 1. Project Scope & Responsibility

This project is strictly focused on **SAM 2.1 segmentation, spatial optimization, and temporal video propagation**:
- **Static High-Resolution Imagery:** Spatial tiling, localized mask representation, spatial grid indexing ($O(N+K)$), multi-condition tile-boundary stitching with Disjoint-Set Union (DSU), quality filtering, and 224×224 crop generation.
- **Drone Video Streams:** Unbroken sequential frame extraction, Frame 0 automatic object discovery, native `SAM2VideoPredictor` temporal memory propagation, track ID continuity, and coordinate scaling back to original video dimensions.
- **Output:** Bounding boxes, segmentation masks, 224×224 PNG crops, and structured JSON metadata.

> **SCOPE BOUNDARY:** Object classification (ResNet, EfficientNet, ViT, or semantic class assignment) is **NOT** part of this project. The generated 224×224 crops and JSON metadata serve as the final output interface consumed by downstream systems.

---

## 2. Architecture & Pipelines

### A. Static High-Resolution Image Pipeline
```
High-Resolution Orthophoto (~60MP / 9504×6336)
       │
       ▼
1. Overlapping Tiling (1536×1536 tiles with 256px overlap)
       │
       ▼
2. SAM 2.1 Automatic Mask Generation per Tile
       │
       ▼
3. Localized Mask Representation (Bounding-box cropped boolean masks with global offset)
       │
       ▼
4. Initial Quality Filtering (min_area ≥ 1500, stability_score ≥ 0.90, predicted_iou ≥ 0.85)
       │
       ▼
5. Spatial Grid Indexing & Local Deduplication (iou_threshold = 0.70)
       │
       ▼
6. Boundary-Aware Stitching across Adjacent Tiles (IoS ≥ 0.40, max_gap = 15px, DSU merging)
       │
       ▼
7. Internal Tile-Edge Artifact Filtering (preserves merged & image-border objects)
       │
       ▼
8. Global BBox Extraction, 224×224 PNG Crop Generation, & JSON Export
```

### B. Temporal Video Pipeline (`SAM2VideoPredictor`)
```
Input Video Stream / MP4 File
       │
       ▼
1. Sequential Frame Extraction (Unbroken sequence: 000000.jpg, 000001.jpg, 000002.jpg...)
       │
       ▼
2. Frame 0 Automatic Object Discovery (SAM 2.1 + Quality Filter + Deduplication)
       │ (Fails fast if 0 objects found; no hardcoded fallback boxes)
       ▼
3. Initial Prompt Registration (SAM2VideoPredictor.add_new_points_or_box on Frame 0)
       │ (Deterministic track IDs: 1, 2, 3...)
       ▼
4. Continuous Temporal Propagation (SAM2VideoPredictor.propagate_in_video across ALL frames)
       │ (VRAM Safety: offload_video_to_cpu=True, offload_state_to_cpu=True)
       ▼
5. Output Export Interval Filtering (e.g., export records every output_interval = 30 frames)
       │
       ▼
6. Coordinate Scaling & Clamping (scale_x, scale_y back to ORIGINAL video resolution)
       │
       ▼
7. Video JSON Metadata Export (outputs/video_results.json)
```

---

## 3. Repository Structure

```text
drone-research/
├── checkpoints/
│   └── sam2.1_hiera_small.pt        # Pretrained SAM 2.1 Hiera-Small weights (~184MB)
├── configs/
│   ├── sam.yaml                     # Primary project configuration
│   └── sam2.1/
│       └── sam2.1_hiera_s.yaml      # SAM 2.1 Hiera-Small model architecture config
├── data/
│   ├── raw/                         # Raw aerial datasets (Esri Packing House District, etc.)
│   └── samples/                     # Test orthophoto samples (sample_01.jpg .. sample_13.jpg)
├── docs/
│   └── sam_setup.md                 # In-depth technical architecture documentation
├── outputs/
│   ├── crops/                       # Generated 224×224 PNG object crops
│   ├── segmentation_results.json    # Static pipeline JSON export
│   └── video_results.json           # Video tracking JSON export
├── sam2_repo/                       # Official SAM 2 submodule source tree
├── scripts/
│   └── test_mask_generation.py      # End-to-end static image segmentation & visualization
├── src/
│   └── segmentation/
│       ├── __init__.py
│       ├── bbox_extractor.py        # Global bbox extraction, crop generation, JSON export
│       ├── mask_filter.py           # Quality filtering and boundary artifact filtering
│       ├── sam_model.py             # SAM 2.1 model builder & image inference dispatch
│       ├── tiling.py                # Tiling, SpatialGridIndex, DSU boundary stitching, dedup
│       └── video_processing.py      # Frame 0 auto-discovery & SAM2VideoPredictor pipeline
├── tests/
│   ├── test_tiling_pipeline.py      # 15 unit tests for tiling, localization, stitching
│   ├── test_spatial_index.py        # 8 unit tests for SpatialGridIndex candidate queries
│   ├── test_performance_equivalence.py # 2 unit tests verifying 100% equivalence & candidate reduction
│   ├── test_video_pipeline.py       # 7 unit tests for video math, Frame 0 discovery, error handling
│   └── test_video_integration.py    # 1 real CUDA integration test with SAM2VideoPredictor
├── requirements.txt                 # Project dependencies
├── test_sam2.py                    # Model loading verification script
└── README.md
```

---

## 4. Hardware & Environment Requirements

- **Operating System:** Windows 10/11 (PowerShell) or Linux (Ubuntu 20.04+)
- **GPU:** NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (Tested on NVIDIA RTX 3050 Laptop GPU 6GB)
- **CUDA:** CUDA 11.8 / 12.1+
- **Python:** Python 3.10 – 3.12
- **PyTorch:** PyTorch 2.1.0+ with CUDA support

---

## 5. Installation Guide

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone <repo-url> drone-research
cd drone-research

# 2. Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA support (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install SAM 2 repository in editable mode
pip install -e sam2_repo

# 5. Install project dependencies
pip install -r requirements.txt
```

### Linux (Bash)

```bash
# 1. Clone the repository
git clone <repo-url> drone-research
cd drone-research

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install SAM 2 repository
pip install -e sam2_repo

# 5. Install project dependencies
pip install -r requirements.txt
```

---

## 6. Checkpoint Setup

Ensure the pretrained SAM 2.1 Hiera-Small weights are located at `checkpoints/sam2.1_hiera_small.pt`:

```powershell
# Verify checkpoint exists
Test-Path checkpoints/sam2.1_hiera_small.pt
```

To download the checkpoint if missing:
```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -OutFile "checkpoints/sam2.1_hiera_small.pt"
```

Verify model loading:
```powershell
python test_sam2.py
```
*Expected Output:* `Model successfully loaded: <class 'sam2.modeling.sam2_base.SAM2Base'>`

---

## 7. Configuration (`configs/sam.yaml`)

```yaml
preprocessing:
  strategy: "tiling"   # Default strategy for high-resolution static images
  resize_max_dim: 1536
  tiling:
    tile_size: 1536
    overlap: 256
    iou_threshold: 0.7

spatial_index:
  enabled: true
  cell_size: 1536

video:
  output_interval: 30  # Export JSON interval (frames). Predictor processes all sequential frames.
  frame_interval: 30   # Backward compatibility alias
  strategy: "resize"
  predictor_enabled: true
  offload_video_to_cpu: true
  offload_state_to_cpu: true

model:
  type: "sam2.1_hiera_small"
  checkpoint_path: "checkpoints/sam2.1_hiera_small.pt"
  config_path: "configs/sam2.1/sam2.1_hiera_s.yaml"

mask_filter:
  min_area: 1500
  min_stability_score: 0.9
  min_predicted_iou: 0.85

  reject_tile_edge: true
  edge_tolerance: 2

output:
  crops_dir: "outputs/crops"
  json_path: "outputs/segmentation_results.json"
  video_json_path: "outputs/video_results.json"
  crop_size: [224, 224]
```

---

## 8. Usage

### A. Run Static Image Segmentation Pipeline
Processes a large orthophoto (`sample_02.jpg`), performs tiling, filtering, deduplication, stitching, artifact rejection, crop extraction, and JSON metadata export:
```powershell
python -u scripts/test_mask_generation.py
```

### B. Run Video Tracking Pipeline (Python API)
```python
from segmentation.sam_model import load_config, load_sam_model
from segmentation.video_processing import (
    build_sam2_video_predictor_from_config,
    process_video_with_predictor,
)

config = load_config("configs/sam.yaml")
mask_generator = load_sam_model(config)
predictor = build_sam2_video_predictor_from_config(config)

results = process_video_with_predictor(
    predictor=predictor,
    video_path="path/to/drone_video.mp4",
    frames_dir="outputs/extracted_frames",
    config=config,
    mask_generator=mask_generator,
    initial_boxes=None,          # Triggers automatic Frame 0 object discovery
    output_interval=30,          # Exports 1 JSON record per second (at 30 FPS)
    json_path="outputs/video_results.json",
)
```

---

## 9. Output Specifications

### Static Image JSON (`outputs/segmentation_results.json`)
```json
[
  {
    "segment_id": 0,
    "bbox": [7180, 0, 1841, 2094],
    "area": 2548776,
    "sam_score": 0.9852,
    "crop_path": "outputs/crops/segment_0.png"
  }
]
```

### Video Tracking JSON (`outputs/video_results.json`)
```json
{
  "video_path": "outputs/test_fixtures/sample_drone_clip.mp4",
  "original_width": 1920,
  "original_height": 1080,
  "total_processed_frames": 300,
  "exported_frames": 10,
  "output_interval": 30,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "segments": [
        {
          "track_id": 1,
          "bbox": [384, 216, 768, 432],
          "area": 32840,
          "sam_score": 0.90
        }
      ]
    }
  ]
}
```

---

## 10. Testing & Verification

Run the entire automated test suite (33 test cases):
```powershell
python -m unittest discover -s tests -v
```

### Test Suite Breakdown:
- **`tests/test_tiling_pipeline.py` (15 tests):** Verifies tile generation, boundary detection, IoU/IoS calculations, DSU mask merging, crop bounding, and JSON export.
- **`tests/test_spatial_index.py` (8 tests):** Verifies `SpatialGridIndex` multi-cell indexing, cell boundary handling, candidate queries, and deterministic sorting.
- **`tests/test_performance_equivalence.py` (2 tests):** Proves 100% exact result equivalence between baseline $O(N^2)$ and optimized $O(N+K)$ algorithms.
- **`tests/test_video_pipeline.py` (7 tests):** Verifies coordinate scaling, clamping, Frame 0 auto-discovery, track ID assignment, and exception handling.
- **`tests/test_video_integration.py` (1 test):** Real end-to-end integration test with CUDA GPU, real SAM 2.1 weights, and `SAM2VideoPredictor` temporal propagation.

---

## 11. Performance Benchmarks

All metrics measured on an **NVIDIA RTX 3050 Laptop GPU (6GB VRAM)**:

| Stage | Benchmark | Result |
| :--- | :--- | :--- |
| **Spatial Optimization** | Deduplication candidate checks | **64.3% reduction** (1,225 $\rightarrow$ 437 checks) |
| **Spatial Optimization** | Stitching candidate pair checks | **63.6% reduction** (1,225 $\rightarrow$ 446 checks) |
| **Static ~60MP Orthophoto** | 40 Tiles Processing | 965 raw $\rightarrow$ 354 final masks in **~6.8 seconds** |
| **Static ~60MP Orthophoto** | Peak GPU VRAM | **< 3.8 GB** |
| **Video Propagation** | Temporal Throughput | **~1.83 frames / sec** (~0.55s / frame) |
| **Video Propagation** | Frame Loading Throughput | **~24.7 frames / sec** |
| **Video Propagation** | VRAM with CPU Offloading | **< 4.0 GB** |

---

## 12. Troubleshooting & FAQ

### 1. `UserWarning: cannot import name '_C' from 'sam2'`
- **Cause:** Optional C++/CUDA post-processing kernel (used for connected-component hole filling) is not compiled on Windows.
- **Impact:** SAM 2 gracefully falls back to the un-postprocessed mask. Backbone feature extraction, attention mechanisms, and temporal propagation are **100% unaffected**.

### 2. `ValueError: No valid initial objects discovered on Frame 0 for video tracking`
- **Cause:** Frame 0 is blank, dark, or contains no objects passing the quality thresholds (`min_area=1500`, `min_stability=0.9`, `min_iou=0.85`).
- **Solution:** Supply explicit `initial_boxes` or adjust `mask_filter` thresholds in `configs/sam.yaml`.

### 3. Windows PowerShell Script Execution Policy
If `.\venv\Scripts\Activate.ps1` fails with an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. CUDA Out of Memory (OOM) on 6GB GPUs
Ensure `offload_video_to_cpu: true` and `offload_state_to_cpu: true` are enabled in `configs/sam.yaml`. This keeps frame storage in system RAM and streams tensors to GPU only during active layer computation.