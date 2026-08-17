# SAM 2.1 Drone Segmentation Setup

## 1. Overview

Bu modul yüksək rezolyusiyalı drone şəkillərində avtomatik obyekt segmentasiyası üçün Meta SAM 2.1 modelindən istifadə edir.

ArcGIS Packing House District datasetindəki şəkillərin ölçüsü `9504×6336` olduğuna görə şəkillər birbaşa bütöv şəkildə modelə verilmir. Modul onları overlap olan tile-lara bölür, hər tile üzərində maskalar yaradır, maskaları filtrasiya və deduplication edir, sonra bounding box və classifier-ready crop-lar çıxarır.

Segmentation modulunun əsas nəticələri:

- SAM maskaları;
- `[x, y, width, height]` əsasında hesablanan bounding box-lar;
- classifier üçün `224×224` crop şəkilləri;
- SAM confidence göstəriciləri;
- shared schema ilə uyğun JSON metadata.

Classifier modelinin implementasiyası bu modulun scope-una daxil deyil. Segmentation nəticələri ümumi layihədə classifier mərhələsinə ötürülür.

Ümumi pipeline:

```text
Raw Drone Image
        ↓
SAM 2.1 Automatic Mask Generation
        ↓
Mask Filtering
        ↓
Mask Stitching and Deduplication
        ↓
Bounding Box Extraction
        ↓
Classifier-Ready Crop Generation
        ↓
SegmentPrediction JSON
        ↓
Classifier
```

---

## 2. Model Selection

İstifadə edilən model:

```text
SAM 2.1 Hiera-Small
```

Modelin seçilmə səbəbləri:

- SAM 1 ilə müqayisədə daha yeni arxitekturadır;
- statik şəkilləri və videoları dəstəkləyir;
- automatic mask generation imkanına malikdir;
- Hiera-Small variantı keyfiyyət və resurs istifadəsi arasında balans yaradır;
- böyük drone şəkillərində tiling ilə istifadə edilə bilir.

Model faylları:

```text
Checkpoint:
checkpoints/sam2.1_hiera_small.pt

Model config:
configs/sam2.1/sam2.1_hiera_s.yaml
```

`config_path` quraşdırılmış rəsmi SAM 2 package daxilindəki Hydra model konfiqurasına istinad edir.

---

## 3. Repository Structure

```text
drone-research/
├── checkpoints/
│   └── sam2.1_hiera_small.pt
├── configs/
│   └── sam.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── classifier/
├── docs/
│   └── sam_setup.md
├── outputs/
│   ├── crops/
│   ├── segmentation_results.json
│   └── video_results.json
├── requirements/
│   └── sam.txt
├── sam2_repo/
├── scripts/
│   └── test_mask_generation.py
├── src/
│   ├── segmentation/
│   │   ├── bbox_extractor.py
│   │   ├── mask_filter.py
│   │   ├── sam_model.py
│   │   ├── tiling.py
│   │   └── video_processing.py
│   └── utils/
│       └── schemas.py
├── tests/
│   ├── test_performance_equivalence.py
│   ├── test_spatial_index.py
│   ├── test_tiling_pipeline.py
│   ├── test_video_integration.py
│   └── test_video_pipeline.py
├── SAM_research_notes.md
└── test_sam2.py
```

Aşağıdakı fayl və qovluqlar local saxlanılır və GitHub-a commit edilmir:

- `checkpoints/`;
- `sam2_repo/`;
- `data/` daxilindəki datasetlər;
- `outputs/`;
- model weights;
- generated crops;
- generated JSON nəticələri.

---

## 4. Requirements

SAM modulu üçün birbaşa dependency-lər:

```text
numpy>=1.26
opencv-python>=4.8
PyYAML>=6.0,<7.0
torch>=2.5.1
torchvision>=0.20.1
```

Onlar aşağıdakı faylda saxlanılır:

```text
requirements/sam.txt
```

Dependency-ləri quraşdırmaq üçün:

```bash
python -m pip install -r requirements/sam.txt
```

---

## 5. Install Official SAM 2

Meta-nın rəsmi SAM 2 repository-sini layihənin root qovluğunda clone et:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
```

SAM 2-ni editable package kimi quraşdır:

```bash
python -m pip install -e sam2_repo
```

Quraşdırmanı yoxla:

```bash
python -c "
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

print('SAM 2 import: PASSED')
"
```

`sam2_repo/` `.gitignore` daxilindədir və GitHub-a push edilmir.

---

## 6. Download SAM 2.1 Checkpoint

Checkpoint qovluğunu yarat:

```bash
mkdir -p checkpoints
```

SAM 2.1 Hiera-Small checkpoint-i yüklə:

```bash
curl -L \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
  -o checkpoints/sam2.1_hiera_small.pt
```

Faylın mövcudluğunu yoxla:

```bash
ls -lh checkpoints/sam2.1_hiera_small.pt
```

Checkpoint GitHub-a commit edilməməlidir.

---

## 7. Configuration

SAM konfiqurasiyası:

```text
configs/sam.yaml
```

Cari konfiqurasiya:

```yaml
preprocessing:
  strategy: "tiling"
  resize_max_dim: 1536
  tiling:
    tile_size: 1536
    overlap: 256
    iou_threshold: 0.7

spatial_index:
  enabled: true
  cell_size: 1536

video:
  output_interval: 30
  frame_interval: 30
  strategy: "resize"
  predictor_enabled: true
  offload_video_to_cpu: true
  offload_state_to_cpu: true

model:
  type: "sam2.1_hiera_small"
  device: "auto"
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

### Device selection

`device: "auto"` olduqda sistem bu ardıcıllıqla seçim edir:

```text
CUDA → MPS → CPU
```

- NVIDIA GPU varsa: `cuda`;
- Apple Silicon və MPS aktivdirsə: `mps`;
- digərləri yoxdursa: `cpu`.

Device-i manual seçmək də mümkündür:

```yaml
device: "cuda"
```

```yaml
device: "mps"
```

```yaml
device: "cpu"
```

Manual seçilən device mövcud deyilsə, modul aydın `RuntimeError` qaytarır.

Device seçimini yoxlamaq üçün:

```bash
python -c "
from src.segmentation.sam_model import load_config, resolve_device

config = load_config()
print('Selected device:', resolve_device(config))
"
```

Apple Silicon Mac-da gözlənilən nəticə:

```text
Selected device: mps
```

---

## 8. Static Image Pipeline

ArcGIS şəkilləri `9504×6336` ölçüsündədir. Bu şəkillərin bütöv şəkildə SAM modelinə verilməsi yüksək yaddaş istifadəsinə səbəb ola bilər.

Buna görə static image pipeline tiling istifadə edir:

```text
9504×6336 Drone Image
        ↓
1536×1536 Tiles
        ↓
256 px Overlap
        ↓
SAM Mask Generation per Tile
        ↓
Global Coordinate Conversion
        ↓
Boundary-Aware Stitching
        ↓
Mask Deduplication
        ↓
Quality Filtering
        ↓
Bounding Box and Crop Export
```

Tiling parametrləri:

```yaml
tile_size: 1536
overlap: 256
iou_threshold: 0.7
```

Overlap tile sərhədində bölünən obyektlərin sonradan birləşdirilməsinə kömək edir.

---

## 9. Mask Filtering

Maskalar aşağıdakı göstəricilərə əsasən filtrasiya edilir:

```yaml
min_area: 1500
min_stability_score: 0.9
min_predicted_iou: 0.85
```

Filtrasiya zamanı:

- çox kiçik maskalar silinir;
- aşağı stability score olan maskalar silinir;
- aşağı predicted IoU olan maskalar silinir;
- tile sərhədində yaranan lazımsız artefaktlar yoxlanılır;
- real image sərhədinə toxunan obyektlər qorunur;
- overlap səbəbindən yaranan təkrarlanan maskalar birləşdirilir.

---

## 10. Bounding Box Format

SAM maskasından bounding box aşağıdakı daxili formatda hesablanır:

```text
[x, y, width, height]
```

Shared JSON contract daxilində isə bbox object kimi saxlanılır:

```json
{
  "x": 120,
  "y": 85,
  "width": 240,
  "height": 190
}
```

Bu format `src/utils/schemas.py` daxilindəki `BoundingBox` modelinə uyğundur.

---

## 11. Segment Output Contract

Hər valid SAM maskası aşağıdakı shared schema formatında export edilir:

```json
{
  "segment_id": 0,
  "bbox": {
    "x": 120,
    "y": 85,
    "width": 240,
    "height": 190
  },
  "area": 32840,
  "sam_score": 0.94,
  "crop_path": "outputs/crops/segment_0.png"
}
```

Bu nəticə aşağıdakı Pydantic modeli ilə validate edilir:

```python
class SegmentPrediction(BaseModel):
    segment_id: int
    bbox: BoundingBox
    area: int
    sam_score: float
    crop_path: str | None = None
```

Invalid segmentlər export edilmir:

- `width <= 0`;
- `height <= 0`;
- `area <= 0`;
- boş crop.

---

## 12. Classifier Crop Generation

Hər valid segment üçün crop yaradılır:

```text
outputs/crops/segment_<segment_id>.png
```

Default crop ölçüsü:

```text
224×224
```

Crop aşağıdakı mərhələdə classifier inputu kimi istifadə olunur:

```text
SAM Segment
    ↓
Bounding Box
    ↓
224×224 Crop
    ↓
Classifier
    ↓
Class Name + Confidence
```

---

## 13. Running Static Segmentation

Əvvəl local sample image hazırla. Dataset şəkilləri repository-yə commit edilməməlidir.

Static segmentation scriptini işə sal:

```bash
python -u scripts/test_mask_generation.py
```

Script aşağıdakı əməliyyatları aparır:

1. SAM 2.1 modelini yükləyir;
2. input şəklini oxuyur;
3. tiling tətbiq edir;
4. avtomatik maskalar yaradır;
5. maskaları filtrasiya edir;
6. bounding box-lar çıxarır;
7. classifier-ready crop-lar yaradır;
8. segmentation JSON faylı export edir.

Default outputlar:

```text
outputs/crops/
outputs/segmentation_results.json
```

---

## 14. Video Pipeline

Video pipeline SAM2 video predictor istifadə edir:

```text
Input Video
    ↓
Sequential Frame Extraction
    ↓
Frame 0 Automatic Object Discovery
    ↓
Initial Bounding Box Prompts
    ↓
SAM2VideoPredictor
    ↓
Temporal Mask Propagation
    ↓
Coordinate Scaling and Clamping
    ↓
Video JSON Export
```

Frame 0 üzərində obyekt tapılmadıqda hardcoded fallback bounding box istifadə edilmir. Bunun əvəzinə aydın xəta qaytarılır.

Video nəticələri:

```text
outputs/video_results.json
```

Video inference üçün uyğun runtime hardware və local SAM 2 checkpoint tələb olunur.

---

## 15. Testing

Bütün unit və integration testlərini işə sal:

```bash
python -m pytest tests -v
```

Son local integration nəticəsi:

```text
36 tests collected
35 passed
1 skipped
```

Test qrupları:

- pipeline merge və threshold testləri;
- spatial grid index testləri;
- tiling testləri;
- mask stitching və deduplication testləri;
- bounding box testləri;
- classifier crop generation testi;
- shared `SegmentPrediction` validation testi;
- video utility testləri;
- performance equivalence testləri.

Skip edilən test real SAM2 video integration testidir. Bu test üçün aşağıdakılar tələb olunur:

- local `sam2_repo/`;
- SAM 2.1 checkpoint;
- uyğun runtime hardware;
- integration test environment.

Yalnız segmentation export testini işə salmaq üçün:

```bash
python -m pytest \
  tests/test_tiling_pipeline.py::TestTilingPipeline::test_15_export_segments_json_format \
  -v
```

---

## 16. Model Loading Test

SAM 2 package və checkpoint hazır olduqdan sonra model loading testini işə sal:

```bash
python test_sam2.py
```

Gözlənilən nəticə:

```text
Model successfully loaded
```

Əgər SAM 2 quraşdırılmayıbsa, `load_sam_model()` aşağıdakı quraşdırma istiqamətini göstərən xəta qaytarır:

```text
SAM 2 is not installed. Clone the official SAM 2 repository into
sam2_repo/ and run: python -m pip install -e sam2_repo
```

---

## 17. Dataset Handoff

Data preprocessing modulu seçilmiş ArcGIS şəkillərini bu formada hazırlayır:

```text
Raw ArcGIS MPO/JPEG
        ↓
MPO Frame 0
        ↓
Single-Frame RGB JPEG
        ↓
Original 9504×6336 Resolution
        ↓
SAM Tiling Pipeline
```

Data modulu şəkli əvvəlcədən `1024×1024` ölçüsünə resize etmir. Original resolution saxlanılır və ölçü idarəsi SAM modulunun tiling mərhələsində aparılır.

Bu contract haqqında əlavə məlumat:

```text
docs/dataset.md
```

---

## 18. Generated Files and Git Safety

Aşağıdakılar GitHub-a commit edilməməlidir:

```text
checkpoints/
sam2_repo/
data/
outputs/
*.pt
*.pth
*.ckpt
*.mp4
*.mov
*.avi
```

Commit etməzdən əvvəl yoxla:

```bash
git status
```

Dataset, checkpoint, generated crop və output JSON faylları görünməməlidir.

---

## 19. Troubleshooting

### `ModuleNotFoundError: No module named 'sam2'`

SAM 2 repository-sini clone və install et:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
python -m pip install -e sam2_repo
```

### `ModuleNotFoundError: No module named 'torch'`

SAM dependency-lərini quraşdır:

```bash
python -m pip install -r requirements/sam.txt
```

### CUDA mövcud deyil

`configs/sam.yaml` daxilində bunu saxla:

```yaml
device: "auto"
```

Sistem avtomatik MPS və ya CPU seçəcək.

### Checkpoint tapılmır

Bu faylın mövcudluğunu yoxla:

```bash
ls -lh checkpoints/sam2.1_hiera_small.pt
```

### Input image açıla bilmir

Input path-i yoxla. Modul invalid path üçün `FileNotFoundError` qaytarır.

### MPS-də bəzi operation-lar dəstəklənmir

CPU istifadə et:

```yaml
model:
  device: "cpu"
```

---

## 20. Module Deliverables

SAM segmentation modulunun əsas deliverable-ları:

```text
src/segmentation/sam_model.py
src/segmentation/mask_filter.py
src/segmentation/bbox_extractor.py
src/segmentation/tiling.py
src/segmentation/video_processing.py
configs/sam.yaml
requirements/sam.txt
docs/sam_setup.md
scripts/test_mask_generation.py
tests/test_tiling_pipeline.py
tests/test_spatial_index.py
tests/test_performance_equivalence.py
tests/test_video_pipeline.py
tests/test_video_integration.py
```

Modulun yekun çıxışı classifier və integration pipeline tərəfindən istifadə edilə bilən valid `SegmentPrediction` məlumatlarıdır.