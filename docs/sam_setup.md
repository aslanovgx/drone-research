# SAM 2.1 Drone Object Segmentation Setup & Architecture Documentation

## 1. Overview
Bu proyekt yüksək rezolyusiyalı (təxminən 9504×6336 / ~60MP) dron və ortofoto şəkillərində obyektlərin avtomatik seqmentasiyası, paçalara bölünmə (tiling), spatial grid indekslənməsi, sərhəd maskalarının çox-şərtli birləşdirilməsi (stitching), kəsilmiş maska crop-larının generasiyası və video kadrlarında Frame 0 auto-discovery daxil olmaqla temporal maska izlənməsi (`SAM2VideoPredictor`) üçün hazırlanmış SAM 2.1 (Segment Anything Model 2.1) əsaslı sistemdir.

> **MÜHÜM SCOPE MƏHDUDİYYƏTİ**: Təsnifatçı (Classifier - ResNet, EfficientNet, ViT) modellərinin inteqrasiyası bu layihənin scope-undan kənardır. Layihənin yekun məsuliyyəti **SAM 2.1 seqmentasiyası, bounding box, 224x224 crop-lar və JSON metadata eksportudur**.

---

## 2. Repository Architecture
Layihənin fayl strukturu və modulların funksiyaları:

```text
drone-research/
├── checkpoints/
│   └── sam2.1_hiera_small.pt        # Pretrained SAM 2.1 Hiera-Small model çəkiləri (~184MB) [IMPLEMENTED]
├── configs/
│   └── sam.yaml                     # Mərkəzi konfiqurasiya faylı [IMPLEMENTED]
├── data/
│   ├── raw/                         # İlkin dron datasets (Esri Packing House District və s.) [IMPLEMENTED]
│   └── samples/                     # Test üçün istifadə olunan sample şəkillər (sample_01..13.jpg) [IMPLEMENTED]
├── docs/
│   └── sam_setup.md                 # Ətraflı texniki sənədləşdirmə faylı [IMPLEMENTED]
├── outputs/
│   ├── crops/                       # Kəsilmiş 224x224 PNG segment şəkilləri [IMPLEMENTED]
│   ├── segmentation_results.json    # Eksport olunan statik JSON metadata faylı [IMPLEMENTED]
│   └── video_results.json           # Eksport olunan video temporal JSON metadata faylı [IMPLEMENTED]
├── sam2_repo/                       # Official SAM 2 repository mənbə kodu [IMPLEMENTED]
├── scripts/
│   └── test_mask_generation.py      # Əsas icra, export və vizualizasiya scripti [IMPLEMENTED]
├── src/
│   └── segmentation/
│       ├── __init__.py
│       ├── bbox_extractor.py        # BBox hesablanması, crop kəsimi və JSON eksportu [IMPLEMENTED]
│       ├── mask_filter.py           # Keyfiyyət süzgəci və daxili tile sərhəd filtri [IMPLEMENTED]
│       ├── sam_model.py             # SAM 2.1 modelinin yüklənməsi və strategiya seçimi [IMPLEMENTED]
│       ├── tiling.py                # Tile generasiyası, SpatialGridIndex, DSU stitching, dedup [IMPLEMENTED]
│       └── video_processing.py      # Frame 0 auto-discovery & SAM2VideoPredictor temporal video pipeline [IMPLEMENTED]
├── tests/
│   ├── test_tiling_pipeline.py      # 15 ədəd avtomatlaşdırılmış unit və regressiya testi [IMPLEMENTED]
│   ├── test_spatial_index.py        # 8 ədəd SpatialGridIndex testləri [IMPLEMENTED]
│   ├── test_performance_equivalence.py # 2 ədəd nəticə ekvivalentliyi və candidate reduction testləri [IMPLEMENTED]
│   ├── test_video_pipeline.py       # 7 ədəd video pipeline, Frame 0 auto-discovery & scaling testləri [IMPLEMENTED]
│   └── test_video_integration.py    # Real SAM2VideoPredictor CUDA integration testi [IMPLEMENTED]
├── README.md                        # Layihə haqqında ümumi məlumat [IMPLEMENTED]
├── requirements.txt                 # Asılılıqların siyahısı [IMPLEMENTED]
├── SAM_research_notes.md            # SAM 1 vs SAM 2.1 tədqiqat qeydləri [EXPERIMENTAL/NOTES]
└── test_sam2.py                    # Modelin yüklənməsini yoxlayan minimal test scripti [IMPLEMENTED]
```

---

## 3. Pipeline Architecture

### Video Temporal Pipeline Execution Flow:
```
Video File / Stream (.mp4)
       │
       ▼ [IMPLEMENTED]
1. Sequential Frame Extraction (Unbroken sequence: 000000.jpg, 000001.jpg, 000002.jpg...)
       │
       ▼ [IMPLEMENTED]
2. Frame 0 Automatic Object Discovery (SAM 2.1 Automatic Generator + Quality Filter)
       │  (No hardcoded fallback box; raises ValueError if 0 objects found)
       ▼ [IMPLEMENTED]
3. Initial Prompt Registration (SAM2VideoPredictor.add_new_points_or_box on Frame 0)
       │  (Assigned deterministic track IDs: 1, 2, 3...)
       ▼ [IMPLEMENTED]
4. Continuous Temporal Mask Propagation (SAM2VideoPredictor.propagate_in_video across ALL frames)
       │  (VRAM Safety: offload_video_to_cpu=True, offload_state_to_cpu=True)
       ▼ [IMPLEMENTED]
5. Output Export Interval Filtering (e.g. export JSON records every output_interval=30 frames)
       │
       ▼ [IMPLEMENTED]
6. Coordinate Scale Conversion & Clamping (scale_x, scale_y back to ORIGINAL video resolution)
       │
       ▼ [IMPLEMENTED]
7. JSON Metadata Export (outputs/video_results.json with track_id, bbox, area, sam_score)
```

---

## 4. Model Selection

- `SAM 1 (ViT-B, ViT-L, ViT-H)`: **[EXPERIMENTAL/HISTORICAL]** Baseline referans.
- `SAM 2.1 Hiera-Small`: **[IMPLEMENTED]** Aktiv pipeline-da istifadə olunan əsas model.
  - Checkpoint: `checkpoints/sam2.1_hiera_small.pt` (~184.4 MB)
  - Config: `configs/sam2.1/sam2.1_hiera_s.yaml`

---

## 5. Configuration (`configs/sam.yaml`) **[IMPLEMENTED]**

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
  output_interval: 30  # Export JSON interval (frames). Predictor processes all unbroken sequential frames.
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

## 6. Native SAM2 Video Predictor Pipeline & Frame 0 Auto-Discovery **[IMPLEMENTED]**

### 1. Frame 0 Automatic Object Discovery:
- Frame 0 avtomatik olaraq `discover_initial_boxes_frame_0` funksiyası ilə seqmentasiya edilir.
- Qeyri-stabil və kiçik maskalar süzülür, yalnız etibarlı bounding box-lar tapılır.
- Hər bir obyektə deterministik `track_id` (1, 2, 3...) verilir.
- Əgər Frame 0-da 0 obyekt tapılarsa, hardcoded fallback box İSTİFADƏ EDİLMİR, aydın `ValueError` xətası qaldırılır.

### 2. Unbroken Sequential Frame Extraction & Temporal Continuity:
- Videonun kadrları `extract_sequential_frames` ilə ardıcıl şəkildə (`000000.jpg`, `000001.jpg`, `000002.jpg`...) kəsilir.
- Predictor bütün ardıcıl kadrlar üzərində temporal maska ötürməsi (`propagate_in_video`) edir, kadrlararası izləmə yaddaşı qırılmır.

### 3. Output Export Interval:
- `output_interval: 30` göstəricisi yalnız eksport olunan `outputs/video_results.json` faylı üçün tətbiq olunur (`frame_idx % output_interval == 0`). Predictor isə kadrlararası yaddaş zəncirini qorumaq üçün bütün ardıcıl kadrları emal edir.

### 4. Video Koordinat Scalinq Və JSON Output:
Bütün maska və bbox-lar **ORİJİNAL video kadr rezolyusiyasına** (`w_orig`, `h_orig`) qaytarılır və clamp olunur.

Çıxış faylı: `outputs/video_results.json`
```json
{
  "video_path": "data/samples/sample_video.mp4",
  "original_width": 3840,
  "original_height": 2160,
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

## 7. Testing Və Verifikasiya Nəticələri **[IMPLEMENTED]**

İcra əmri: `python -m unittest discover -s tests -v`

### Dəqiq Test Qrupları:
- **`test_tiling_pipeline.py`**: 15 test (Keçdi - PASSED)
- **`test_spatial_index.py`**: 8 test (Keçdi - PASSED)
- **`test_performance_equivalence.py`**: 2 test (Keçdi - PASSED)
- **`test_video_pipeline.py`**: 7 test (Keçdi - PASSED)
- **`test_video_integration.py`**: 1 test (Keçdi - PASSED / Real CUDA SAM2VideoPredictor integration testi)
- **Ümumi Test Sayı**: **33 test** **[VERIFIED]**

---

## 8. How to Run

### 1. Bütün Unit Testləri İcra Etmək:
```bash
python -m unittest discover -s tests -v
```

### 2. Statik Şəkil Pipeline-nı İcra Etmək:
```bash
python -u scripts/test_mask_generation.py
```

### 3. Model Yüklənmə Testi:
```bash
python test_sam2.py
```