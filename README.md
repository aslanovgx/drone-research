# 🚁 SAM-Based Drone Object Segmentation and Classification

An end-to-end computer vision pipeline for **object segmentation and classification in aerial drone imagery**.

The system uses **SAM 2.1** to generate candidate object masks. Each valid mask is converted into a bounding box and image crop. A **MobileNetV3-Small** classifier then assigns one of four semantic classes to every crop.

---

## 📌 Overview

The project combines:

* **SAM 2.1** for automatic object segmentation
* Mask filtering and deduplication
* Boundary stitching
* Bounding-box generation
* Crop extraction
* **MobileNetV3-Small** classification
* Confidence-based filtering
* Visualization
* Structured JSON output

The complete pipeline can be executed on drone images and extracted video frames.

---

## 🔄 Pipeline

```text
Drone Image / Extracted Video Frame
                ↓
       SAM 2.1 Mask Generation
                ↓
Mask Filtering, Deduplication
    and Boundary Stitching
                ↓
 Bounding Box + Crop Generation
                ↓
 MobileNetV3-Small Classifier
                ↓
 Class Label + Confidence Filter
                ↓
 Annotated Image + JSON Output
```

---

## 📤 Output

The final pipeline output contains:

* Segmentation candidates
* Bounding boxes
* Class labels
* Classifier confidence scores
* SAM confidence scores
* Annotated images
* Structured JSON prediction files

---

## 🏷️ Classes

The classifier currently supports four semantic classes:

| Class      | Description                                                           |
| ---------- | --------------------------------------------------------------------- |
| `building` | Building roofs and dominant building structures                       |
| `car`      | Cars and other clearly visible road vehicles                          |
| `tree`     | Trees and dominant vegetation regions                                 |
| `other`    | Roads, shadows, HVAC units, poles, signs and other non-target regions |

### Why is the `other` class important?

The `other` class prevents every region proposed by SAM from being forced into one of the target object classes.

This is especially important because SAM may generate masks for:

* roads;
* shadows;
* rooftops containing mixed content;
* HVAC units;
* poles;
* signs;
* background regions;
* other visually distinct objects.

---

## 📊 Data

The **ArcGIS Packing House District drone dataset** was used for classifier dataset preparation.

Source images were divided **geographically before crop generation** to reduce spatial leakage between the training and validation datasets.

### Dataset Distribution

| Split      | Building |    Car |   Tree |   Other |   Total |
| ---------- | -------: | -----: | -----: | ------: | ------: |
| Training   |       49 |     63 |     57 |     118 | **287** |
| Validation |       39 |     25 |     42 |      97 | **203** |
| **Total**  |   **88** | **88** | **99** | **215** | **490** |

The repository includes reproducibility manifests for:

* Geographic source-image splitting
* Source-patch selection
* Crop labels
* Original SAM crop paths

> [!NOTE]
> The original drone images, generated crops, model checkpoints and videos are not committed to the repository because of their size and licensing constraints.

---

## 🧠 Classifier

The selected baseline classifier is:

**MobileNetV3-Small**

The model is initialized using **ImageNet pretrained weights** and fine-tuned on the generated drone-object crop dataset.

---

## 📈 Classifier Results

### Validation Metrics

| Metric   |     Result |
| -------- | ---------: |
| Accuracy | **75.86%** |
| Macro F1 | **0.7717** |

### Per-Class Results

| Class    | Precision | Recall |         F1 |
| -------- | --------: | -----: | ---------: |
| Building |    0.6250 | 0.6410 |     0.6329 |
| Car      |    0.8000 | 0.9600 | **0.8727** |
| Other    |    0.7952 | 0.6804 |     0.7333 |
| Tree     |    0.7800 | 0.9286 | **0.8478** |

The strongest validation performance is currently obtained for:

1. **Cars**
2. **Trees**

Building classification remains more sensitive to:

* Roof scale
* Surrounding vegetation
* Shadows
* Viewing angle
* Domain shift

---

## 🎥 External DroneStock Evaluation

Two independently selected stock drone videos were used as **external unseen-domain tests**:

* Suburban residential flyover
* Top-down Central Park aerial video

Frames were extracted at approximately:

* **20%**
* **50%**
* **80%**

of each video.

All six extracted frames successfully completed the full segmentation and classification pipeline.

### Results

| Scene        | Frame | SAM Segments | Final Detections | Building | Car | Tree | Other |
| ------------ | ----: | -----------: | ---------------: | -------: | --: | ---: | ----: |
| Central Park |   20% |            5 |                2 |        0 |   0 |    1 |     1 |
| Central Park |   50% |            6 |                2 |        0 |   0 |    0 |     2 |
| Central Park |   80% |            5 |                1 |        0 |   0 |    0 |     1 |
| Suburbs      |   20% |           13 |                6 |        0 |   0 |    6 |     0 |
| Suburbs      |   50% |           26 |               14 |        0 |   0 |    8 |     6 |
| Suburbs      |   80% |           21 |               10 |        0 |   0 |    5 |     5 |

These runs verify that the complete system can execute successfully on **previously unseen drone footage**.

> [!IMPORTANT]
> These results should **not** be interpreted as formal external accuracy measurements because the DroneStock frames do not contain ground-truth annotations.

For the complete evaluation, interpretation and limitations, see:

```text
docs/final_evaluation.md
```

---

## 📁 Repository Structure

```text
.
├── configs/
│   └── SAM, classifier and pipeline configurations
│
├── data/
│   └── manifests/
│       └── Reproducible source splits and crop labels
│
├── docs/
│   └── Documentation and evaluation reports
│
├── requirements/
│   └── Module-specific Python dependencies
│
├── scripts/
│   └── Dataset and labeling workflow utilities
│
├── src/
│   ├── classification/
│   │   └── Classifier training and inference
│   │
│   ├── data/
│   │   └── Dataset analysis and preprocessing
│   │
│   ├── segmentation/
│   │   └── SAM loading, tiling, filtering and crop export
│   │
│   ├── visualization/
│   │   └── Prediction visualization
│   │
│   ├── pipeline.py
│   │   └── Prediction merging and output generation
│   │
│   └── run_pipeline.py
│       └── End-to-end command-line runner
│
└── tests/
    └── Unit and integration tests
```

---

# ⚙️ Installation

## Requirements

Development was performed using:

```text
Python 3.12
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

---

## 📦 Install Project Dependencies

Install each dependency group:

```bash
python -m pip install -r requirements/data.txt
python -m pip install -r requirements/sam.txt
python -m pip install -r requirements/classifier.txt
python -m pip install -r requirements/integration.txt
```

---

## 🧩 Install SAM 2

Clone the official SAM 2 repository:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
```

Install it:

```bash
SAM2_BUILD_CUDA=0 python -m pip install -e ./sam2_repo
```

---

# 📥 Model Checkpoints

Create the checkpoint directory:

```bash
mkdir -p checkpoints
```

## SAM 2.1 Checkpoint

Download the official **SAM 2.1 Hiera Small** checkpoint:

```bash
curl -L --fail \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" \
  -o checkpoints/sam2.1_hiera_small.pt
```

---

## Classifier Checkpoint

Place the trained classifier checkpoint at:

```text
checkpoints/classifier.pt
```

> [!NOTE]
> Model checkpoint files are excluded from Git because of their size.

The classifier baseline should be downloaded from the project's **GitHub Release assets** or provided separately.

---

# ▶️ Run the End-to-End Pipeline

## Default Image Pipeline

Run the pipeline on a drone image:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m src.run_pipeline \
  path/to/drone_image.jpg \
  --sam-config configs/sam.yaml \
  --classifier-config configs/classifier.yaml \
  --pipeline-config configs/pipeline.yaml \
  --checkpoint checkpoints/classifier.pt
```

---

## DroneStock Configuration

Run the configuration used for external DroneStock evaluation:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m src.run_pipeline \
  path/to/extracted_frame.jpg \
  --sam-config configs/sam_dronestock.yaml \
  --classifier-config configs/classifier.yaml \
  --pipeline-config configs/pipeline_dronestock.yaml \
  --checkpoint checkpoints/classifier.pt
```

> [!TIP]
> `PYTORCH_ENABLE_MPS_FALLBACK=1` is useful when running the project on Apple Silicon.
>
> It can normally be omitted on CPU or CUDA-based systems.

---

## 📂 Generated Files

Generated files are written to the output directories configured inside the SAM and pipeline YAML configuration files.

Typical outputs include:

```text
outputs/
├── crops/
├── segmentation_results.json
├── json/
│   └── <image_name>.json
└── predictions/
    └── <image_name>_annotated.jpg
```

---

# 🏋️ Train the Classifier

The classifier dataset must follow this directory structure:

```text
data/classifier/
├── train/
│   ├── building/
│   ├── car/
│   ├── other/
│   └── tree/
│
└── validation/
    ├── building/
    ├── car/
    ├── other/
    └── tree/
```

Start training:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m src.classification.train \
  --config configs/classifier.yaml \
  --device auto
```

The best-performing checkpoint is saved according to the checkpoint path configured in:

```text
configs/classifier.yaml
```

---

# 🧪 Tests

Compile the source files:

```bash
python -m compileall -q src scripts
```

Run the complete test suite:

```bash
python -m pytest tests -q
```

### Latest Verified Result

```text
39 passed, 1 skipped
```

The skipped test requires the real **SAM 2 video predictor** and its external assets.

---

# ⚠️ Current Limitations

The current implementation has several known limitations:

* The classifier was trained on **490 labeled crops**, rather than crops generated from the entire **307-image ArcGIS dataset**.
* The `building` class requires more diverse roof types, scales and lighting conditions.
* Small cars may not be proposed by SAM when external video resolution or flight altitude differs significantly from the training domain.
* SAM automatic mask generation can miss objects.
* SAM may occasionally generate large masks containing mixed semantic content.
* The DroneStock evaluation demonstrates successful external pipeline execution but does not provide formal accuracy measurements without ground-truth labels.
* Model checkpoints and original datasets must be downloaded separately.

---

# 🚀 Future Work

Recommended next steps include:

* [ ] Generate and label crops from more of the **307 ArcGIS source images**
* [ ] Add geographically separated training, validation and test regions
* [ ] Improve `building` class diversity
* [ ] Improve small-object and small-vehicle coverage
* [ ] Tune SAM parameters for different flight altitudes
* [ ] Tune SAM parameters for different video resolutions
* [ ] Evaluate manually annotated external DroneStock frames
* [ ] Add temporal object tracking
* [ ] Add prediction smoothing for continuous video
* [ ] Publish versioned classifier checkpoints through **GitHub Releases**

---

## 🛠️ Technologies

Main technologies used in the project:

* Python
* PyTorch
* SAM 2.1
* MobileNetV3-Small
* torchvision
* OpenCV
* NumPy
* Pydantic
* YAML
* pytest

---

## 📚 Documentation

Additional project documentation is available inside:

```text
docs/
```

Including the detailed external evaluation:

```text
docs/final_evaluation.md
```

---

## ✅ Project Status

The current pipeline successfully supports:

```text
Drone Image
    ↓
SAM Segmentation
    ↓
Mask Filtering
    ↓
Bounding Boxes
    ↓
Object Crops
    ↓
MobileNetV3 Classification
    ↓
Confidence Filtering
    ↓
Visualization
    ↓
JSON Predictions
```

The end-to-end system has been verified on both the development dataset and previously unseen external drone footage.
