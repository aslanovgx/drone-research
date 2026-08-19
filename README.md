SAM-Based Drone Object Segmentation and Classification

This project implements an end-to-end pipeline for detecting and classifying objects in aerial drone imagery.

The system uses SAM 2.1 to generate candidate object masks. Each valid mask is converted into a bounding box and a crop. A MobileNetV3-Small classifier then assigns one of four semantic classes to every crop.

Pipeline

Drone image or extracted video frame
        ↓
SAM 2.1 mask generation
        ↓
Mask filtering, deduplication and boundary stitching
        ↓
Bounding-box and crop generation
        ↓
MobileNetV3-Small classification
        ↓
Class label and confidence filtering
        ↓
Annotated image and JSON output

The final output contains:

* segmentation candidates;
* bounding boxes;
* class labels;
* classifier confidence scores;
* SAM confidence scores;
* annotated images;
* structured JSON prediction files.

Classes

The classifier currently supports four classes:

Class	Description
building	Building roofs and dominant building structures
car	Cars and other clearly visible road vehicles
tree	Trees and dominant vegetation regions
other	Roads, shadows, HVAC units, poles, signs and other non-target regions

The other class prevents every SAM region from being forced into a target object class.

Data

The ArcGIS Packing House District drone dataset was used for classifier dataset preparation.

Source images were divided geographically before crop generation to reduce spatial leakage between training and validation data.

Split	Building	Car	Tree	Other	Total
Training	49	63	57	118	287
Validation	39	25	42	97	203
Total	88	88	99	215	490

The repository includes reproducibility manifests for:

* geographic source-image splitting;
* source-patch selection;
* crop labels and their original SAM crop paths.

The original images, generated crops, model checkpoints and videos are not committed because of their size and licensing constraints.

Classifier Results

The selected baseline is an unweighted MobileNetV3-Small classifier initialized with ImageNet weights.

Validation results:

Metric	Result
Accuracy	75.86%
Macro F1	0.7717

Per-class results:

Class	Precision	Recall	F1
Building	0.6250	0.6410	0.6329
Car	0.8000	0.9600	0.8727
Other	0.7952	0.6804	0.7333
Tree	0.7800	0.9286	0.8478

The strongest validation performance is currently obtained for cars and trees. Building classification remains more sensitive to roof scale, surrounding vegetation, shadows and domain shift.

External DroneStock Evaluation

Two independently selected stock drone videos were used as external, unseen-domain tests:

* a suburban residential flyover;
* a top-down Central Park aerial video.

Frames were extracted at approximately 20%, 50% and 80% of each video. All six frames completed the full pipeline successfully.

Scene	Frame	SAM segments	Final detections	Building	Car	Tree	Other
Central Park	20%	5	2	0	0	1	1
Central Park	50%	6	2	0	0	0	2
Central Park	80%	5	1	0	0	0	1
Suburbs	20%	13	6	0	0	6	0
Suburbs	50%	26	14	0	0	8	6
Suburbs	80%	21	10	0	0	5	5

These runs verify that the complete system works on previously unseen drone footage. They are not formal external accuracy measurements because the stock-video frames do not have ground-truth annotations.

See docs/final_evaluation.md for the detailed evaluation, interpretation and limitations.

Repository Structure

configs/                 SAM, classifier and pipeline configurations
data/manifests/          Reproducible source splits and crop labels
docs/                    Documentation and evaluation report
requirements/            Module-specific Python dependencies
scripts/                 Dataset and labeling workflow utilities
src/classification/      Classifier training and inference
src/data/                Dataset analysis and preprocessing
src/segmentation/        SAM loading, tiling, filtering and crop export
src/visualization/       Prediction visualization
src/pipeline.py          Prediction merging and output generation
src/run_pipeline.py      End-to-end command-line runner
tests/                   Unit and integration tests

Installation

Python 3.12 was used during development.

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

Install the project dependencies:

python -m pip install -r requirements/data.txt
python -m pip install -r requirements/sam.txt
python -m pip install -r requirements/classifier.txt
python -m pip install -r requirements/integration.txt

Install the official SAM 2 package:

git clone https://github.com/facebookresearch/sam2.git sam2_repo
SAM2_BUILD_CUDA=0 python -m pip install -e ./sam2_repo

Model Checkpoints

Create the checkpoint directory:

mkdir -p checkpoints

Download the official SAM 2.1 Hiera Small checkpoint:

curl -L --fail \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" \
  -o checkpoints/sam2.1_hiera_small.pt

The trained classifier checkpoint must be placed at:

checkpoints/classifier.pt

Model checkpoint files are excluded from Git because of their size. The classifier baseline should be downloaded from the project’s GitHub Release assets or provided separately.

Run the End-to-End Pipeline

Run the default pipeline on an image:

PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m src.run_pipeline \
  path/to/drone_image.jpg \
  --sam-config configs/sam.yaml \
  --classifier-config configs/classifier.yaml \
  --pipeline-config configs/pipeline.yaml \
  --checkpoint checkpoints/classifier.pt

Run the configuration used for the external DroneStock frames:

PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m src.run_pipeline \
  path/to/extracted_frame.jpg \
  --sam-config configs/sam_dronestock.yaml \
  --classifier-config configs/classifier.yaml \
  --pipeline-config configs/pipeline_dronestock.yaml \
  --checkpoint checkpoints/classifier.pt

PYTORCH_ENABLE_MPS_FALLBACK=1 is useful on Apple Silicon. It may be omitted on CPU or CUDA systems.

Generated files are written to the output directories configured in the SAM and pipeline YAML files.

Typical outputs include:

outputs/crops/
outputs/segmentation_results.json
outputs/json/<image_name>.json
outputs/predictions/<image_name>_annotated.jpg

Train the Classifier

The dataset must follow this directory structure:

data/classifier/
├── train/
│   ├── building/
│   ├── car/
│   ├── other/
│   └── tree/
└── validation/
    ├── building/
    ├── car/
    ├── other/
    └── tree/

Start training:

PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m src.classification.train \
  --config configs/classifier.yaml \
  --device auto

The best checkpoint is saved according to the checkpoint path in configs/classifier.yaml.

Tests

Run the complete test suite:

python -m compileall -q src scripts
python -m pytest tests -q

Latest verified result:

39 passed, 1 skipped

The skipped test requires the real SAM 2 video predictor and its external assets.

Current Limitations

* The classifier was trained on 490 labeled crops rather than the entire 307-image ArcGIS dataset.
* The building class requires more diverse roofs, scales and lighting conditions.
* Small cars may not be proposed by SAM when the external video resolution or viewing altitude differs significantly from the training data.
* SAM automatic mask generation can miss objects or produce large mixed-content regions.
* The DroneStock evaluation demonstrates external execution but does not provide formal accuracy without ground-truth labels.
* Model checkpoints and original datasets must be downloaded separately.

Future Work

Recommended next steps:

* generate and label crops from more of the 307 ArcGIS source images;
* add more geographically separated training, validation and test regions;
* improve building and small-object coverage;
* tune SAM parameters for different video resolutions and flight altitudes;
* evaluate on manually annotated external DroneStock frames;
* add temporal tracking and smoothing for continuous video output;
* publish versioned classifier checkpoints through GitHub Releases.