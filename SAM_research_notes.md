# SAM Research Notes

## 1. Project Goal

The project focuses on image and video segmentation for drone imagery.

The provided dataset is the **Packing House District** drone dataset from Esri. The dataset will be used to prepare and train/fine-tune our segmentation model. The final model will be evaluated on unseen drone stock videos.

## 2. SAM Versions

### SAM 1

The original Segment Anything Model (SAM) provides three main image encoder variants:

- `vit_b` — Vision Transformer Base
- `vit_l` — Vision Transformer Large
- `vit_h` — Vision Transformer Huge

The main difference between them is the model size, computational cost, speed, and segmentation quality.

- `vit_b`: fastest and requires the least computational resources.
- `vit_l`: better segmentation quality with higher computational requirements.
- `vit_h`: largest and most computationally expensive, with higher quality potential.

Because the final evaluation will use video, SAM 1 is considered mainly as a baseline/reference rather than the primary model.

## 3. SAM 2.1

SAM 2.1 is designed for both image and video segmentation. Unlike the original SAM, it provides a memory mechanism that allows segmentation information to be propagated across video frames.

SAM 2.1 provides several Hiera-based model variants:

- `Hiera-Tiny`
- `Hiera-Small`
- `Hiera-Base+`
- `Hiera-Large`

Since the final evaluation will be performed using drone videos, SAM 2.1 is currently considered the primary model family for the project.

## 4. Initial Model Selection

Our current plan is to test:

1. SAM 2.1 Hiera-Tiny
2. SAM 2.1 Hiera-Small
3. SAM 2.1 Hiera-Base+

The models will be evaluated using the same dataset and validation procedure.

The main metrics will include:

- Segmentation quality
- Inference time
- GPU memory usage
- Training/fine-tuning time
- Video segmentation stability

The final model will be selected based on the best balance between segmentation quality and computational cost.

## 5. Adaptive Model Selection

An additional research idea is to use a dynamic model-selection approach.

Instead of always using the largest model, the system could initially process an image/frame using a smaller model and only use a larger model when the segmentation quality is insufficient.

Example:

Hiera-Tiny → quality check → Hiera-Small if necessary

This approach will only be used if experiments show that it provides a meaningful reduction in computational cost without significantly reducing segmentation quality.

## 6. Hardware

Development will be performed on a laptop with:

- GPU: NVIDIA RTX 3050 Ti
- VRAM: 6 GB

Because of the limited VRAM, Hiera-Large is not currently considered a practical primary model for local training/inference.

## 7. Dataset

Dataset:

**Packing House District — Esri Drone Dataset**

Source:

https://www.esri.com/en-us/arcgis/products/arcgis-reality/resources/sample-drone-datasets

The dataset will be inspected before deciding the final training strategy.

Important information to determine:

- Number of images
- Image resolution
- Available annotations
- Object/classes to segment
- Image overlap
- Flight/scene structure
- Suitable train/validation/test split

The dataset should not be randomly split if consecutive images originate from the same flight/scene, because this could cause data leakage.

## 8. Training Strategy

The project will use a pretrained SAM 2.1 checkpoint as the starting point and adapt it to the Packing House District dataset.

The exact training strategy will be determined after inspecting the dataset and annotations.

> Note: The term "training" used in the project may refer to adapting/fine-tuning the pretrained SAM 2.1 model rather than training the foundation model completely from scratch.

## 9. Final Evaluation

After training, the selected model will be tested on previously unseen drone stock videos.

The goal is to evaluate how well the model generalizes from the Packing House District drone imagery to different real-world drone video footage.

## 10. Current Decision

**Primary direction:** SAM 2.1

**First models to investigate:** Hiera-Tiny and Hiera-Small

**Potential advanced approach:** Hiera-Tiny → Hiera-Small adaptive routing

**Baseline/reference:** Original SAM 1 (ViT-B/L/H)

The adaptive approach will not be assumed to be more efficient beforehand. It must be validated experimentally against using Hiera-Small alone.