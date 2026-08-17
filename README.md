# SAM-Based Drone Object Segmentation and Classification

This project processes raw drone imagery using the Segment Anything
Model (SAM). SAM automatically generates candidate object masks.
Each mask is converted into a bounding box and image crop. A trained
classifier then assigns a semantic class to each region.

The system produces:

- segmentation masks;
- bounding boxes;
- object class labels;
- confidence scores;
- annotated images and videos;
- JSON prediction files.

## Data

- ArcGIS Packing House District: development and model preparation
- DroneStock footage: external testing on unseen drone scenes

## Pipeline

Raw Image → SAM → Mask Filtering → Bounding Box → Crop →
Classifier → Class + Confidence → Visualization