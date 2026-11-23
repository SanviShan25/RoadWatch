# RoadWatch – Dataset Preparation Pipeline

## 1. Project Overview

RoadWatch is a road damage detection project that uses computer vision to identify different types of road damage.

This part of the project focuses on preparing and validating the dataset for YOLO-based object detection.

The dataset is derived from the **RDD2022 (Road Damage Dataset 2022) – India** subset.

The pipeline converts the original XML annotations into YOLO format, selects the required damage classes, includes negative images, creates training/validation/testing splits, and verifies the final dataset.

## 2. Dataset Source

- Dataset: RDD2022
- Subset: India
- Original image directory: `RDD2022_India/India/train/images`
- Original annotation directory: `RDD2022_India/India/train/annotations/xmls`

The original training dataset contained:

- 7,706 images
- 7,706 XML annotation files
- 8,203 annotated damage objects
- 10 original damage classes

## 3. Selected Damage Classes

Four damage classes were selected for this project.

| Class ID | Damage Class             |
| -------- | ------------------------ |
| 0        | D00 – Longitudinal Crack |
| 1        | D10 – Transverse Crack   |
| 2        | D20 – Alligator Crack    |
| 3        | D40 – Pothole            |

The selected classes account for 6,831 annotated objects in the original dataset.

Other damage classes were excluded from the detection task.

## 4. Dataset Preparation Process

### Step 1: Dataset Inspection

The original dataset was inspected to understand its directory structure, image files, XML annotations, and damage class distribution.

### Step 2: Annotation Validation

XML annotation files were checked for:

- Valid XML structure
- Missing or empty annotations
- Valid bounding-box coordinates
- Corresponding image availability

All 7,706 XML files were valid, and all 7,706 images could be opened.

### Step 3: Class Selection

Only D00, D10, D20, and D40 were retained for the project.

Images containing at least one selected damage class were used for the positive dataset.

### Step 4: XML-to-YOLO Conversion

The original annotations were stored in Pascal VOC XML format.

They were converted into YOLO text format.

Each YOLO annotation follows this structure:

`class_id x_center y_center width height`

Bounding-box coordinates are normalized between 0 and 1.

### Step 5: Negative Image Preparation

Images without any selected damage annotations were also included.

These images were assigned empty YOLO label files. They help the model learn to distinguish between road damage and undamaged road scenes.

### Step 6: Final Dataset Creation

The positive and negative images were combined into one final dataset.

The final dataset contains:

- 3,223 positive images
- 3,921 negative images
- 7,144 total images
- 7,144 corresponding label files

### Step 7: Train, Validation and Test Split

The final dataset was divided into three subsets:

| Subset     |    Images | Percentage |
| ---------- | --------: | ---------: |
| Training   |     5,000 |        70% |
| Validation |     1,428 |        20% |
| Testing    |       716 |        10% |
| **Total**  | **7,144** |   **100%** |

The split was performed with a fixed random seed of 42 for reproducibility. Positive and negative images were split separately.

## 5. Final Dataset Structure

The final dataset is organised as follows:

```text
RoadWatch/
│
├── data/
│   └── RoadWatch_YOLO_Final/
│       ├── images/
│       │   ├── all/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       │
│       └── labels/
│           ├── all/
│           ├── train/
│           ├── val/
│           └── test/
│
└── dataset_pipeline/
    ├── data.yaml
    └── scripts/
```

## 6. YOLO Configuration

The `data.yaml` file specifies the dataset paths, number of classes, and class names.

```yaml
path: ../data/RoadWatch_YOLO_Final
train: images/train
val: images/val
test: images/test

nc: 4

names:
  0: D00
  1: D10
  2: D20
  3: D40
```

The paths are relative to the location of `data.yaml`.

## 7. Scripts Used

The following scripts were created as part of the dataset pipeline:

| Script                       | Purpose                                            |
| ---------------------------- | -------------------------------------------------- |
| `class_audit.py`             | Analyses the distribution of damage classes        |
| `analyze_excluded_images.py` | Examines images containing excluded damage classes |
| `validate_dataset.py`        | Validates original dataset files and annotations   |
| `convert_to_yolo.py`         | Converts XML annotations into YOLO format          |
| `visualize_annotations.py`   | Visualises bounding boxes on images                |
| `verify_yolo.py`             | Checks converted YOLO annotations                  |
| `verify_negative_images.py`  | Checks negative images and their empty labels      |
| `prepare_final_dataset.py`   | Combines positive and negative images              |
| `split_final_dataset.py`     | Creates train, validation and test subsets         |
| `verify_final_dataset.py`    | Performs final dataset validation                  |

## 8. Final Validation Results

The final dataset verification confirmed:

- 7,144 images present.
- 7,144 corresponding label files present.
- Training, validation and testing image/label counts match.
- No missing labels.
- No extra labels.
- No invalid bounding boxes.
- 6,831 total annotated damage objects.
- Zero validation errors.

All images were found to have dimensions of 720 × 720 pixels during the earlier image-dimension analysis.

## 9. Important Dataset Limitations

1. **Class imbalance:** D10 has only 68 annotated objects in the entire dataset, which may affect detection performance for this class.

2. **Excluded classes:** Images containing both selected and excluded damage classes were retained, but annotations for excluded classes were not included. Consequently, some unlabelled damage objects may be treated as background during training.

3. **Test split:** The test subset was created from the original labelled training data. The official RDD2022 test images were not used because corresponding annotations were unavailable in the dataset used for this pipeline.

4. **Duplicate checking:** Cross-split duplicate and near-duplicate image checking has not yet been completed.

## 10. Handover to Model Training

The prepared dataset is ready to be used for YOLO model training after the remaining dataset quality checks.

- **Dataset directory:** `data/RoadWatch_YOLO_Final/`
- **YOLO configuration:** `dataset_pipeline/data.yaml`

The model-training stage can use the YAML configuration to locate the training, validation and test datasets.

---

**Dataset preparation and preprocessing completed as part of the RoadWatch project.**
