# RoadWatch – Dataset Preparation Pipeline

## 1. Project Overview

RoadWatch is a road damage detection project that uses computer vision to identify different types of road damage.

This part of the project focuses on preparing and validating the dataset for YOLO-based object detection.

The dataset is derived from the **RDD2022 (Road Damage Dataset 2022) – India** subset.

The pipeline converts the original XML annotations into YOLO format, selects the required damage classes, includes negative images, creates training/validation/testing splits, and performs dataset quality checks.

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

| Class ID | Damage Class |
|---|---|
| 0 | D00 – Longitudinal Crack |
| 1 | D10 – Transverse Crack |
| 2 | D20 – Alligator Crack |
| 3 | D40 – Pothole |

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

### Step 7: Initial Train, Validation and Test Split

The original dataset was divided into three subsets:

| Subset | Images | Percentage |
|---|---:|---:|
| Training | 5,000 | 70% |
| Validation | 1,428 | 20% |
| Testing | 716 | 10% |
| **Total** | **7,144** | **100%** |

The split was performed with a fixed random seed of 42 for reproducibility. Positive and negative images were split separately.

### Step 8: Duplicate and Near-Duplicate Investigation

An additional investigation was performed to identify potentially similar images across the training, validation and testing subsets.

The following techniques were used:

- **SHA-256 hashing:** Checked for byte-identical images. No exact duplicate groups were found.
- **Difference Hash (dHash):** Identified visually similar image candidates using Hamming distance.
- **SSIM:** Compared structural similarity between candidate images.
- **ORB feature matching:** Compared visual features between image pairs.
- **RANSAC homography:** Checked geometric consistency between matched image features.
- **Visual inspection:** High-priority candidate pairs were manually compared using image contact sheets.

The near-duplicate investigation identified 30 candidate pairs with dHash distance of 0 or 1, forming 15 connected candidate groups across different splits.

Four high-priority pairs were selected for further inspection using ORB and RANSAC results. These pairs showed strong visual similarity and were treated as related road-scene candidates.

### Step 9: Group-Aware Dataset Splitting

To prevent the four visually reviewed image pairs from being distributed across different subsets, a separate group-aware splitting script was created.

The script:

- Reads the confirmed image groups from `confirmed_scene_groups.csv`.
- Keeps images belonging to the same confirmed group together.
- Separately distributes positive and negative image groups.
- Assigns the remaining images individually.
- Preserves all original images and annotation files.
- Creates a separate dataset without modifying the original dataset.

The resulting grouped dataset contains:

| Subset | Images | Positive Images | Negative Images |
|---|---:|---:|---:|
| Training | 5,001 | 2,256 | 2,745 |
| Validation | 1,429 | 645 | 784 |
| Testing | 714 | 322 | 392 |
| **Total** | **7,144** | **3,223** | **3,921** |

All four confirmed groups passed the group-integrity check.

**Note:** This grouping addresses the four visually reviewed candidate pairs. It does not establish that the entire dataset is free from near-duplicate or related-scene leakage. The remaining candidates have not all been manually verified.

## 5. Final Dataset Structure

The project now contains two dataset versions.

### Original dataset

`data/RoadWatch_YOLO_Final/`

### Group-aware dataset

`data/RoadWatch_YOLO_Grouped/`

The grouped dataset is organised as follows:

```text
RoadWatch_YOLO_Grouped/
├── images/
│   ├── all/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── all/
    ├── train/
    ├── val/
    └── test/
```

Both dataset versions are preserved separately.

## 6. YOLO Configuration

Two YAML configuration files are available.

### Original configuration

`dataset_pipeline/data.yaml`

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

### Group-aware configuration

`dataset_pipeline/data_grouped.yaml`

```yaml
path: ../data/RoadWatch_YOLO_Grouped
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

The paths are relative to the location of their respective YAML files.

The grouped configuration can be used for the model-training stage when the team chooses to use the group-aware dataset.

## 7. Scripts Used

The following scripts were created as part of the dataset pipeline.

| Script | Purpose |
|---|---|
| `class_audit.py` | Analyses the distribution of damage classes |
| `analyze_excluded_images.py` | Examines images containing excluded damage classes |
| `validate_dataset.py` | Validates original dataset files and annotations |
| `convert_to_yolo.py` | Converts XML annotations into YOLO format |
| `visualize_annotations.py` | Visualises bounding boxes on images |
| `verify_yolo.py` | Checks converted YOLO annotations |
| `verify_negative_images.py` | Checks negative images and their empty labels |
| `prepare_final_dataset.py` | Combines positive and negative images |
| `split_final_dataset.py` | Creates the original random split |
| `verify_final_dataset.py` | Performs final dataset validation |
| `check_duplicate_images.py` | Checks for byte-identical images using SHA-256 |
| `check_near_duplicates.py` | Finds near-duplicate candidates using dHash |
| `group_near_duplicates.py` | Groups candidate images based on dHash similarity |
| `verify_near_duplicate_pairs.py` | Evaluates image-pair similarity using SSIM, ORB and RANSAC |
| `visualize_image_groups.py` | Creates contact sheets for visual inspection |
| `inspect_strong_matches.py` | Displays high-priority image-pair comparisons |
| `create_grouped_split.py` | Creates the separate group-aware dataset |

## 8. Final Validation Results

The grouped dataset was checked for:

- 7,144 images in the combined dataset.
- 7,144 corresponding label files.
- Matching image and label counts in all subsets.
- Zero missing labels.
- Zero extra labels.
- All four confirmed image groups kept intact.
- All six train, validation and test image/label directories present.
- Valid YAML configuration and dataset paths.
- 6,831 total annotated damage objects.

### Class-wise object distribution in the grouped dataset

| Damage Class | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| D00 | 1,092 | 319 | 144 | 1,555 |
| D10 | 47 | 13 | 8 | 68 |
| D20 | 1,411 | 399 | 211 | 2,021 |
| D40 | 2,260 | 609 | 318 | 3,187 |
| **Total** | **4,810** | **1,340** | **681** | **6,831** |

The class-wise totals match the original selected-class annotation counts.

## 9. Important Dataset Limitations

1. **Class imbalance:** D10 has only 68 annotated objects in the entire dataset, which may affect detection performance for this class.

2. **Excluded classes:** Images containing both selected and excluded damage classes were retained, but annotations for excluded classes were not included. Consequently, some unlabelled damage objects may be treated as background during training.

3. **Test split:** The test subset was created from the original labelled training data. The official RDD2022 test images were not used because corresponding annotations were unavailable in the dataset used for this pipeline.

4. **Near-duplicate investigation:** Four high-priority image pairs were visually reviewed and kept within their respective splits. Other near-duplicate candidates may remain, so complete elimination of related-scene leakage has not been established.

5. **Test-set limitations:** The D10 class has only 8 annotated objects in the grouped test subset. Therefore, its evaluation metrics may be unstable.

## 10. Handover to Model Training

Two prepared dataset configurations are available for the YOLO model-training stage.

- **Original dataset:** `data/RoadWatch_YOLO_Final/`
- **Original configuration:** `dataset_pipeline/data.yaml`
- **Group-aware dataset:** `data/RoadWatch_YOLO_Grouped/`
- **Group-aware configuration:** `dataset_pipeline/data_grouped.yaml`

The team can select the appropriate YAML configuration for YOLO training and evaluation.

---

**Dataset preparation, preprocessing, validation and group-aware splitting completed as part of the RoadWatch project.**