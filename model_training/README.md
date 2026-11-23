# Model Training (Sanvi)

This folder covers **model training and evaluation** for RoadWatch. It trains a YOLO road-damage detector on
Manasvi's group-aware dataset (`RoadWatch_YOLO_Grouped`, see the main [README](../README.md) and
[`dataset_pipeline/data_grouped.yaml`](../dataset_pipeline/data_grouped.yaml)).

Classes: `0 D00` longitudinal crack · `1 D10` transverse crack · `2 D20` alligator crack · `3 D40` pothole.

## Folder layout

```text
model_training/
├── README.md          this file
├── notebooks/
│   └── 01_baseline.ipynb   YOLOv8n baseline: download data, train, resume, evaluate (Colab)
├── scripts/
│   ├── evaluate.py         test-split metrics (overall + per class) -> JSON
│   └── benchmark_cpu.py    CPU speed (ms/image) and file size, optional ONNX export
├── reports/           metrics JSON files and plots that are committed
└── model_card.json    model card template; fill in the <FILL IN> values after training
```

Weights (`*.pt`, `*.onnx`) and `runs/` are git-ignored. Training runs are stored on Google Drive under
`MyDrive/RoadWatch/runs/`.

## Running the notebook (Google Colab)

1. Open `notebooks/01_baseline.ipynb` in Colab (*File → Upload notebook*, or open it from GitHub).
2. *Runtime → Change runtime type → **T4 GPU***.
3. Run the cells from top to bottom:
   - **a** installs `ultralytics` and `gdown` and mounts Google Drive.
   - **b** downloads the dataset zip and unzips it to `/content/data`. If `gdown` fails, use
     *"Add shortcut to Drive"* and the fallback cell.
   - **c** finds the dataset root and prints counts (expected about 5001 / 1429 / 714 images).
   - **d** writes `/content/data.yaml`.
   - **e** trains YOLOv8n (50 epochs, imgsz 640, batch 16, seed 42, patience 15).
   - **f** resumes from `last.pt` if Colab disconnected. Re-run a–d first.
   - **g** evaluates `best.pt` on the **test** split and saves `test_metrics.json` next to the run.
4. Copy the metrics JSON into `reports/` and fill in `model_card.json`.

## Running the scripts (locally or in Colab)

Requires `pip install ultralytics`. The dataset YAML must have a `test:` entry. In Colab, use `/content/data.yaml`.
Locally, make a copy of `dataset_pipeline/data_grouped.yaml` with `path:` set to the **absolute** path of
`RoadWatch_YOLO_Grouped`. Ultralytics resolves a relative `path:` from the current working directory, not from
the YAML file.

```bash
# Test-split metrics -> model_training/reports/v8n_baseline_test_metrics.json
python model_training/scripts/evaluate.py \
    --weights path/to/v8n_baseline/weights/best.pt \
    --data path/to/data.yaml

# CPU speed over 100 test images (+ ONNX export and benchmark)
python model_training/scripts/benchmark_cpu.py \
    --weights path/to/v8n_baseline/weights/best.pt \
    --images data/RoadWatch_YOLO_Grouped/images/test \
    --num-images 100 --export-onnx
```

## Notes

- **D10** has only 8 objects in the test split, so its AP will be unstable. Report it with that caveat.
- Use the **test** split only for final numbers. Model selection (best epoch) uses the val split.
