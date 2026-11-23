
from pathlib import Path
import random
import shutil

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Final combined dataset
source_image_dir = (
    PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" / "images" / "all"
)
source_label_dir = (
    PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" / "labels" / "all"
)

# Split configuration
random.seed(42)

train_ratio = 0.70
val_ratio = 0.20

splits = {
    "train": [],
    "val": [],
    "test": []
}


def distribute_files(files):
    files = sorted(files)
    random.shuffle(files)

    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }


# Check source folders
if not source_image_dir.exists() or not source_label_dir.exists():
    raise FileNotFoundError("Final combined dataset not found.")

# Identify positive and negative images using label files
positive_files = []
negative_files = []

for label_path in sorted(source_label_dir.glob("*.txt")):
    image_name = label_path.stem + ".jpg"
    image_path = source_image_dir / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_name}")

    if label_path.stat().st_size == 0:
        negative_files.append(image_name)
    else:
        positive_files.append(image_name)

print("Positive images:", len(positive_files))
print("Negative images:", len(negative_files))

# Split positive and negative images separately
positive_splits = distribute_files(positive_files)
negative_splits = distribute_files(negative_files)

# Combine categories in each split
for split in splits:
    splits[split] = (
        positive_splits[split] + negative_splits[split]
    )
    random.shuffle(splits[split])

# Create split folders
for split in splits:
    image_output = (
        PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" /
        "images" / split
    )
    label_output = (
        PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" /
        "labels" / split
    )

    # Prevent accidental mixing with old split files
    if image_output.exists() and any(image_output.iterdir()):
        raise FileExistsError(f"{split} image folder is not empty.")

    if label_output.exists() and any(label_output.iterdir()):
        raise FileExistsError(f"{split} label folder is not empty.")

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for filename in splits[split]:
        label_name = Path(filename).stem + ".txt"

        shutil.copy2(
            source_image_dir / filename,
            image_output / filename
        )

        shutil.copy2(
            source_label_dir / label_name,
            label_output / label_name
        )

    print(
        f"{split.upper()}: {len(splits[split])} images | "
        f"Positive: {len(positive_splits[split])} | "
        f"Negative: {len(negative_splits[split])}"
    )

print("\n========== FINAL DATASET SPLIT ==========")
print("Total images:", sum(len(files) for files in splits.values()))
print("Random seed: 42")
print("Split completed!")
