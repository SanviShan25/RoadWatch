
from pathlib import Path
import csv
import random
import shutil

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data" / "RoadWatch_YOLO_Final"
OUTPUT = ROOT / "data" / "RoadWatch_YOLO_Grouped"

IMAGE_SOURCE = SOURCE / "images" / "all"
LABEL_SOURCE = SOURCE / "labels" / "all"

GROUP_FILE = ROOT / "confirmed_scene_groups.csv"

# --------------------------------------------------
# 2. SPLIT CONFIGURATION
# --------------------------------------------------

random.seed(42)

RATIOS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10
}

# --------------------------------------------------
# 3. CHECK SOURCE FILES
# --------------------------------------------------

if not IMAGE_SOURCE.exists() or not LABEL_SOURCE.exists():
    raise FileNotFoundError("Original combined dataset not found.")

if not GROUP_FILE.exists():
    raise FileNotFoundError("confirmed_scene_groups.csv not found.")

if OUTPUT.exists():
    raise FileExistsError(
        "Grouped output folder already exists. "
        "Do not overwrite it. Inspect it first."
    )

# --------------------------------------------------
# 4. READ ALL IMAGE-LABEL PAIRS
# --------------------------------------------------

all_images = {}

for label_path in sorted(LABEL_SOURCE.glob("*.txt")):
    image_name = label_path.stem + ".jpg"
    image_path = IMAGE_SOURCE / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_name}")

    all_images[image_name] = (
        label_path.stat().st_size > 0
    )

if len(all_images) != 7144:
    raise ValueError(
        f"Expected 7144 images, found {len(all_images)}"
    )

# --------------------------------------------------
# 5. READ CONFIRMED GROUPS
# --------------------------------------------------

groups = {}

with open(GROUP_FILE, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        group_id = row["group_id"]
        image_name = row["image"]

        if image_name not in all_images:
            raise ValueError(
                f"Grouped image not found: {image_name}"
            )

        groups.setdefault(group_id, []).append(image_name)

grouped_images = set()

for group_id, members in groups.items():
    if len(members) < 2:
        raise ValueError(f"Group {group_id} has fewer than 2 images.")

    for image_name in members:
        if image_name in grouped_images:
            raise ValueError(
                f"Image appears in multiple groups: {image_name}"
            )

        grouped_images.add(image_name)

print("Confirmed groups:", len(groups))
print("Images inside groups:", len(grouped_images))

# --------------------------------------------------
# 6. CREATE INDIVISIBLE UNITS
# --------------------------------------------------

positive_units = []
negative_units = []

# Confirmed groups stay together
for members in groups.values():
    positive = all(all_images[name] for name in members)
    negative = all(not all_images[name] for name in members)

    if not positive and not negative:
        raise ValueError(
            "A group contains both positive and negative images."
        )

    if positive:
        positive_units.append(members)
    else:
        negative_units.append(members)

# Remaining images become individual units
for image_name, is_positive in all_images.items():
    if image_name in grouped_images:
        continue

    if is_positive:
        positive_units.append([image_name])
    else:
        negative_units.append([image_name])

# --------------------------------------------------
# 7. DISTRIBUTE UNITS WITHOUT BREAKING GROUPS
# --------------------------------------------------

def distribute_units(units):
    random.shuffle(units)

    total_images = sum(len(unit) for unit in units)

    targets = {
        split: total_images * ratio
        for split, ratio in RATIOS.items()
    }

    assigned = {
        split: []
        for split in RATIOS
    }

    counts = {
        split: 0
        for split in RATIOS
    }

    # Larger groups are assigned first
    units.sort(key=len, reverse=True)

    for unit in units:
        # Assign the unit to the split with
        # the largest remaining target deficit
        chosen_split = max(
            RATIOS,
            key=lambda split: targets[split] - counts[split]
        )

        assigned[chosen_split].extend(unit)
        counts[chosen_split] += len(unit)

    return assigned


positive_splits = distribute_units(positive_units)
negative_splits = distribute_units(negative_units)

splits = {}

for split in RATIOS:
    splits[split] = (
        positive_splits[split] +
        negative_splits[split]
    )
    random.shuffle(splits[split])

# --------------------------------------------------
# 8. CREATE OUTPUT DATASET
# --------------------------------------------------

for category in ["images", "labels"]:
    for split in ["all", "train", "val", "test"]:
        (OUTPUT / category / split).mkdir(
            parents=True,
            exist_ok=True
        )

# Copy complete combined dataset
for image_name in all_images:
    label_name = Path(image_name).stem + ".txt"

    shutil.copy2(
        IMAGE_SOURCE / image_name,
        OUTPUT / "images" / "all" / image_name
    )

    shutil.copy2(
        LABEL_SOURCE / label_name,
        OUTPUT / "labels" / "all" / label_name
    )

# Copy split files
for split, filenames in splits.items():
    for image_name in filenames:
        label_name = Path(image_name).stem + ".txt"

        shutil.copy2(
            IMAGE_SOURCE / image_name,
            OUTPUT / "images" / split / image_name
        )

        shutil.copy2(
            LABEL_SOURCE / label_name,
            OUTPUT / "labels" / split / label_name
        )

# --------------------------------------------------
# 9. VERIFY GROUPS
# --------------------------------------------------

image_to_split = {}

for split, filenames in splits.items():
    for image_name in filenames:
        image_to_split[image_name] = split

for group_id, members in groups.items():
    assigned_splits = {
        image_to_split[name]
        for name in members
    }

    if len(assigned_splits) != 1:
        raise RuntimeError(
            f"Group leakage detected in {group_id}"
        )

# --------------------------------------------------
# 10. FINAL REPORT
# --------------------------------------------------

print("\n========== GROUPED DATASET SPLIT ==========")

for split in RATIOS:
    positive_count = len(positive_splits[split])
    negative_count = len(negative_splits[split])
    total = len(splits[split])

    print(f"\n{split.upper()}")
    print("Total images:", total)
    print("Positive images:", positive_count)
    print("Negative images:", negative_count)

print("\nTotal images:", sum(len(x) for x in splits.values()))
print("Confirmed groups kept intact:", len(groups))
print("All group checks passed.")
print("Output:", OUTPUT)
