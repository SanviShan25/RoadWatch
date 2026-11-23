
from pathlib import Path
from collections import Counter
import math

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Final dataset directory
base_dir = PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final"

class_names = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40"
}

total_images = 0
total_boxes = 0
total_errors = 0

for split in ["train", "val", "test"]:

    image_dir = base_dir / "images" / split
    label_dir = base_dir / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing folders for {split}")

    images = {
        f.stem for f in image_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }

    labels = {
        f.stem for f in label_dir.glob("*.txt")
    }

    missing_labels = images - labels
    extra_labels = labels - images

    class_counts = Counter()
    empty_labels = 0
    positive_labels = 0
    invalid_boxes = 0

    for label_path in label_dir.glob("*.txt"):

        with open(label_path, "r") as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            empty_labels += 1
            continue

        positive_labels += 1

        for line in lines:
            try:
                values = line.split()

                if len(values) != 5:
                    raise ValueError("Expected 5 values")

                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])

                if class_id not in class_names:
                    raise ValueError("Invalid class ID")

                if not all(math.isfinite(v) for v in
                           [x_center, y_center, width, height]):
                    raise ValueError("Non-finite coordinate")

                if not (
                    0 <= x_center <= 1
                    and 0 <= y_center <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                ):
                    raise ValueError("Invalid bounding box coordinates")

                class_counts[class_id] += 1

            except (ValueError, OverflowError):
                invalid_boxes += 1

    image_count = len(images)
    total_images += image_count
    total_boxes += sum(class_counts.values())
    total_errors += (
        len(missing_labels) + len(extra_labels) + invalid_boxes
    )

    print(f"\n========== {split.upper()} ==========")
    print("Images:", image_count)
    print("Label files:", len(labels))
    print("Missing labels:", len(missing_labels))
    print("Extra labels:", len(extra_labels))
    print("Positive labels:", positive_labels)
    print("Empty labels:", empty_labels)
    print("Invalid boxes:", invalid_boxes)

    for class_id, name in class_names.items():
        print(f"{name}: {class_counts[class_id]} objects")

print("\n========== FINAL SUMMARY ==========")
print("Total images:", total_images)
print("Total bounding boxes:", total_boxes)
print("Total validation errors:", total_errors)
