
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original RDD2022 dataset
RDD2022_ROOT = Path(r"C:\Users\KIIT\Desktop\RDD2022_India\India")
original_image_dir = RDD2022_ROOT / "train" / "images"
xml_dir = RDD2022_ROOT / "train" / "annotations" / "xmls"

# Existing positive images and labels
positive_image_dir = PROJECT_ROOT / "data" / "RoadWatch_YOLO" / "images" / "all"
positive_label_dir = PROJECT_ROOT / "data" / "RoadWatch_YOLO" / "labels" / "all"

# Final dataset output
output_image_dir = PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" / "images" / "all"
output_label_dir = PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" / "labels" / "all"

# Check input folders
if not positive_image_dir.exists() or not positive_label_dir.exists():
    raise FileNotFoundError("Converted positive dataset not found.")

if not original_image_dir.exists() or not xml_dir.exists():
    raise FileNotFoundError("Original RDD2022 dataset not found. Check RDD2022_ROOT.")

# Prevent accidental overwriting
if output_image_dir.exists() and any(output_image_dir.iterdir()):
    raise FileExistsError("Final image folder is not empty. Stop and inspect it.")

if output_label_dir.exists() and any(output_label_dir.iterdir()):
    raise FileExistsError("Final label folder is not empty. Stop and inspect it.")

output_image_dir.mkdir(parents=True, exist_ok=True)
output_label_dir.mkdir(parents=True, exist_ok=True)

# Copy positive images and labels
positive_images = list(positive_image_dir.glob("*"))

for image_path in positive_images:
    shutil.copy2(image_path, output_image_dir / image_path.name)

for label_path in positive_label_dir.glob("*.txt"):
    shutil.copy2(label_path, output_label_dir / label_path.name)

# Add genuinely negative images (XML files with no objects)
negative_count = 0

for xml_file in xml_dir.glob("*.xml"):
    root = ET.parse(xml_file).getroot()

    if len(root.findall("object")) == 0:
        image_name = xml_file.stem + ".jpg"
        source_image = original_image_dir / image_name

        if source_image.exists():
            shutil.copy2(source_image, output_image_dir / image_name)

            # Empty YOLO label file represents no objects
            (output_label_dir / (xml_file.stem + ".txt")).write_text(
                "", encoding="utf-8"
            )
            negative_count += 1

print("========== FINAL DATASET PREPARATION ==========")
print("Positive images:", len(positive_images))
print("Negative images added:", negative_count)
print("Total images:", len(list(output_image_dir.glob("*"))))
print("Total labels:", len(list(output_label_dir.glob("*.txt"))))
print("Output:", output_image_dir.parent.parent)
