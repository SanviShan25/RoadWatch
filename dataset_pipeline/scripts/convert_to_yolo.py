
import os
import xml.etree.ElementTree as ET
from PIL import Image

# Original dataset paths
image_dir = r"C:\Users\KIIT\Desktop\RDD2022_India\India\train\images"
annotation_dir = r"C:\Users\KIIT\Desktop\RDD2022_India\India\train\annotations\xmls"

# Output dataset paths
output_dir = "RoadWatch_YOLO"
output_images = os.path.join(output_dir, "images", "train")
output_labels = os.path.join(output_dir, "labels", "train")

# Create output folders
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Selected classes and their YOLO IDs
class_mapping = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3
}

converted_images = 0
converted_boxes = 0
excluded_images = 0

# Process every XML annotation
for xml_file in os.listdir(annotation_dir):

    if not xml_file.lower().endswith(".xml"):
        continue

    xml_path = os.path.join(annotation_dir, xml_file)
    root = ET.parse(xml_path).getroot()

    # Find corresponding image
    image_name = root.findtext("filename")

    if not image_name:
        image_name = os.path.splitext(xml_file)[0] + ".jpg"

    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print("Missing image:", image_name)
        continue

    # Get image dimensions
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    yolo_annotations = []

    # Convert selected objects
    for obj in root.findall("object"):

        class_name = obj.findtext("name")

        # Ignore classes outside our selected four
        if class_name not in class_mapping:
            continue

        box = obj.find("bndbox")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Convert pixel coordinates to normalized YOLO coordinates
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        class_id = class_mapping[class_name]

        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

        converted_boxes += 1

    # Keep only images with at least one selected-class object
    if not yolo_annotations:
        excluded_images += 1
        continue

    # Copy image to output folder
    output_image_path = os.path.join(output_images, image_name)

    with Image.open(image_path) as img:
        img.save(output_image_path)

    # Create corresponding YOLO label file
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(output_labels, label_name)

    with open(label_path, "w") as file:
        file.write("\n".join(yolo_annotations))

    converted_images += 1


# Final report
print("\n========== YOLO CONVERSION REPORT ==========")
print("Selected classes:", list(class_mapping.keys()))
print("Images converted:", converted_images)
print("Bounding boxes converted:", converted_boxes)
print("Images excluded (no selected-class objects):", excluded_images)
print("Output directory:", output_dir)
