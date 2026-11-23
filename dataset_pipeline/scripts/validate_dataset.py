
import os
import xml.etree.ElementTree as ET
from PIL import Image

# Dataset paths
image_dir = r"C:\Users\KIIT\Desktop\RDD2022_India\India\train\images"
annotation_dir = r"C:\Users\KIIT\Desktop\RDD2022_India\India\train\annotations\xmls"

# Counters
total_xml = 0
valid_xml = 0
invalid_xml = 0
valid_images = 0
corrupted_images = 0
total_boxes = 0
invalid_boxes = 0

issues = []

# Check every XML annotation
for xml_file in os.listdir(annotation_dir):

    if not xml_file.lower().endswith(".xml"):
        continue

    total_xml += 1
    xml_path = os.path.join(annotation_dir, xml_file)

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        invalid_xml += 1
        issues.append(f"XML parsing error: {xml_file} - {e}")
        continue

    # Get corresponding image filename
    image_name = root.findtext("filename")

    if not image_name:
        image_name = os.path.splitext(xml_file)[0] + ".jpg"

    image_path = os.path.join(image_dir, image_name)

    # Check whether image exists and opens
    if not os.path.exists(image_path):
        corrupted_images += 1
        issues.append(f"Missing image: {image_name}")
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        # Reopen to read dimensions after verification
        with Image.open(image_path) as img:
            width, height = img.size

        valid_images += 1

    except Exception as e:
        corrupted_images += 1
        issues.append(f"Image error: {image_name} - {e}")
        continue

    xml_has_invalid_box = False

    # Check every bounding box
    for obj in root.findall("object"):
        total_boxes += 1
        box = obj.find("bndbox")

        if box is None:
            invalid_boxes += 1
            xml_has_invalid_box = True
            issues.append(f"Missing bounding box: {xml_file}")
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))

            if (
                xmin >= xmax
                or ymin >= ymax
                or xmin < 0
                or ymin < 0
                or xmax > width
                or ymax > height
            ):
                invalid_boxes += 1
                xml_has_invalid_box = True
                issues.append(
                    f"Invalid box in {xml_file}: "
                    f"({xmin}, {ymin}, {xmax}, {ymax})"
                )

        except (TypeError, ValueError):
            invalid_boxes += 1
            xml_has_invalid_box = True
            issues.append(f"Non-numeric box coordinates: {xml_file}")

    if xml_has_invalid_box:
        invalid_xml += 1
    else:
        valid_xml += 1


# Final report
print("\n========== DATASET VALIDATION REPORT ==========")
print("Total XML files:", total_xml)
print("Valid XML annotations:", valid_xml)
print("XML files with invalid boxes:", invalid_xml)
print("Images successfully opened:", valid_images)
print("Missing/corrupted images:", corrupted_images)
print("Total bounding boxes:", total_boxes)
print("Invalid bounding boxes:", invalid_boxes)

print("\n========== ISSUES FOUND ==========")

if issues:
    for issue in issues[:30]:
        print(issue)

    if len(issues) > 30:
        print("... and", len(issues) - 30, "more issues.")
else:
    print("No issues found!")
