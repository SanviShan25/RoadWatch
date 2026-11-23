
import os
import xml.etree.ElementTree as ET
from collections import Counter

xml_dir = r"..\RDD2022_India\India\train\annotations\xmls"

selected_classes = {"D00", "D10", "D20", "D40"}

empty_images = 0
excluded_only_images = 0
mixed_images = 0

excluded_class_counts = Counter()

for filename in os.listdir(xml_dir):

    if not filename.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_dir, filename))
    root = tree.getroot()

    classes_in_image = set()

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name:
            classes_in_image.add(name)

    if not classes_in_image:
        empty_images += 1

    elif classes_in_image.isdisjoint(selected_classes):
        excluded_only_images += 1

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name:
                excluded_class_counts[name] += 1

    elif classes_in_image - selected_classes:
        mixed_images += 1

print("========== EXCLUDED IMAGE ANALYSIS ==========")
print("Images with no damage annotations:", empty_images)
print("Images containing only excluded classes:", excluded_only_images)
print("Images containing selected and excluded classes:", mixed_images)

print("\nExcluded class object counts:")
for name, count in excluded_class_counts.items():
    print(f"{name}: {count}")
