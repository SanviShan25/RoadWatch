
import os
import xml.etree.ElementTree as ET
from collections import Counter

annotation_dir = r"C:\Users\KIIT\Desktop\RDD2022_India\India\train\annotations\xmls"

class_counts = Counter()
total_objects = 0
missing_labels = 0

for filename in os.listdir(annotation_dir):

    if not filename.lower().endswith(".xml"):
        continue

    path = os.path.join(annotation_dir, filename)
    root = ET.parse(path).getroot()

    for obj in root.findall("object"):
        total_objects += 1

        name_element = obj.find("name")

        if name_element is None or not name_element.text:
            missing_labels += 1
            class_counts["<MISSING OR EMPTY LABEL>"] += 1
        else:
            # repr helps reveal hidden spaces or unusual characters
            class_name = name_element.text
            class_counts[class_name] += 1

print("COMPLETE CLASS AUDIT")
print("--------------------")

for class_name, count in sorted(class_counts.items()):
    print(f"{repr(class_name)} : {count}")

print("\nTotal objects:", total_objects)
print("Sum of class counts:", sum(class_counts.values()))
print("Missing/empty labels:", missing_labels)
print("Unique labels:", len(class_counts))
