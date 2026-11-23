
import os
import xml.etree.ElementTree as ET

xml_dir = r"..\RDD2022_India\India\train\annotations\xmls"
image_dir = r"..\RDD2022_India\India\train\images"

negative_count = 0
missing_images = []

for filename in os.listdir(xml_dir):

    if not filename.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, filename)
    root = ET.parse(xml_path).getroot()

    if len(root.findall("object")) == 0:

        image_name = filename.replace(".xml", ".jpg")
        image_path = os.path.join(image_dir, image_name)

        if os.path.exists(image_path):
            negative_count += 1
        else:
            missing_images.append(image_name)

print("========== NEGATIVE IMAGE VERIFICATION ==========")
print("Verified negative images:", negative_count)
print("Missing corresponding images:", len(missing_images))

if missing_images:
    print("Examples:", missing_images[:10])
