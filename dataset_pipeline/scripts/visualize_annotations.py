
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

image_dir = r"C:\Users\KIIT\Desktop\RDD2022_India\India\train\images"

annotation_dir = r"C:\Users\KIIT\Desktop\RDD2022_India\India\train\annotations\xmls"

xml_file = "India_000078.xml"

# Read XML annotation
xml_path = os.path.join(annotation_dir, xml_file)
root = ET.parse(xml_path).getroot()

# Find corresponding image
image_name = root.findtext("filename")

if not image_name:
    image_name = os.path.splitext(xml_file)[0] + ".jpg"

image_path = os.path.join(image_dir, image_name)

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Open image
image = Image.open(image_path)

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

print("Image:", image_name)
print("Image dimensions:", image.size)

# Draw every annotated bounding box
for obj in root.findall("object"):

    class_name = obj.findtext("name")
    box = obj.find("bndbox")

    xmin = int(box.findtext("xmin"))
    ymin = int(box.findtext("ymin"))
    xmax = int(box.findtext("xmax"))
    ymax = int(box.findtext("ymax"))

    width = xmax - xmin
    height = ymax - ymin

    rectangle = patches.Rectangle(
        (xmin, ymin),
        width,
        height,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rectangle)

    ax.text(
        xmin,
        ymin - 5,
        class_name,
        color="white",
        fontsize=10,
        bbox=dict(facecolor="red", alpha=0.8)
    )

    print(f"{class_name}: ({xmin}, {ymin}, {xmax}, {ymax})")

ax.set_title("Road Damage - Ground Truth Annotations")
ax.axis("off")

plt.tight_layout()
plt.show()
