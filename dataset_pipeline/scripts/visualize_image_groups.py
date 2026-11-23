
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict
import csv

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "RoadWatch_YOLO_Final" / "images"
CSV_FILE = ROOT / "near_duplicate_groups.csv"
OUTPUT = ROOT / "image_group_contact_sheet.jpg"

groups = defaultdict(list)

with open(CSV_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        groups[int(row["Group ID"])].append(
            (row["Split"], row["Image"])
        )

thumb_width = 240
thumb_height = 180
images_per_row = 3
group_gap = 45

group_heights = []

for members in groups.values():
    rows = (len(members) + images_per_row - 1) // images_per_row
    group_heights.append(35 + rows * (thumb_height + 25) + group_gap)

sheet_height = sum(group_heights)

sheet = Image.new(
    "RGB",
    (images_per_row * thumb_width, sheet_height),
    "white"
)

draw = ImageDraw.Draw(sheet)
current_y = 0

for group_id, members in sorted(groups.items()):
    draw.text(
        (5, current_y + 5),
        f"GROUP {group_id} - {len(members)} images",
        fill="red"
    )

    current_y += 35

    for index, (split, filename) in enumerate(sorted(members)):
        row = index // images_per_row
        column = index % images_per_row

        x = column * thumb_width
        y = current_y + row * (thumb_height + 25)

        image_path = IMAGE_DIR / split / filename

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_width - 10, thumb_height - 10))
            sheet.paste(image, (x + 5, y + 20))

        draw.text(
            (x + 5, y + 3),
            f"{split}: {filename}",
            fill="black"
        )

    rows = (len(members) + images_per_row - 1) // images_per_row
    current_y += rows * (thumb_height + 25) + group_gap

sheet.save(OUTPUT)

print("Groups displayed:", len(groups))
print("Images displayed:", sum(len(v) for v in groups.values()))
print("Contact sheet saved:", OUTPUT)
