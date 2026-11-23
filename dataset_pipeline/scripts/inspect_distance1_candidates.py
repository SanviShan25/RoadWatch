
from pathlib import Path
from PIL import Image, ImageDraw
import csv

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "RoadWatch_YOLO_Final" / "images"
CSV_FILE = ROOT / "near_duplicate_candidates.csv"
OUTPUT = ROOT / "distance1_candidates.jpg"

pairs = []

with open(CSV_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if int(row["Hash Distance"]) == 1:
            pairs.append(row)

width = 600
row_height = 190

sheet = Image.new("RGB", (width, len(pairs) * row_height), "white")
draw = ImageDraw.Draw(sheet)

for index, pair in enumerate(pairs):
    y = index * row_height

    for column, split_key, image_key in [
        (0, "Split 1", "Image 1"),
        (1, "Split 2", "Image 2")
    ]:
        path = IMAGE_DIR / pair[split_key] / pair[image_key]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((290, 155))
            x = column * 300
            sheet.paste(image, (x, y + 25))

        draw.text(
            (column * 300 + 5, y + 5),
            f'{pair[split_key]}: {pair[image_key]}',
            fill="black"
        )

    draw.text(
        (5, y + 180),
        f"Pair {index + 1} | Distance: 1",
        fill="red"
    )

sheet.save(OUTPUT)

print("Pairs displayed:", len(pairs))
print("Saved contact sheet:", OUTPUT)
