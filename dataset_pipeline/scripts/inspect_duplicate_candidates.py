
from pathlib import Path
from PIL import Image, ImageDraw
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" / "images"
CSV_FILE = PROJECT_ROOT / "near_duplicate_candidates.csv"

output = PROJECT_ROOT / "exact_hash_candidates.jpg"

pairs = []

with open(CSV_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if int(row["Hash Distance"]) == 0:
            pairs.append(row)

thumb_width = 360
thumb_height = 240
sheet_height = len(pairs) * 310

sheet = Image.new("RGB", (thumb_width * 2, sheet_height), "white")
draw = ImageDraw.Draw(sheet)

for index, pair in enumerate(pairs):
    y = index * 310

    for column, split_key, image_key in [
        (0, "Split 1", "Image 1"),
        (1, "Split 2", "Image 2")
    ]:
        image_path = IMAGE_DIR / pair[split_key] / pair[image_key]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_width - 10, thumb_height - 10))

            x = column * thumb_width
            sheet.paste(image, (x, y + 35))

        draw.text(
            (column * thumb_width + 5, y + 5),
            f'{pair[split_key]}: {pair[image_key]}',
            fill="black"
        )

    draw.text(
        (5, y + 280),
        f"Pair {index + 1} | Hash distance: 0",
        fill="red"
    )

sheet.save(output)

print("Pairs inspected:", len(pairs))
print("Contact sheet saved at:", output)
