
from pathlib import Path
from PIL import Image, ImageDraw
import csv

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "RoadWatch_YOLO_Final" / "images"
CSV_FILE = ROOT / "verified_similarity_results.csv"
OUTPUT = ROOT / "strong_match_comparison.jpg"

pairs = []

with open(CSV_FILE, newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if int(row["RANSAC Inliers"]) >= 20:
            pairs.append(row)

pairs.sort(
    key=lambda row: int(row["RANSAC Inliers"]),
    reverse=True
)

width = 1000
row_height = 310

sheet = Image.new("RGB", (width, len(pairs) * row_height), "white")
draw = ImageDraw.Draw(sheet)

for index, pair in enumerate(pairs):
    y = index * row_height

    draw.text(
        (10, y + 5),
        f'PAIR {index + 1} | RANSAC inliers: {pair["RANSAC Inliers"]}',
        fill="red"
    )

    for column, split_key, image_key in [
        (0, "Split 1", "Image 1"),
        (1, "Split 2", "Image 2")
    ]:
        path = IMAGE_DIR / pair[split_key] / pair[image_key]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((480, 260))
            sheet.paste(image, (column * 500 + 5, y + 35))

        draw.text(
            (column * 500 + 5, y + 15),
            f'{pair[split_key]}: {pair[image_key]}',
            fill="black"
        )

sheet.save(OUTPUT)

print("Strong pairs displayed:", len(pairs))
print("Saved:", OUTPUT)
