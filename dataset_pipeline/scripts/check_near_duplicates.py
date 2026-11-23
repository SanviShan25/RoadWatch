
from pathlib import Path
from PIL import Image
from itertools import combinations
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" / "images"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

THRESHOLD = 5


def calculate_hash(image_path):
    with Image.open(image_path) as image:
        image = image.convert("L").resize((9, 8))

        pixels = list(image.getdata())
        hash_value = 0

        for row in range(8):
            for col in range(8):
                left = pixels[row * 9 + col]
                right = pixels[row * 9 + col + 1]

                hash_value = (hash_value << 1) | (left > right)

        return hash_value


def hamming_distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()


print("Calculating perceptual hashes...\n")

images = {}

for split in SPLITS:
    split_dir = BASE_DIR / split

    for path in split_dir.iterdir():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            images[path] = {
                "split": split,
                "hash": calculate_hash(path)
            }
        except Exception as error:
            print(f"Could not process {path.name}: {error}")


print("Images processed:", len(images))
print("\nComparing images across different splits...")

candidates = []

for image1, image2 in combinations(images.keys(), 2):
    info1 = images[image1]
    info2 = images[image2]

    if info1["split"] == info2["split"]:
        continue

    distance = hamming_distance(info1["hash"], info2["hash"])

    if distance <= THRESHOLD:
        candidates.append([
            info1["split"],
            image1.name,
            info2["split"],
            image2.name,
            distance
        ])


report_path = PROJECT_ROOT / "near_duplicate_candidates.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Split 1", "Image 1",
        "Split 2", "Image 2",
        "Hash Distance"
    ])
    writer.writerows(candidates)


print("\n========== NEAR-DUPLICATE RESULTS ==========")
print("Images processed:", len(images))
print("Distance threshold:", THRESHOLD)
print("Potential near-duplicate pairs:", len(candidates))

if candidates:
    print("\nFirst 30 candidate pairs:")

    for item in candidates[:30]:
        print(item)
else:
    print("\nNo potential near-duplicates found!")

print("\nFull report saved to:", report_path)
print("\n========== CHECK COMPLETE ==========")
