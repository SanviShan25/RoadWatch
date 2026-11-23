
from pathlib import Path
import hashlib
from collections import defaultdict

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Image split directories
BASE_DIR = PROJECT_ROOT / "data" / "RoadWatch_YOLO_Final" / "images"

SPLITS = ["train", "val", "test"]

# Store image paths grouped by their SHA-256 hash
image_hashes = defaultdict(list)

print("Starting exact duplicate image check...\n")

for split in SPLITS:
    split_dir = BASE_DIR / split

    for image_path in split_dir.iterdir():
        if not image_path.is_file():
            continue

        # Calculate SHA-256 hash
        sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

        image_hashes[sha256].append((split, image_path.name))

# Identify duplicate groups
duplicates = {
    sha: files
    for sha, files in image_hashes.items()
    if len(files) > 1
}

print("========== DUPLICATE CHECK RESULTS ==========")
print("Total unique image hashes:", len(image_hashes))
print("Duplicate groups found:", len(duplicates))

if duplicates:
    print("\nDuplicate images:")

    for sha, files in duplicates.items():
        print("\nHash:", sha)

        for split, filename in files:
            print(f"  {split}: {filename}")
else:
    print("\nNo exact duplicate images found!")

print("\n========== CHECK COMPLETE ==========")
