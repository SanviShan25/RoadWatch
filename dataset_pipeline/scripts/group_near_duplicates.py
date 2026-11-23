
from pathlib import Path
from collections import defaultdict
import csv

ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = ROOT / "near_duplicate_candidates.csv"
OUTPUT_FILE = ROOT / "near_duplicate_groups.csv"

THRESHOLD = 1

parent = {}
edges = []


def find(node):
    if parent[node] != node:
        parent[node] = find(parent[node])
    return parent[node]


def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# Read candidate pairs
with open(CSV_FILE, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        distance = int(row["Hash Distance"])

        if distance > THRESHOLD:
            continue

        image_a = (row["Split 1"], row["Image 1"])
        image_b = (row["Split 2"], row["Image 2"])

        parent.setdefault(image_a, image_a)
        parent.setdefault(image_b, image_b)

        union(image_a, image_b)
        edges.append((image_a, image_b, distance))


# Build connected groups
groups = defaultdict(list)

for image in parent:
    groups[find(image)].append(image)

valid_groups = [
    members for members in groups.values()
    if len(members) > 1
]

valid_groups.sort(key=lambda group: (-len(group), sorted(group)))

# Save the groups
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Group ID", "Split", "Image"])

    for group_id, members in enumerate(valid_groups, start=1):
        for split, filename in sorted(members):
            writer.writerow([group_id, split, filename])


print("========== IMAGE GROUPING RESULTS ==========")
print("Hash distance threshold:", THRESHOLD)
print("Candidate pairs used:", len(edges))
print("Potential image groups:", len(valid_groups))

for group_id, members in enumerate(valid_groups, start=1):
    print(f"\nGroup {group_id} ({len(members)} images):")

    for split, filename in sorted(members):
        print(f"  {split}: {filename}")

print("\nReport saved at:", OUTPUT_FILE)
print("No dataset images were modified.")
