
import os

label_dir = "RoadWatch_YOLO/labels/train"

total_files = 0
valid_files = 0
invalid_files = 0
total_boxes = 0
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

issues = []

for filename in os.listdir(label_dir):

    if not filename.endswith(".txt"):
        continue

    total_files += 1
    file_path = os.path.join(label_dir, filename)

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        file_valid = True

        for line_number, line in enumerate(lines, start=1):

            values = line.strip().split()

            if len(values) != 5:
                issues.append(f"{filename}: Invalid number of values")
                file_valid = False
                continue

            try:
                class_id = int(values[0])
                coordinates = list(map(float, values[1:]))

                if class_id not in class_counts:
                    issues.append(f"{filename}: Invalid class ID {class_id}")
                    file_valid = False
                    continue

                if not all(0 <= value <= 1 for value in coordinates):
                    issues.append(f"{filename}: Coordinates outside 0-1")
                    file_valid = False
                    continue

                x, y, width, height = coordinates

                if width <= 0 or height <= 0:
                    issues.append(f"{filename}: Invalid box dimensions")
                    file_valid = False
                    continue

                if x - width / 2 < -0.000001 or x + width / 2 > 1.000001:
                    issues.append(f"{filename}: Box exceeds horizontal boundary")
                    file_valid = False
                    continue

                if y - height / 2 < -0.000001 or y + height / 2 > 1.000001:
                    issues.append(f"{filename}: Box exceeds vertical boundary")
                    file_valid = False
                    continue

                class_counts[class_id] += 1
                total_boxes += 1

            except ValueError:
                issues.append(f"{filename}: Non-numeric values")
                file_valid = False

        if file_valid:
            valid_files += 1
        else:
            invalid_files += 1

    except Exception as e:
        invalid_files += 1
        issues.append(f"{filename}: {e}")


print("\n========== YOLO LABEL VALIDATION ==========")
print("Total label files:", total_files)
print("Valid label files:", valid_files)
print("Invalid label files:", invalid_files)
print("Total bounding boxes:", total_boxes)

print("\nClass distribution:")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count}")

print("\nIssues:")
if issues:
    for issue in issues[:20]:
        print(issue)
else:
    print("No issues found!")
