
from pathlib import Path
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "data" / "RoadWatch_YOLO_Final" / "images"
INPUT_CSV = ROOT / "near_duplicate_candidates.csv"
OUTPUT_CSV = ROOT / "verified_similarity_results.csv"

# Only inspect the strongest candidates
MAX_HASH_DISTANCE = 1

orb = cv2.ORB_create(nfeatures=1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def compare_images(path1, path2):
    image1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        return None

    # SSIM comparison
    resized1 = cv2.resize(image1, (256, 256))
    resized2 = cv2.resize(image2, (256, 256))

    ssim_score = structural_similarity(resized1, resized2)

    # ORB feature detection
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        return ssim_score, 0, 0, 0

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    # Lowe's ratio test
    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    inliers = 0
    inlier_ratio = 0.0

    if len(good_matches) >= 4:
        points1 = np.float32([
            keypoints1[m.queryIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)

        points2 = np.float32([
            keypoints2[m.trainIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(
            points1,
            points2,
            cv2.RANSAC,
            5.0
        )

        if mask is not None:
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(good_matches)

    return (
        ssim_score,
        len(good_matches),
        inliers,
        inlier_ratio
    )


results = []

with open(INPUT_CSV, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        distance = int(row["Hash Distance"])

        if distance > MAX_HASH_DISTANCE:
            continue

        path1 = IMAGE_DIR / row["Split 1"] / row["Image 1"]
        path2 = IMAGE_DIR / row["Split 2"] / row["Image 2"]

        metrics = compare_images(path1, path2)

        if metrics is None:
            print("Could not read:", path1, path2)
            continue

        ssim_score, good_matches, inliers, ratio = metrics

        result = {
            "Split 1": row["Split 1"],
            "Image 1": row["Image 1"],
            "Split 2": row["Split 2"],
            "Image 2": row["Image 2"],
            "Hash Distance": distance,
            "SSIM": round(ssim_score, 4),
            "ORB Good Matches": good_matches,
            "RANSAC Inliers": inliers,
            "Inlier Ratio": round(ratio, 4)
        }

        results.append(result)

        print(
            f'{row["Image 1"]} vs {row["Image 2"]}: '
            f'SSIM={ssim_score:.3f}, '
            f'ORB={good_matches}, '
            f'Inliers={inliers}'
        )


fields = [
    "Split 1", "Image 1",
    "Split 2", "Image 2",
    "Hash Distance", "SSIM",
    "ORB Good Matches", "RANSAC Inliers",
    "Inlier Ratio"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("\n========== VERIFICATION COMPLETE ==========")
print("Pairs analysed:", len(results))
print("Report saved:", OUTPUT_CSV)
print("Original dataset was not modified.")
