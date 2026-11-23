"""Measure CPU inference speed and file size of a RoadWatch YOLO model.

Example:
    python model_training/scripts/benchmark_cpu.py \
        --weights runs/v8n_baseline/weights/best.pt \
        --images data/RoadWatch_YOLO_Grouped/images/test \
        --num-images 100 --export-onnx

Timing is end to end per image (pre-processing + inference + post-processing), after a warm-up.
The ONNX benchmark needs the `onnx` and `onnxruntime` packages (Ultralytics may try to install them).
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO inference on the CPU.")
    parser.add_argument("--weights", required=True, help="Path to the weights, e.g. best.pt")
    parser.add_argument("--images", required=True, help="A folder of images (or a single image)")
    parser.add_argument("--num-images", type=int, default=100, help="Images to time (default: 100)")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warm-up runs (default: 10)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--threads", type=int, default=None, help="Limit PyTorch CPU threads (default: all)")
    parser.add_argument("--export-onnx", action="store_true", help="Also export to ONNX and benchmark it")
    return parser.parse_args()


def list_images(path: Path, count: int):
    """Return `count` image paths; repeats the list if the folder has fewer images."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"No images found in {path}")
    return [files[i % len(files)] for i in range(count)]


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def benchmark(model: YOLO, images, warmup: int, imgsz: int):
    """Warm up, then time each image individually. Returns per-image times in milliseconds."""
    for image in images[:warmup]:
        model.predict(str(image), imgsz=imgsz, device="cpu", verbose=False)

    times_ms = []
    for image in images:
        start = time.perf_counter()
        model.predict(str(image), imgsz=imgsz, device="cpu", verbose=False)
        times_ms.append((time.perf_counter() - start) * 1000)
    return times_ms


def report(label: str, weights_path: Path, times_ms):
    mean = statistics.mean(times_ms)
    p95 = sorted(times_ms)[max(0, int(len(times_ms) * 0.95) - 1)]
    print(f"\n{label}")
    print(f"  file           : {weights_path} ({size_mb(weights_path):.2f} MB)")
    print(f"  images timed   : {len(times_ms)}")
    print(f"  ms per image   : mean {mean:.1f} | median {statistics.median(times_ms):.1f} | p95 {p95:.1f}")
    print(f"  throughput     : {1000 / mean:.1f} images/s")


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"CPU threads used by PyTorch: {torch.get_num_threads()}")

    images = list_images(Path(args.images), args.num_images)

    # 1) PyTorch weights on the CPU.
    model = YOLO(str(weights))
    report("PyTorch (.pt) on CPU", weights, benchmark(model, images, args.warmup, args.imgsz))

    # 2) Optional ONNX export, benchmarked the same way.
    if args.export_onnx:
        onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz))
        onnx_model = YOLO(str(onnx_path), task="detect")
        report("ONNX on CPU", onnx_path, benchmark(onnx_model, images, args.warmup, args.imgsz))


if __name__ == "__main__":
    main()
