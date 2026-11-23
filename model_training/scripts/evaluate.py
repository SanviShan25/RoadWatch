"""Evaluate a trained RoadWatch YOLO model on the test split and save the metrics as JSON.

Example:
    python model_training/scripts/evaluate.py \
        --weights runs/v8n_baseline/weights/best.pt \
        --data /content/data.yaml

By default the JSON goes to model_training/reports/<run name>_<split>_metrics.json.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on a dataset split.")
    parser.add_argument("--weights", required=True, help="Path to the trained weights, e.g. best.pt")
    parser.add_argument("--data", required=True, help="Path to the dataset YAML (with a test: entry)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", default=None, help="'cpu', '0' for the first GPU, or leave empty for auto")
    parser.add_argument("--out", default=None, help="Where to save the JSON (default: model_training/reports/...)")
    return parser.parse_args()


def default_output_path(weights: Path, split: str) -> Path:
    # weights usually live in <run>/weights/best.pt, so use the run folder name in the file name.
    run_name = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    return reports_dir / f"{run_name}_{split}_metrics.json"


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,   # we only need the numbers here
        verbose=False,
    )
    box = metrics.box

    # Overall metrics (averaged over classes).
    overall = {
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }

    # Per-class metrics. Ultralytics only reports classes that occur in the split; ap_class_index
    # gives their class ids in the same order as the per-class arrays. Missing classes get None.
    per_class = {}
    reported = {int(class_id): i for i, class_id in enumerate(box.ap_class_index)}
    for class_id, name in metrics.names.items():
        i = reported.get(int(class_id))
        per_class[name] = {
            "class_id": int(class_id),
            "AP50": round(float(box.ap50[i]), 4) if i is not None else None,
            "AP50_95": round(float(box.ap[i]), 4) if i is not None else None,
            "precision": round(float(box.p[i]), 4) if i is not None else None,
            "recall": round(float(box.r[i]), 4) if i is not None else None,
        }

    # Print a readable summary.
    print(f"\nResults on the '{args.split}' split for {weights}")
    for key, value in overall.items():
        print(f"  {key:10s}: {value:.4f}")
    print(f"\n  {'class':6s} {'AP50':>7s} {'AP50-95':>8s} {'P':>7s} {'R':>7s}")
    for name, m in per_class.items():
        cells = [f"{m[k]:.4f}" if m[k] is not None else "   n/a" for k in ("AP50", "AP50_95", "precision", "recall")]
        print(f"  {name:6s} {cells[0]:>7s} {cells[1]:>8s} {cells[2]:>7s} {cells[3]:>7s}")

    # Save everything to JSON.
    out = Path(args.out) if args.out else default_output_path(weights, args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "weights": str(weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": overall,
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in metrics.speed.items()},
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()
