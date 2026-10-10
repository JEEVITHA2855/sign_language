"""Evaluation script for Indian Sign Language (ISL) models.

Measures:
- Accuracy, Precision, Recall, F1-Score (macro and weighted)
- Confusion Matrix
- Real-Time Performance: Inference latency (mean, median, p95), FPS throughput
- Error Analysis on Uncertain Predictions

Usage:
    python evaluation/evaluate_isl.py --dataset path/to/dataset --mode ALPHABET
    python evaluation/evaluate_isl.py --benchmark-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.isl_recognizer import ISLRecognizer
from src.recognition.recognition_result import RecognitionResult


def benchmark_isl_inference(recognizer: ISLRecognizer, num_iterations: int = 50) -> Dict[str, float]:
    """Measure inference latency and FPS without needing an external dataset."""
    print(f"\n[ISL Benchmark] Running {num_iterations} synthetic frame passes on {recognizer.device}...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    latencies = []
    # Warmup
    for _ in range(5):
        _ = recognizer.predict([dummy_frame])

    for i in range(num_iterations):
        t0 = time.perf_counter()
        _ = recognizer.predict([dummy_frame])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    metrics = {
        "mean_latency_ms": round(mean_ms, 2),
        "median_latency_ms": round(median_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "throughput_fps": round(fps, 1),
        "iterations": num_iterations,
    }

    print("\n--- Benchmark Results ---")
    print(f"Mean Latency   : {metrics['mean_latency_ms']} ms")
    print(f"Median Latency : {metrics['median_latency_ms']} ms")
    print(f"95th Percentile: {metrics['p95_latency_ms']} ms")
    print(f"Estimated FPS  : {metrics['throughput_fps']} FPS")
    return metrics


def evaluate_isl_dataset(
    recognizer: ISLRecognizer,
    dataset_dir: str,
    output_report: str = "evaluation/isl_eval_report.json",
) -> Dict[str, Any]:
    """Evaluate ISL model on image directory structured by class folders."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return {}

    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    print(f"\n[ISL Evaluation] Found {len(class_dirs)} class directories in {dataset_dir}")

    import cv2
    for c_dir in sorted(class_dirs):
        true_label = c_dir.name.upper()
        images = [f for f in c_dir.iterdir() if f.suffix.lower() in image_extensions]
        print(f"Evaluating class '{true_label}': {len(images)} samples...")

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            res = recognizer.predict([img])
            y_true.append(true_label)
            y_pred.append(res.label.upper())
            confidences.append(res.confidence)

    if not y_true:
        print("No valid images processed.")
        return {}

    # Calculate metrics with scikit-learn
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    results = {
        "dataset": str(dataset_dir),
        "total_samples": len(y_true),
        "accuracy": round(acc, 4),
        "classification_report": report,
        "labels": labels,
        "confusion_matrix": cm,
        "mean_confidence": round(float(np.mean(confidences)), 4),
    }

    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Evaluation Summary ---")
    print(f"Total Samples   : {len(y_true)}")
    print(f"Accuracy        : {acc * 100:.2f}%")
    print(f"Mean Confidence : {results['mean_confidence'] * 100:.2f}%")
    print(f"Report saved to : {output_report}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL Recognition Model")
    parser.add_argument("--dataset", type=str, default=None, help="Directory with subfolders per class")
    parser.add_argument("--mode", choices=["ALPHABET", "WORD"], default="ALPHABET")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--output", type=str, default="evaluation/isl_report.json")
    args = parser.parse_args()

    recognizer = ISLRecognizer(
        mode=args.mode,
        confidence_threshold=args.threshold,
    )

    if args.benchmark_only or args.dataset is None:
        benchmark_isl_inference(recognizer)
    else:
        evaluate_isl_dataset(recognizer, args.dataset, args.output)


if __name__ == "__main__":
    main()
