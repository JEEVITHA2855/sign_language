"""Evaluation script for Speech-to-Text models (Whisper ASR).

Measures:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Real-Time Factor (RTF = audio_duration / processing_time)
- Inference Latency (mean, median, p95)

Usage:
    python evaluation/evaluate_speech.py --benchmark-only
    python evaluation/evaluate_speech.py --manifest path/to/eval_manifest.json
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

from src.speech.speech_recognizer import SpeechRecognizer


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) using Levenshtein distance on words."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    return float(d[len(ref_words), len(hyp_words)]) / float(len(ref_words))


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER)."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())

    if not ref_chars:
        return 1.0 if hyp_chars else 0.0

    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=np.int32)
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j - 1] + 1, d[i, j - 1] + 1, d[i - 1, j] + 1)

    return float(d[len(ref_chars), len(hyp_chars)]) / float(len(ref_chars))


def benchmark_whisper_speed(recognizer: SpeechRecognizer, num_iterations: int = 15) -> Dict[str, Any]:
    """Measure transcription latency on 3-second audio buffers."""
    sample_rate = 16000
    duration_s = 3.0
    num_samples = int(sample_rate * duration_s)
    # Generate random synthetic speech audio
    dummy_audio = np.random.uniform(-0.1, 0.1, num_samples).astype(np.float32)

    print(f"\n[Speech ASR Benchmark] Model: {recognizer.model_name} on {recognizer.device}")
    print(f"Testing {num_iterations} runs with {duration_s}s audio clips...")

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = recognizer.predict(dummy_audio, sample_rate=sample_rate)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    rtf = (mean_ms / 1000.0) / duration_s

    metrics = {
        "model_name": recognizer.model_name,
        "device": recognizer.device,
        "clip_duration_s": duration_s,
        "mean_latency_ms": round(mean_ms, 2),
        "median_latency_ms": round(median_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "real_time_factor_rtf": round(rtf, 3),
        "is_faster_than_realtime": rtf < 1.0,
    }

    print("\n--- Speech Benchmark Results ---")
    print(f"Mean Latency        : {metrics['mean_latency_ms']} ms")
    print(f"Median Latency      : {metrics['median_latency_ms']} ms")
    print(f"95th Percentile     : {metrics['p95_latency_ms']} ms")
    print(f"Real-Time Factor    : {metrics['real_time_factor_rtf']}x ( <1.0 = faster than real-time )")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speech-to-Text Whisper Model")
    parser.add_argument("--model", type=str, default="openai/whisper-base")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    recognizer = SpeechRecognizer(model_name=args.model)
    metrics = benchmark_whisper_speed(recognizer, num_iterations=args.iterations)

    out_file = Path("evaluation/speech_eval_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Report written to {out_file}")


if __name__ == "__main__":
    main()
