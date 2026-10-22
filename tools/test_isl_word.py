"""Offline Evaluation & Test Runner for Pretrained ISL Word Model (Abs6187/isl-models).

Runs identical preprocessing and inference pipeline used by the SIGNBRIDGE application:
Video -> 30-frame temporal sampling -> ImageNet normalization -> 154-dim MediaPipe Landmarks ->
Pretrained ISL Hierarchical Transformer -> Prediction + Confidence + Top-5 Predictions.

Usage:
    python tools/test_isl_word.py --video samples/isl_test.mp4
    python tools/test_isl_word.py --video samples/isl_test.mp4 --ground-truth "she"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Relaunch in venv if torch not present
try:
    import torch
except ImportError:
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        import subprocess
        res = subprocess.run([str(venv_py)] + sys.argv, check=False)
        sys.exit(res.returncode)
    raise

import cv2
import numpy as np

from src.recognition.isl_word_recognizer import ISLWordRecognizer
from src.landmark_extractor import LandmarkExtractor


def generate_sample_isl_video(output_path: Path, num_frames: int = 30) -> Path:
    """Generate a synthetic test ISL video sample if none exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 30, 25)

        t = i / float(num_frames)
        cx = int(width * (0.4 + 0.2 * np.sin(2 * np.pi * t)))
        cy = int(height * (0.6 - 0.15 * np.cos(np.pi * t)))

        cv2.circle(frame, (width // 2, 60), 25, (180, 180, 200), -1)
        cv2.ellipse(frame, (width // 2, 160), (45, 60), 0, 0, 360, (120, 100, 90), -1)
        cv2.line(frame, (width // 2 - 30, 95), (cx, cy), (160, 150, 140), 6)
        cv2.circle(frame, (cx, cy), 14, (200, 180, 160), -1)

        cv2.putText(
            frame,
            f"ISL Sample - Frame {i + 1}/{num_frames}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    return output_path


def run_isl_video_evaluation(
    video_path: Optional[str] = None,
    ground_truth: Optional[str] = None,
    confidence_threshold: float = 0.20,
    device: str = "auto",
) -> int:
    target_video = Path(video_path) if video_path else PROJECT_ROOT / "samples" / "isl_test.mp4"
    if not target_video.exists():
        if (PROJECT_ROOT / "sample.mp4").exists():
            target_video = PROJECT_ROOT / "sample.mp4"
        else:
            print(f"[Notice] Video '{target_video.name}' not found. Generating synthetic sample...")
            target_video = generate_sample_isl_video(target_video, num_frames=30)

    print("\n" + "=" * 65)
    print("      ISL WORD RECOGNITION — SAMPLE VIDEO EVALUATION")
    print("=" * 65)
    print(f"Input Video         : {target_video.name} ({target_video})")

    # 1. Initialize Recognizer
    t0_load = time.perf_counter()
    recognizer = ISLWordRecognizer(
        model_id="Abs6187/isl-models",
        model_dir=str(PROJECT_ROOT / "models" / "isl"),
        confidence_threshold=confidence_threshold,
        device=device,
        use_hierarchical=True,
        enabled=True,
    )
    t_load = (time.perf_counter() - t0_load) * 1000.0

    if not recognizer.is_loaded():
        print(f"[ERROR] Model loading failed: {recognizer.load_error}")
        return 1

    print(f"Model Status        : Loaded in {t_load:.1f} ms ({len(recognizer.get_vocab())} vocabulary classes)")

    # 2. Read Video & Preprocess Frames
    cap = cv2.VideoCapture(str(target_video))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"[ERROR] Could not read any video frames from: {target_video}")
        return 1

    print(f"Captured Video      : {len(frames)} raw frames read")

    # 3. Extract 154-dimensional MediaPipe landmarks
    pose_landmarks = []
    landmarker_warning = None
    try:
        extractor = LandmarkExtractor(model_path="models/hand_landmarker.task")
        for f in frames:
            feats = extractor.extract_isl_features(f)
            pose_landmarks.append(feats)
        extractor.close()
        print(f"Landmark Features   : Extracted 154-dim vectors for {len(pose_landmarks)} frames")
    except Exception as exc:
        landmarker_warning = f"Landmark extractor warning: {exc}. Falling back to visual stream."
        pose_landmarks = None
        print(f"Landmark Features   : [FALLBACK] {landmarker_warning}")

    # 4. Predict
    t0_pred = time.perf_counter()
    result = recognizer.predict(frames, pose_landmarks=pose_landmarks)
    t_pred = (time.perf_counter() - t0_pred) * 1000.0

    raw_label = result.metadata.get("raw_label", result.label)
    confidence = float(result.confidence)
    top_k = result.metadata.get("top_k", [])
    model_type = result.metadata.get("model_type", "N/A")
    diag = result.metadata.get("diagnostics", {})

    print(f"\n" + "-" * 65)
    print("                      PREDICTION RESULTS")
    print("-" * 65)
    print(f"Predicted Word      : {raw_label} (Display Label: {result.label})")
    print(f"Model Confidence    : {confidence:.2%} ({confidence:.4f})")
    print(f"Threshold Cutoff    : {confidence_threshold:.2%}")
    if diag.get("gating_group"):
        print(f"Hierarchical Group  : {diag.get('gating_group')} (Group Prob: {diag.get('gating_confidence', 0):.2%})")
    print(f"Inference Time      : {t_pred:.1f} ms")
    print(f"Model Architecture  : {model_type}")

    print(f"\nTop-5 Predictions:")
    for idx, item in enumerate(top_k[:5], start=1):
        c = item.get("class", item.get("label", "UNKNOWN"))
        conf = float(item.get("confidence", 0.0))
        print(f"  {idx}. {c:<18} ({conf:.2%})")

    # 5. Ground-Truth Evaluation
    print(f"\n" + "-" * 65)
    print("                   ACCURACY / GROUND TRUTH")
    print("-" * 65)
    if ground_truth:
        gt_clean = ground_truth.strip().upper()
        pred_clean = raw_label.strip().upper()
        is_match = (gt_clean == pred_clean)
        print(f"Ground Truth Label  : {ground_truth}")
        print(f"Predicted Word      : {raw_label}")
        print(f"Evaluation Match    : {'CORRECT (MATCH)' if is_match else 'INCORRECT (MISMATCH)'}")
        print(f"Sample Accuracy     : {100.0 if is_match else 0.0:.1f}% (1 / 1 evaluated sample)")
    else:
        print("Ground-Truth Label  : None provided.")
        print("Note                : Exact accuracy percentage requires labelled ground-truth test data.")
        print("                      Demonstrations without ground-truth labels cannot claim accuracy.")

    if landmarker_warning:
        print(f"\n[Warning] {landmarker_warning}")

    print("=" * 65 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL word model on a sample video.")
    parser.add_argument("--video", type=str, default=None, help="Path to ISL test video (.mp4)")
    parser.add_argument("--ground-truth", type=str, default=None, help="Expected ground-truth word label")
    parser.add_argument("--threshold", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device")
    args = parser.parse_args()

    code = run_isl_video_evaluation(
        video_path=args.video,
        ground_truth=args.ground_truth,
        confidence_threshold=args.threshold,
        device=args.device,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
