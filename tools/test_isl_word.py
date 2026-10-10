"""Offline Standalone Test for Pretrained Hugging Face ISL Model (Abs6187/isl-models).

Tests the pretrained model independently without requiring a physical webcam:
Video File -> Preprocessing -> Hugging Face ISL Model -> Prediction + Confidence + Top 5.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        import subprocess
        res = subprocess.run([str(venv_py)] + sys.argv, check=False)
        sys.exit(res.returncode)
    raise

from src.models.isl_model import SignLanguageInference, load_isl_model
from src.landmark_extractor import LandmarkExtractor


def generate_sample_isl_video(output_path: Path, num_frames: int = 30) -> Path:
    """Generate a synthetic test ISL video sample if none exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

    # Animate a hand-like gesture motion across frames
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 30, 25)

        # Draw moving hand/pose figure
        t = i / float(num_frames)
        cx = int(width * (0.4 + 0.2 * np.sin(2 * np.pi * t)))
        cy = int(height * (0.6 - 0.15 * np.cos(np.pi * t)))

        # Head / Torso
        cv2.circle(frame, (width // 2, 60), 25, (180, 180, 200), -1)
        cv2.ellipse(frame, (width // 2, 160), (45, 60), 0, 0, 360, (120, 100, 90), -1)

        # Arms and Hand
        cv2.line(frame, (width // 2 - 30, 95), (cx, cy), (160, 150, 140), 6)
        cv2.circle(frame, (cx, cy), 14, (200, 180, 160), -1)

        # Annotation
        cv2.putText(
            frame,
            f"ISL Test Sample - Frame {i + 1}/{num_frames}",
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


def run_offline_isl_test(video_path: Optional[str] = None) -> None:
    """Run independent offline inference on video sample."""
    sample_file = Path(video_path) if video_path else PROJECT_ROOT / "sample.mp4"
    if not sample_file.exists():
        print(f"[Notice] Video file '{sample_file.name}' not found. Generating test sample...")
        sample_file = generate_sample_isl_video(sample_file, num_frames=30)

    # 1. Load Pretrained ISL Model
    model_dir = PROJECT_ROOT / "models" / "isl"
    inference = load_isl_model(
        model_id="Abs6187/isl-models",
        checkpoint_name="best_model_new.pth",
        metadata_name="metadata.json",
        cache_dir=str(model_dir),
        device="auto",
        use_hierarchical=True,
    )

    if not inference.is_ready:
        print("ERROR: ISL model failed to initialize!")
        sys.exit(1)

    # 2. Extract frames from video
    cap = cv2.VideoCapture(str(sample_file))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"ERROR: Could not read frames from video: {sample_file}")
        sys.exit(1)

    # 3. Extract MediaPipe landmarks if landmarker task available
    pose_landmarks = []
    try:
        extractor = LandmarkExtractor(model_path="models/hand_landmarker.task")
        for f in frames:
            feats = extractor.extract_isl_features(f)
            pose_landmarks.append(feats)
        extractor.close()
    except Exception:
        pose_landmarks = None

    # 4. Model Preprocessing & Inference
    result = inference.predict(frames, pose_input=pose_landmarks)
    top_pred = result.get("top_prediction", {})
    predicted_class = top_pred.get("class", result.get("class", "UNKNOWN"))
    confidence = float(top_pred.get("confidence", result.get("confidence", 0.0)))
    top_k = result.get("top_k", [])

    # Output matching Requirement 19
    print()
    print("Input:")
    print(sample_file.name)
    print()
    print("Prediction:")
    print(predicted_class)
    print()
    print("Confidence:")
    print(f"{confidence:.2f}")
    print()
    print("Top 5:")
    for idx, item in enumerate(top_k[:5], start=1):
        c = item.get("class", item.get("label", "UNKNOWN"))
        conf = float(item.get("confidence", 0.0))
        print(f"{idx}. {c:<16} ({conf:.2f})")
    print()
    print(f"Latency: {result.get('inference_ms', 0):.1f} ms")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ISL Pretrained Model Test.")
    parser.add_argument("--video", type=str, default=None, help="Path to ISL test video (.mp4)")
    args = parser.parse_args()
    run_offline_isl_test(args.video)


if __name__ == "__main__":
    main()
