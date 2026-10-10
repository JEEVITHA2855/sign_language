"""Dedicated Webcam Test for Pretrained ISL Word/Sign Recognition (Abs6187/isl-models).

Tests only the live pipeline:
Camera -> MediaPipe -> Temporal Buffer -> Hugging Face ISL Model -> Prediction

Displays on screen:
ISL WORD MODE
Frames: X / 30
Prediction: <label>
Confidence: XX%
State: COLLECTING / PREDICTING / WAITING
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

from src.capture import VideoCaptureWrapper
from src.landmark_extractor import LandmarkExtractor
from src.recognition.isl_word_recognizer import ISLWordRecognizer
from src.recognition.video_buffer import VideoBuffer
from src.recognition.word_stabilizer import WordStabilizer


def run_isl_webcam_test(
    camera_index: int = 0,
    mock_mode: bool = False,
    max_frames: Optional[int] = None,
    confidence_threshold: float = 0.40,
) -> None:
    print("\n" + "=" * 60)
    print("  ISL WEBCAM STANDALONE TEST (Abs6187/isl-models)")
    print("=" * 60)
    print(f"  Camera Index : {camera_index} (Mock Mode: {mock_mode})")
    print(f"  Conf Cutoff  : {confidence_threshold:.0%}")
    print("=" * 60 + "\n")

    # 1. Initialize Existing Camera Abstraction
    capture = VideoCaptureWrapper(
        camera_index=camera_index,
        width=640,
        height=480,
        fps=30,
        mock_mode=mock_mode,
    )

    # 2. Shared MediaPipe Landmark Extractor
    extractor = None
    try:
        extractor = LandmarkExtractor(model_path="models/hand_landmarker.task")
    except Exception as exc:
        print(f"[Warning] HandLandmarker task not loaded: {exc}")

    # 3. ISL Word Recognizer
    recognizer = ISLWordRecognizer(
        confidence_threshold=confidence_threshold,
        device="auto",
        use_hierarchical=True,
        enabled=True,
    )

    if not recognizer.is_loaded():
        print(f"ERROR: ISL Word Recognizer failed to load: {recognizer.load_error}")
        capture.release()
        sys.exit(1)

    print(f"[Ready] ISL Model loaded with {len(recognizer.get_vocab())} vocabulary signs.")

    # 4. Temporal Video & Feature Buffer (30 frames)
    video_buffer = VideoBuffer(max_frames=45, min_frames=15, sequence_length=30)
    word_stabilizer = WordStabilizer(window=1, min_confidence=confidence_threshold)

    # State tracking: WAITING -> COLLECTING -> PREDICTING -> WAITING
    state = "WAITING"
    last_prediction = "NONE"
    last_confidence = 0.0
    last_inference_ms = 0.0
    top_5_display = []
    frame_count = 0

    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break

            ret, frame = capture.read()
            if not ret or frame is None:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            # Extract hand & pose landmarks
            primary_hand = None
            all_hands = []
            handedness = []
            isl_features = None

            if extractor is not None:
                primary_hand, all_hands, handedness = extractor.extract(frame)
                if all_hands:
                    extractor.draw_landmarks(frame, all_hands, handedness)
                    isl_features = extractor.extract_isl_features(
                        frame, all_hands_xyz=all_hands, handedness_labels=handedness
                    )

            # State Machine: GESTURE DETECTION & TEMPORAL BUFFER
            if primary_hand is not None:
                state = "COLLECTING"
                video_buffer.add(frame, landmarks=isl_features)

                # Once enough frames are collected, run prediction
                if video_buffer.frame_count() >= 30:
                    state = "PREDICTING"
                    seq_frames, seq_lms = video_buffer.get_data()
                    t0 = time.perf_counter()
                    res = recognizer.predict(seq_frames, pose_landmarks=seq_lms)
                    last_inference_ms = (time.perf_counter() - t0) * 1000.0

                    last_prediction = res.label
                    last_confidence = res.confidence
                    top_5_display = res.metadata.get("top_k", [])[:5]

                    # Update stabilizer
                    committed = word_stabilizer.update(res.label, res.confidence)
                    if committed:
                        print(f"[{state}] COMMITTED ISL WORD: {committed} ({res.confidence:.1%}) in {last_inference_ms:.1f}ms")

                    # Keep a rolling window or reset buffer
                    video_buffer.clear()
                    state = "WAITING"
            else:
                # No hand in frame
                word_stabilizer.mark_released()
                if video_buffer.frame_count() >= video_buffer.min_frames:
                    # Finalize sign if user just released hand
                    state = "PREDICTING"
                    seq_frames, seq_lms = video_buffer.get_data()
                    t0 = time.perf_counter()
                    res = recognizer.predict(seq_frames, pose_landmarks=seq_lms)
                    last_inference_ms = (time.perf_counter() - t0) * 1000.0

                    last_prediction = res.label
                    last_confidence = res.confidence
                    top_5_display = res.metadata.get("top_k", [])[:5]

                    committed = word_stabilizer.update(res.label, res.confidence)
                    if committed:
                        print(f"[{state}] COMMITTED ISL WORD: {committed} ({res.confidence:.1%}) in {last_inference_ms:.1f}ms")

                video_buffer.clear()
                state = "WAITING"

            # Render Overlay matching Requirement 20
            # Dark HUD card
            hud_w, hud_h = 320, 220
            cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 24), -1)
            cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (0, 240, 255), 1)

            cv2.putText(frame, "ISL WORD MODE", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2, cv2.LINE_AA)

            buf_len = video_buffer.frame_count()
            cv2.putText(frame, "Frames:", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
            cv2.putText(frame, f"{buf_len} / 30", (140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, "Prediction:", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
            pred_col = (0, 255, 180) if last_confidence >= confidence_threshold else (0, 180, 255)
            cv2.putText(frame, str(last_prediction), (140, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, pred_col, 2)

            cv2.putText(frame, "Confidence:", (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
            cv2.putText(frame, f"{last_confidence * 100:.0f}%", (140, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

            cv2.putText(frame, "State:", (25, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
            state_col = (0, 240, 120) if state == "PREDICTING" else ((0, 200, 255) if state == "COLLECTING" else (160, 160, 160))
            cv2.putText(frame, state, (140, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_col, 2)

            if last_inference_ms > 0:
                cv2.putText(frame, f"Latency: {last_inference_ms:.0f} ms", (25, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

            # Top-5 Box on right side if available
            if top_5_display:
                tbox_x = w - 240
                cv2.rectangle(frame, (tbox_x, 10), (w - 10, 160), (20, 20, 24), -1)
                cv2.rectangle(frame, (tbox_x, 10), (w - 10, 160), (60, 60, 70), 1)
                cv2.putText(frame, "TOP 5 ISL PREDICTIONS", (tbox_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
                for idx, item in enumerate(top_5_display):
                    c = item.get("class", item.get("label", ""))[:12]
                    cf = float(item.get("confidence", 0.0))
                    cv2.putText(
                        frame,
                        f"{c:<12} {cf:.2f}",
                        (tbox_x + 10, 55 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (200, 200, 200),
                        1,
                    )

            # Display window (if not headless)
            try:
                cv2.imshow("ISL Webcam Standalone Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key in (ord("c"), ord("C")):
                    video_buffer.clear()
                    word_stabilizer.reset()
            except Exception:
                pass

    finally:
        capture.release()
        if extractor is not None:
            extractor.close()
        cv2.destroyAllWindows()
        print("\nWebcam test finished safely.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Webcam ISL Word Recognition Test.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--mock", action="store_true", help="Run with mock feed")
    parser.add_argument("--frames", type=int, default=None, help="Run for N frames then exit")
    parser.add_argument("--threshold", type=float, default=0.40, help="Confidence threshold")
    args = parser.parse_args()

    run_isl_webcam_test(
        camera_index=args.camera,
        mock_mode=args.mock,
        max_frames=args.frames,
        confidence_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
