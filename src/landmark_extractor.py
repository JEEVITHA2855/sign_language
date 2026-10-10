"""MediaPipe Hand Landmarker extraction and visualization wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple, Optional, List
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Canonical hand skeleton connection definitions
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (5, 9), (9, 13), (13, 17),
]


class LandmarkExtractor:
    """Extracts 21 3D hand landmarks using Google MediaPipe Tasks API."""

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_file = Path(model_path).resolve()
        if not model_file.exists():
            raise FileNotFoundError(
                f"Hand Landmarker task file not found at: {model_file}. "
                "Ensure models/hand_landmarker.task is downloaded."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_file))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def extract(
        self, bgr_frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[str]]:
        """Extract hand landmarks from a BGR image frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            OpenCV image in BGR format (H, W, 3).

        Returns
        -------
        Tuple[Optional[np.ndarray], List[np.ndarray], List[str]]
            - primary_hand_xyz: (21, 3) array of landmarks for the dominant hand, or None
            - all_hands_xyz: List of (21, 3) arrays for all detected hands
            - handedness_labels: List of handedness strings ('Left' / 'Right')
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return None, [], []

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        all_hands_xyz: List[np.ndarray] = []
        handedness_labels: List[str] = []

        if detection_result.hand_landmarks:
            for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
                coords = np.zeros((21, 3), dtype=np.float32)
                for i, lm in enumerate(landmarks):
                    coords[i, 0] = lm.x
                    coords[i, 1] = lm.y
                    coords[i, 2] = lm.z
                all_hands_xyz.append(coords)

                # Determine handedness label if available
                label = "Hand"
                if (
                    detection_result.handedness
                    and hand_idx < len(detection_result.handedness)
                    and detection_result.handedness[hand_idx]
                ):
                    label = detection_result.handedness[hand_idx][0].category_name
                handedness_labels.append(label)

        primary_hand = all_hands_xyz[0] if all_hands_xyz else None
        return primary_hand, all_hands_xyz, handedness_labels

    def extract_isl_features(
        self,
        bgr_frame: np.ndarray,
        all_hands_xyz: Optional[List[np.ndarray]] = None,
        handedness_labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Extract exact 154-feature vector (Hands + Pose) for ISL model input.

        Feature breakdown:
        - Left hand: 21 landmarks * 3 (x, y, z) = 63 values [0:63]
        - Right hand: 21 landmarks * 3 (x, y, z) = 63 values [63:126]
        - Upper body pose: 14 key landmarks * 2 (x, y) = 28 values [126:154]
        Total dimensions: 63 + 63 + 28 = 154 features
        """
        features = np.zeros(154, dtype=np.float32)
        if bgr_frame is None or bgr_frame.size == 0:
            return features

        if all_hands_xyz is None or handedness_labels is None:
            _, all_hands_xyz, handedness_labels = self.extract(bgr_frame)

        # Map hands into slots
        for hand_coords, label in zip(all_hands_xyz, handedness_labels):
            flat = hand_coords.flatten()[:63]
            if "Left" in label:
                features[0:len(flat)] = flat
            elif "Right" in label:
                features[63:63 + len(flat)] = flat
            else:
                if np.all(features[0:63] == 0):
                    features[0:len(flat)] = flat
                else:
                    features[63:63 + len(flat)] = flat

        # Extract pose landmarks for [126:154] (28 features)
        if not hasattr(self, "_pose_detector"):
            try:
                self._pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self._pose_detector = None

        if getattr(self, "_pose_detector", None) is not None:
            try:
                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                pose_res = self._pose_detector.process(rgb)
                if pose_res and pose_res.pose_landmarks:
                    lms = pose_res.pose_landmarks.landmark
                    upper_idx = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                    pose_coords = []
                    for idx in upper_idx:
                        if idx < len(lms):
                            pose_coords.extend([float(lms[idx].x), float(lms[idx].y)])
                        else:
                            pose_coords.extend([0.0, 0.0])
                    features[126:154] = np.array(pose_coords[:28], dtype=np.float32)
            except Exception:
                pass

        return features

    def draw_landmarks(
        self,
        frame: np.ndarray,
        all_hands_xyz: List[np.ndarray],
        handedness_labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Draw aesthetic 21-point hand skeleton overlay with connections and bounding boxes.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to draw onto.
        all_hands_xyz : List[np.ndarray]
            List of (21, 3) landmark arrays normalized to [0, 1].
        handedness_labels : List[str], optional
            Labels ('Left', 'Right') for each detected hand.

        Returns
        -------
        np.ndarray
            Frame with overlaid skeleton graphics.
        """
        h, w = frame.shape[:2]

        for hand_idx, coords in enumerate(all_hands_xyz):
            pixel_pts: List[Tuple[int, int]] = []
            xs, ys = [], []

            for i in range(21):
                px = int(np.clip(coords[i, 0] * w, 0, w - 1))
                py = int(np.clip(coords[i, 1] * h, 0, h - 1))
                pixel_pts.append((px, py))
                xs.append(px)
                ys.append(py)

            # Draw bones (skeleton connections)
            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = pixel_pts[start_idx]
                pt2 = pixel_pts[end_idx]
                cv2.line(frame, pt1, pt2, (0, 240, 255), 2, cv2.LINE_AA)

            # Draw joint nodes
            for i, (px, py) in enumerate(pixel_pts):
                # Fingertips (4, 8, 12, 16, 20) in vibrant magenta, others in cyan
                if i in (4, 8, 12, 16, 20):
                    cv2.circle(frame, (px, py), 6, (255, 0, 180), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
                elif i == 0:  # Wrist
                    cv2.circle(frame, (px, py), 7, (0, 140, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)

            # Bounding box around hand
            pad = 20
            min_x = max(0, min(xs) - pad)
            min_y = max(0, min(ys) - pad)
            max_x = min(w, max(xs) + pad)
            max_y = min(h, max(ys) + pad)

            cv2.rectangle(
                frame,
                (min_x, min_y),
                (max_x, max_y),
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Handedness tag
            label = "Hand"
            if handedness_labels and hand_idx < len(handedness_labels):
                label = handedness_labels[hand_idx]

            cv2.putText(
                frame,
                f"{label} Hand",
                (min_x, max(18, min_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 240, 255),
                1,
                cv2.LINE_AA,
            )

        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
        if hasattr(self, "_pose_detector") and self._pose_detector:
            self._pose_detector.close()
