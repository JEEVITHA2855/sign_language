"""Spatial-Temporal Landmark Feature Pipeline for Dynamic Sign Recognition.

Extracts, normalizes, and temporal-aligns MediaPipe hand and pose landmarks into
consistent (T=30, D=154) numerical representations invariant to camera distance,
user position, and lighting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.landmark_extractor import LandmarkExtractor


class LandmarkFeaturePipeline:
    """End-to-end pipeline to convert video frames into invariant temporal feature arrays."""

    def __init__(
        self,
        target_seq_len: int = 30,
        feature_dim: int = 154,
        extractor: Optional[LandmarkExtractor] = None,
    ) -> None:
        self.target_seq_len = target_seq_len
        self.feature_dim = feature_dim
        self.extractor = extractor or LandmarkExtractor()

    def normalize_hand_landmarks(self, hand_lms: np.ndarray) -> np.ndarray:
        """Normalize 21x3 hand landmarks relative to wrist and bounding scale.

        Parameters
        ----------
        hand_lms : np.ndarray
            (21, 3) raw hand landmarks in image space.

        Returns
        -------
        np.ndarray
            (21, 3) invariant normalized coordinates centered at wrist.
        """
        if hand_lms is None or len(hand_lms) < 21:
            return np.zeros((21, 3), dtype=np.float32)

        coords = hand_lms.copy()
        wrist = coords[0].copy()

        # Center at wrist
        coords -= wrist

        # Compute scale (distance from wrist to middle MCP joint 9)
        scale = np.linalg.norm(coords[9, :2])
        if scale > 1e-4:
            coords /= scale
        else:
            # Fallback max dimension scale
            max_d = np.max(np.abs(coords))
            if max_d > 1e-4:
                coords /= max_d

        return coords.astype(np.float32)

    def extract_frame_features(
        self,
        frame: np.ndarray,
        all_hands_xyz: Optional[List[np.ndarray]] = None,
        handedness_labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Extract exact 154-feature vector for a single video frame.

        Layout:
        - [0:63]   : Left Hand 21x3 normalized landmarks
        - [63:126] : Right Hand 21x3 normalized landmarks
        - [126:154]: Upper Body Pose 14 keypoints (x, y) = 28 features
        """
        if frame is None or frame.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        return self.extractor.extract_isl_features(
            frame,
            all_hands_xyz=all_hands_xyz,
            handedness_labels=handedness_labels,
        )

    def interpolate_missing_frames(self, sequence: np.ndarray) -> np.ndarray:
        """Interpolate zero/missing landmark frames across the temporal dimension.

        Parameters
        ----------
        sequence : np.ndarray
            (T, D) sequence where some frames may be all zeros (detection drop).

        Returns
        -------
        np.ndarray
            (T, D) smoothed sequence with missing frames linearly interpolated.
        """
        T, D = sequence.shape
        if T <= 1:
            return sequence

        is_present = np.any(sequence != 0, axis=1)
        present_indices = np.where(is_present)[0]

        if len(present_indices) == 0:
            return sequence  # All empty, cannot interpolate

        if len(present_indices) == T:
            return sequence  # All present, no interpolation needed

        out = sequence.copy()
        for d in range(D):
            # 1D linear interpolation for feature dimension d
            feat_vals = sequence[present_indices, d]
            out[:, d] = np.interp(
                np.arange(T),
                present_indices,
                feat_vals,
            )

        return out.astype(np.float32)

    def process_sequence(
        self,
        frames: Sequence[np.ndarray],
        interpolate: bool = True,
    ) -> np.ndarray:
        """Extract normalized features from a sequence of video frames and align to (target_seq_len, 154).

        Parameters
        ----------
        frames : Sequence[np.ndarray]
            List of BGR video frames.
        interpolate : bool
            Whether to linearly interpolate missing frames.

        Returns
        -------
        np.ndarray
            (30, 154) standardized float32 array.
        """
        if not frames:
            return np.zeros((self.target_seq_len, self.feature_dim), dtype=np.float32)

        raw_features = [self.extract_frame_features(f) for f in frames]
        arr = np.stack(raw_features, axis=0)  # (N, 154)

        if interpolate:
            arr = self.interpolate_missing_frames(arr)

        N, D = arr.shape
        if N != self.target_seq_len:
            if N == 0:
                return np.zeros((self.target_seq_len, self.feature_dim), dtype=np.float32)
            indices = np.linspace(0, N - 1, self.target_seq_len, dtype=int)
            arr = arr[indices]

        return arr.astype(np.float32)

    def process_video_file(
        self,
        video_path: Union[str, Path],
        output_npy_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Process a video file on disk, extract temporal landmark features, and optionally save as .npy.

        Parameters
        ----------
        video_path : Union[str, Path]
            Path to video file (.mp4, .avi, etc.).
        output_npy_path : Optional[Union[str, Path]]
            Destination path to save the (30, 154) .npy array.

        Returns
        -------
        np.ndarray
            (30, 154) feature array.
        """
        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        cap = cv2.VideoCapture(str(v_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()

        features = self.process_sequence(frames, interpolate=True)

        if output_npy_path:
            out_p = Path(output_npy_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_p), features)

        return features
