"""Landmark normalization and feature extraction pipeline."""

from __future__ import annotations

from typing import Any
import numpy as np


def normalize_landmarks(landmarks: np.ndarray | list[Any]) -> np.ndarray:
    """Normalize 21 3D hand landmarks for scale and translation invariance.

    Parameters
    ----------
    landmarks : np.ndarray or list
        Array or list containing 21 (x, y, z) landmarks.
        Expected shape: (21, 3) or flat array of length 63.

    Returns
    -------
    np.ndarray
        Normalized 1D feature vector of shape (63,) with dtype float32.
        - Translation invariance: Wrist (index 0) shifted to (0, 0, 0).
        - Scale invariance: Coordinates scaled by the wrist-to-middle-MCP (index 9) distance.
    """
    arr = np.asarray(landmarks, dtype=np.float64)

    # Reshape if provided as a flat array of 63 elements
    if arr.ndim == 1:
        if arr.shape[0] != 63:
            raise ValueError(f"Expected 63 elements for flat landmark array, got {arr.shape[0]}")
        arr = arr.reshape((21, 3))
    elif arr.ndim == 2:
        if arr.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {arr.shape}")
    else:
        raise ValueError(f"Unsupported landmark dimension: {arr.ndim}")

    # 1. Translation invariance: center coordinates around the wrist (landmark 0)
    wrist = arr[0]
    centered = arr - wrist

    # 2. Scale invariance: normalize by reference bone length (wrist -> middle finger MCP, index 9)
    ref_dist = float(np.linalg.norm(centered[9]))
    if ref_dist < 1e-6:
        # Fallback to max span if wrist and middle MCP coincide
        ref_dist = float(np.max(np.linalg.norm(centered, axis=1))) + 1e-6

    scaled = centered / ref_dist

    return scaled.flatten().astype(np.float32)


def extract_features_from_mediapipe_hand(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe Hand Landmark object into a normalized 63-element feature vector.

    Parameters
    ----------
    hand_landmarks : iterable of Landmark objects
        List of 21 landmarks each having attributes x, y, z.

    Returns
    -------
    np.ndarray
        Normalized feature vector of shape (63,).
    """
    coords = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(hand_landmarks):
        coords[i, 0] = lm.x
        coords[i, 1] = lm.y
        coords[i, 2] = getattr(lm, "z", 0.0)

    return normalize_landmarks(coords)
