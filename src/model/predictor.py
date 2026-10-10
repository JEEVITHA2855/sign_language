"""High-level prediction wrapper that adapts the existing sign-classification model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.classifier import GestureClassifier
from src.landmark_extractor import LandmarkExtractor
from src.preprocessing import normalize_landmarks


def predict_sign(
    frame: Optional[np.ndarray],
    extractor: Optional[LandmarkExtractor] = None,
    classifier: Optional[GestureClassifier] = None,
    *,
    hand_label: str = "Hand",
) -> Dict[str, Any]:
    """Run the current project model on a single frame and return a normalized prediction.

    Returns a dictionary with the recognized label, confidence score, and metadata
    needed by the temporal stabilizer and UI without replacing the underlying
    MediaPipe + ONNX/scikit-learn pipeline already in use by the project.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return {"label": "NONE", "confidence": 0.0, "hand_detected": False, "stable": False}

    if extractor is not None:
        primary_hand, _all_hands, _handedness = extractor.extract(frame)
        if primary_hand is not None:
            feature_vec = normalize_landmarks(primary_hand)
            if classifier is None:
                raise ValueError("A GestureClassifier instance is required when an extractor is provided.")
            label, confidence, probs = classifier.predict(feature_vec)
            return {
                "label": label,
                "confidence": float(confidence),
                "hand_detected": True,
                "probs": probs,
                "stable": confidence >= 0.8,
                "hand_label": hand_label,
            }

    return {"label": "NONE", "confidence": 0.0, "hand_detected": False, "stable": False}
