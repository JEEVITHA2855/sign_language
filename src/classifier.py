"""Gesture classification inference module supporting ONNX Runtime and scikit-learn."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for logits along the last axis safely."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class GestureClassifier:
    """Inference engine for sign gesture classification."""

    def __init__(
        self,
        model_path: str = "models/gesture_classifier.onnx",
        labels: Optional[List[str]] = None,
        sklearn_fallback_path: Optional[str] = "models/gesture_classifier.joblib",
    ) -> None:
        self.model_path = Path(model_path)
        labels_file = self.model_path.parent / "labels.json"
        if labels_file.exists():
            import json
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
            except Exception:
                self.labels = labels or []
        else:
            self.labels = labels or []
        self.session = None
        self.input_name = None
        self.fallback_model = None

        if self.model_path.exists():
            import onnxruntime as ort

            # Set up single-thread CPU execution for sub-millisecond real-time loop
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
        elif sklearn_fallback_path and Path(sklearn_fallback_path).exists():
            import joblib

            self.fallback_model = joblib.load(sklearn_fallback_path)
        else:
            # Model not yet trained or file not found
            pass

    def is_ready(self) -> bool:
        """Check if any valid model is loaded."""
        return self.session is not None or self.fallback_model is not None

    def predict(
        self, feature_vector: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict sign class and confidence score for a 63-element feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray
            Normalized 63-element landmark vector.

        Returns
        -------
        Tuple[str, float, Dict[str, float]]
            - best_label: Predicted sign label string
            - confidence: Softmax probability of top class [0.0, 1.0]
            - probs_dict: Mapping of {label: probability} for all classes
        """
        if not self.is_ready():
            return "UNKNOWN", 0.0, {}

        features = np.asarray(feature_vector, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        probs: np.ndarray

        if self.session is not None:
            raw_outputs = self.session.run(None, {self.input_name: features})[0]
            # If the model outputs raw logits, apply softmax
            if np.any(raw_outputs < 0.0) or not np.isclose(np.sum(raw_outputs[0]), 1.0, atol=1e-3):
                probs = softmax(raw_outputs[0])
            else:
                probs = raw_outputs[0]
        else:
            # Scikit-learn fallback
            if hasattr(self.fallback_model, "predict_proba"):
                probs = self.fallback_model.predict_proba(features)[0]
            else:
                preds = self.fallback_model.predict(features)
                probs = np.zeros(len(self.labels), dtype=np.float32)
                probs[preds[0]] = 1.0

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        best_label = (
            self.labels[best_idx]
            if best_idx < len(self.labels)
            else f"CLASS_{best_idx}"
        )

        probs_dict = {
            self.labels[i] if i < len(self.labels) else f"CLASS_{i}": float(p)
            for i, p in enumerate(probs)
        }

        return best_label, confidence, probs_dict

    def get_top_k(
        self, feature_vector: np.ndarray, k: int = 3
    ) -> List[Tuple[str, float]]:
        """Return the top-K highest confidence predictions."""
        _, _, probs_dict = self.predict(feature_vector)
        sorted_items = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
        return sorted_items[:k]
