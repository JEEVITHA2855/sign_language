"""ISL Alphabet Recognition adapter using Hugging Face image classification.

Uses the ``Hemg/Indian-sign-language-classification`` model for static
ISL alphabet recognition from single webcam frames.

Model Card Summary
------------------
- Model: Hemg/Indian-sign-language-classification
- Task: Image Classification
- Input: Single RGB image
- Output: ISL alphabet class
- Architecture: Fine-tuned vision classifier
- License: Apache-2.0
- Dataset: Indian Sign Language alphabet dataset
- Recognition scope: Static ISL alphabet gestures only
- Limitations: Does NOT recognize words, phrases, or dynamic signs
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import cv2

from src.recognition.recognition_result import RecognitionResult


class ISLAlphabetAdapter:
    """Loads a Hugging Face image classification model for ISL alphabet recognition.

    Falls back gracefully if transformers or the model is unavailable.
    """

    # ISL alphabet mapping — the model's class labels will be used directly
    # This is populated from the model's config after loading
    DEFAULT_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    def __init__(
        self,
        model_name: str = "Hemg/Indian-sign-language-classification",
        device: str = "auto",
        confidence_threshold: float = 0.60,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._pipeline = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._labels: List[str] = []
        self._lock = threading.Lock()

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self) -> None:
        """Load the HF image classification pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            device_idx = 0 if self.device == "cuda" else -1
            self._pipeline = hf_pipeline(
                "image-classification",
                model=self.model_name,
                device=device_idx,
            )

            # Extract labels from model config
            if hasattr(self._pipeline.model, "config") and hasattr(self._pipeline.model.config, "id2label"):
                id2label = self._pipeline.model.config.id2label
                self._labels = [str(id2label[k]) for k in sorted(id2label.keys())]
            else:
                self._labels = self.DEFAULT_LABELS

            self._loaded = True
            print(f"[ISLAlphabetAdapter] Loaded {self.model_name} on {self.device}")
            print(f"[ISLAlphabetAdapter] Classes ({len(self._labels)}): {self._labels[:10]}...")

        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._loaded = False
            print(f"[ISLAlphabetAdapter] Failed to load: {self._load_error}")

    def is_loaded(self) -> bool:
        return self._loaded and self._pipeline is not None

    def get_labels(self) -> List[str]:
        """Return the list of class labels supported by this model."""
        return list(self._labels) if self._labels else self.DEFAULT_LABELS

    def predict_frame(self, frame: np.ndarray) -> RecognitionResult:
        """Classify a single webcam frame as an ISL alphabet letter.

        Parameters
        ----------
        frame : np.ndarray
            BGR or RGB image from webcam (any resolution).

        Returns
        -------
        RecognitionResult
        """
        if not self.is_loaded():
            return RecognitionResult(
                label="MODEL_UNAVAILABLE",
                confidence=0.0,
                language="ISL",
                mode="ALPHABET",
                timestamp=time.time(),
                source="isl_alphabet_unavailable",
                metadata={"reason": self._load_error or "ISL alphabet model not loaded."},
            )

        if frame is None or frame.size == 0 or np.all(frame == 0) or float(np.std(frame)) < 1.0:
            return RecognitionResult(
                label="UNCERTAIN",
                confidence=0.0,
                language="ISL",
                mode="ALPHABET",
                timestamp=time.time(),
                source="isl_alphabet_empty",
                metadata={"reason": "Empty or blank frame provided."},
            )

        inference_start = time.perf_counter()

        try:
            from PIL import Image

            # Convert BGR to RGB if needed
            if frame.ndim == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame

            pil_image = Image.fromarray(rgb_frame)

            with self._lock:
                results = self._pipeline(pil_image, top_k=5)

            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            if results and len(results) > 0:
                top = results[0]
                label = str(top["label"]).upper()
                confidence = float(top["score"])

                top_k = [
                    {"label": str(r["label"]).upper(), "confidence": float(r["score"])}
                    for r in results[:5]
                ]

                return RecognitionResult(
                    label=label,
                    confidence=confidence,
                    language="ISL",
                    mode="ALPHABET",
                    timestamp=time.time(),
                    source="isl_alphabet_hf",
                    metadata={
                        "model": self.model_name,
                        "inference_ms": inference_ms,
                        "top_k": top_k,
                        "num_classes": len(self._labels),
                        "device": self.device,
                    },
                )

            return RecognitionResult(
                label="UNKNOWN",
                confidence=0.0,
                language="ISL",
                mode="ALPHABET",
                timestamp=time.time(),
                source="isl_alphabet_hf",
                metadata={"inference_ms": inference_ms, "reason": "No prediction returned."},
            )

        except Exception as exc:
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            return RecognitionResult(
                label="ERROR",
                confidence=0.0,
                language="ISL",
                mode="ALPHABET",
                timestamp=time.time(),
                source="isl_alphabet_error",
                metadata={
                    "error": f"{type(exc).__name__}: {exc}",
                    "inference_ms": inference_ms,
                },
            )

    def predict(self, frames: Sequence[Any]) -> RecognitionResult:
        """Adapter interface matching BaseSignRecognizer.predict(frames).

        For static alphabet recognition, uses only the last frame.
        """
        if not frames:
            return RecognitionResult(
                label="",
                confidence=0.0,
                language="ISL",
                mode="ALPHABET",
                timestamp=time.time(),
                source="isl_alphabet_empty",
                metadata={"reason": "No frames provided."},
            )
        last_frame = np.asarray(frames[-1])
        return self.predict_frame(last_frame)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for documentation."""
        return {
            "model_name": self.model_name,
            "loaded": self._loaded,
            "device": self.device,
            "num_classes": len(self._labels),
            "labels": self._labels,
            "recognition_scope": "Static ISL alphabet gestures only",
            "limitations": "Does NOT recognize words, phrases, or dynamic signs",
            "load_error": self._load_error,
        }
