"""ISL Recognizer supporting both Alphabet and Word recognition pipelines.

Integrates:
- Alphabet mode: Hugging Face image classification (Hemg/Indian-sign-language-classification)
- Word mode: Video buffer sequence recognition adapter
- Confidence filtering and temporal smoothing integration
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.base_recognizer import BaseSignRecognizer
from src.recognition.recognition_result import RecognitionResult
from src.recognition.isl_model_adapter import ISLModelAdapter
from src.recognition.isl_alphabet_adapter import ISLAlphabetAdapter
from src.recognition.isl_word_recognizer import ISLWordRecognizer


class ISLRecognizer(BaseSignRecognizer):
    """Indian Sign Language (ISL) Recognizer supporting Alphabet and Word recognition."""

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool = True,
        mode: str = "ALPHABET",  # "ALPHABET" or "WORD"
        confidence_threshold: float = 0.50,
        device: str = "auto",
        use_hierarchical: bool = True,
    ) -> None:
        self.model_name = model_name or "Hemg/Indian-sign-language-classification"
        self.enabled = enabled
        self.mode = mode.upper()
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_hierarchical = use_hierarchical

        self.alphabet_adapter: Optional[ISLAlphabetAdapter] = None
        self.word_adapter: Optional[ISLModelAdapter] = None
        self.isl_word_recognizer: Optional[ISLWordRecognizer] = None

        if self.enabled:
            # Initialize alphabet adapter
            try:
                self.alphabet_adapter = ISLAlphabetAdapter(
                    model_name=self.model_name,
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                )
            except Exception as exc:
                print(f"[ISLRecognizer] Alphabet adapter init notice: {exc}")

            # Initialize Hugging Face ISL Word Recognizer (Abs6187/isl-models)
            try:
                self.isl_word_recognizer = ISLWordRecognizer(
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                    use_hierarchical=self.use_hierarchical,
                    enabled=True,
                )
            except Exception as exc:
                print(f"[ISLRecognizer] ISL Word Recognizer init notice: {exc}")

            # Fallback word adapter
            candidate = Path("models/isl")
            model_dir = str(candidate) if candidate.exists() else None
            labels_path = str(candidate / "labels.json") if candidate.exists() and (candidate / "labels.json").exists() else None
            try:
                self.word_adapter = ISLModelAdapter(model_dir=model_dir, labels_path=labels_path)
            except Exception as exc:
                print(f"[ISLRecognizer] Word adapter fallback notice: {exc}")

    def set_mode(self, mode: str) -> None:
        """Switch between ALPHABET and WORD recognition modes."""
        valid_modes = {"ALPHABET", "WORD"}
        if mode.upper() in valid_modes:
            self.mode = mode.upper()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    def get_mode(self) -> str:
        return self.mode

    def is_loaded(self) -> bool:
        if self.mode == "ALPHABET":
            return self.alphabet_adapter is not None and self.alphabet_adapter.is_loaded()
        elif self.mode == "WORD":
            if self.isl_word_recognizer is not None and self.isl_word_recognizer.is_loaded():
                return True
            return self.word_adapter is not None and self.word_adapter.is_loaded()
        return False

    def predict(
        self,
        frames: Sequence[Any],
        pose_landmarks: Optional[Any] = None,
    ) -> RecognitionResult:
        if not self.enabled or not frames:
            return RecognitionResult(
                label="MODEL_UNAVAILABLE",
                confidence=0.0,
                language="ISL",
                mode=self.mode,
                timestamp=time.time(),
                source="isl_unavailable",
                metadata={"reason": "ISL recognizer disabled or no frames provided."},
            )

        if self.mode == "ALPHABET":
            if self.alphabet_adapter is not None and self.alphabet_adapter.is_loaded():
                result = self.alphabet_adapter.predict(frames)
                # Apply confidence thresholding
                if result.confidence < self.confidence_threshold:
                    result.metadata["raw_label"] = result.label
                    result.metadata["raw_confidence"] = result.confidence
                    result.label = "UNCERTAIN"
                return result
            else:
                return RecognitionResult(
                    label="MODEL_UNAVAILABLE",
                    confidence=0.0,
                    language="ISL",
                    mode="ALPHABET",
                    timestamp=time.time(),
                    source="isl_alphabet_unavailable",
                    metadata={
                        "reason": (
                            self.alphabet_adapter._load_error
                            if self.alphabet_adapter is not None
                            else "ISL alphabet adapter not initialized"
                        ),
                        "adapter_exists": True,
                        "weights_loaded": False,
                    },
                )

        elif self.mode == "WORD":
            if self.isl_word_recognizer is not None and self.isl_word_recognizer.is_loaded():
                return self.isl_word_recognizer.predict(frames, pose_landmarks=pose_landmarks)
            elif self.word_adapter is not None and self.word_adapter.is_loaded():
                result = self.word_adapter.predict(frames)
                if result.confidence < self.confidence_threshold:
                    result.metadata["raw_label"] = result.label
                    result.metadata["raw_confidence"] = result.confidence
                    result.label = "UNCERTAIN"
                return result
            else:
                reason = (
                    self.isl_word_recognizer.load_error
                    if self.isl_word_recognizer is not None
                    else "ISL word recognizer unavailable."
                )
                return RecognitionResult(
                    label="MODEL_UNAVAILABLE",
                    confidence=0.0,
                    language="ISL",
                    mode="WORD",
                    timestamp=time.time(),
                    source="isl_word_placeholder",
                    metadata={
                        "reason": reason,
                        "adapter_exists": True,
                        "weights_loaded": False,
                        "inference_available": False,
                    },
                )

        return RecognitionResult(
            label="UNKNOWN_MODE",
            confidence=0.0,
            language="ISL",
            mode=self.mode,
            timestamp=time.time(),
            source="isl_error",
            metadata={"reason": f"Unknown mode: {self.mode}"},
        )
