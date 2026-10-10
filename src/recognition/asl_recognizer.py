from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.base_recognizer import BaseSignRecognizer
from src.recognition.recognition_result import RecognitionResult
from src.recognition.asl_model_adapter import ASLModelAdapter


class ASLRecognizer(BaseSignRecognizer):
    """ASL word recognizer backed by the local WLASL I3D checkpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool = True,
        model_path: str | None = None,
        labels_path: str | None = None,
    ) -> None:
        self.model_name = model_name or "WLASL-I3D-2000"
        self.enabled = enabled
        self.adapter = None
        self.load_error = None
        if self.enabled:
            candidates = []
            if model_path:
                candidates.append(Path(model_path))
            candidates.extend([
                Path("models/asl/pytorch_model.bin"),
                Path("models/asl_download_test/pytorch_model.bin"),
            ])
            selected_model = next((path for path in candidates if path.exists()), candidates[0])

            label_candidates = []
            if labels_path:
                label_candidates.append(Path(labels_path))
            label_candidates.extend([
                selected_model.parent / "labels.json",
                Path("models/asl/labels.json"),
                Path("models/asl_download_test/labels.json"),
            ])
            selected_labels = next((str(path) for path in label_candidates if path.exists()), str(selected_model.parent / "labels.json"))

            try:
                self.adapter = ASLModelAdapter(
                    model_path=str(selected_model),
                    labels_path=selected_labels,
                )
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"

    def predict(self, frames: List[Any]) -> RecognitionResult:
        if not self.enabled or not frames:
            return RecognitionResult(
                label="MODEL_UNAVAILABLE",
                confidence=0.0,
                language="ASL",
                mode="WORD",
                timestamp=0.0,
                source="asl_unavailable",
                metadata={"reason": "No ASL model loaded or no frames provided."},
            )

        if self.adapter is not None and self.adapter.is_loaded():
            return self.adapter.predict(frames)

        return RecognitionResult(
            label="MODEL_UNAVAILABLE",
            confidence=0.0,
            language="ASL",
            mode="WORD",
            timestamp=0.0,
            source="asl_placeholder",
            metadata={
                "reason": self.load_error or "ASL model weights are not available.",
                "adapter_exists": True,
                "weights_loaded": False,
            },
        )

    def is_loaded(self) -> bool:
        return self.adapter is not None and self.adapter.is_loaded()


if __name__ == "__main__":
    import numpy as np
    recognizer = ASLRecognizer()
    print(f"ASLRecognizer loaded: {recognizer.is_loaded()} (vocab size: {len(recognizer.adapter.labels) if recognizer.is_loaded() else 0})")
    sample_seq = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
    res = recognizer.predict(sample_seq)
    print(f"Prediction result: {res.label} ({res.confidence:.1%})")

