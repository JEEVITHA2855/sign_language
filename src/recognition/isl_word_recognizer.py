"""ISL Word Recognizer integrating the pretrained Hugging Face checkpoint (Abs6187/isl-models).

Supports:
- Hierarchical Two-Stream Transformer (Gating + 3 Groups = 28 classes)
- Monolithic Mode (8 classes: Hospital, How_Are_You_, Monday, Night, Shoe, Street, Thank You, Yellow)
- Temporal sequence preprocessing (30 frames)
- Confidence thresholding & top-5 debug metadata
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.base_recognizer import BaseSignRecognizer
from src.recognition.recognition_result import RecognitionResult
from src.models.isl_model import SignLanguageInference, load_isl_model


class ISLWordRecognizer(BaseSignRecognizer):
    """Real-Time ISL Word/Sign Recognizer powered by Abs6187/isl-models."""

    def __init__(
        self,
        model_id: str = "Abs6187/isl-models",
        checkpoint_name: str = "best_model_new.pth",
        metadata_name: str = "metadata.json",
        model_dir: Optional[str] = "models/isl",
        confidence_threshold: float = 0.50,
        device: str = "auto",
        use_hierarchical: bool = True,
        enabled: bool = True,
    ) -> None:
        self.model_id = model_id
        self.checkpoint_name = checkpoint_name
        self.metadata_name = metadata_name
        self.model_dir = Path(model_dir or "models/isl")
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_hierarchical = use_hierarchical
        self.enabled = enabled

        self.inference: Optional[SignLanguageInference] = None
        self.load_error: Optional[str] = None
        self.vocab: List[str] = []

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            self.inference = load_isl_model(
                model_id=self.model_id,
                checkpoint_name=self.checkpoint_name,
                metadata_name=self.metadata_name,
                cache_dir=str(self.model_dir),
                device=self.device,
                use_hierarchical=self.use_hierarchical,
            )
            self.vocab = self.inference.class_names
            self.load_error = None
        except Exception as exc:
            self.load_error = str(exc)
            self.inference = None
            print(f"[ISLWordRecognizer] Load notice: {exc}")

    def is_loaded(self) -> bool:
        return self.inference is not None and self.inference.is_ready

    def get_vocab(self) -> List[str]:
        return list(self.vocab)

    def predict(
        self,
        sequence: Sequence[Any],
        pose_landmarks: Optional[Any] = None,
    ) -> RecognitionResult:
        """Run temporal sign prediction on a sequence of frames."""
        if not self.enabled or not self.is_loaded() or not sequence:
            reason = self.load_error or "ISL Word Recognizer not loaded or empty sequence."
            return RecognitionResult(
                label="MODEL_UNAVAILABLE",
                confidence=0.0,
                language="ISL",
                mode="WORD",
                timestamp=time.time(),
                source="isl_unavailable",
                metadata={"reason": reason, "top_k": []},
            )

        res = self.inference.predict(sequence, pose_input=pose_landmarks)
        raw_label = res.get("class", "UNKNOWN").strip()
        raw_conf = float(res.get("confidence", 0.0))
        top_k = res.get("top_k", [])
        inference_ms = float(res.get("inference_ms", 0.0))

        # Check confidence against threshold
        if raw_conf < self.confidence_threshold:
            display_label = "UNCERTAIN"
        else:
            display_label = raw_label

        metadata = {
            "raw_label": raw_label,
            "raw_confidence": raw_conf,
            "top_k": top_k,
            "inference_ms": inference_ms,
            "model_type": "HierarchicalTwoStreamTransformer (Abs6187/isl-models)",
            "vocab_size": len(self.vocab),
        }

        return RecognitionResult(
            label=display_label,
            confidence=raw_conf,
            language="ISL",
            mode="WORD",
            timestamp=time.time(),
            source="Abs6187/isl-models",
            metadata=metadata,
        )


if __name__ == "__main__":
    import numpy as np

    print("\n" + "=" * 60)
    print("  ISL Word Recognizer Standalone Verification")
    print("=" * 60)
    rec = ISLWordRecognizer(use_hierarchical=True, confidence_threshold=0.40)
    print(f"Loaded: {rec.is_loaded()} | Vocab ({len(rec.get_vocab())} signs): {rec.get_vocab()[:10]}...")
    sample_frames = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(30)]
    result = rec.predict(sample_frames)
    print(f"Prediction: {result.label} ({result.confidence:.1%})")
    print(f"Top-5 Debug Predictions:")
    for item in result.metadata.get("top_k", []):
        print(f"  - {item['class']:<15} {item['confidence']:.2%}")
    print(f"Latency: {result.metadata.get('inference_ms', 0):.1f} ms")
    print("=" * 60 + "\n")
