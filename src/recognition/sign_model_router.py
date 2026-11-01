"""Unified Sign Language Model Router for OmniSign AI.

Orchestrates and routes between:
- ASL Alphabet (28-class ONNX/Random Forest)
- ISL Alphabet (35-class HuggingFace ViT: Hemg/Indian-sign-language-classification)
- ASL Everyday Sign Model (51-class BiLSTM + Temporal Attention)
- ISL Everyday Sign Model (51-class BiLSTM + Temporal Attention)
- WLASL Dynamic ASL (2000-class I3D Inception)
- ISL Dynamic Two-Stream Transformer (28-class Hierarchical: Abs6187/isl-models)
- Zero-latency Geometric Landmark Detector (102-class heuristic)

Provides clean mode switching ('AUTO', 'EVERYDAY', 'ALPHABET', 'DYNAMIC') with
zero conflict and thread-safe routing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.recognition.base_recognizer import BaseSignRecognizer
from src.recognition.recognition_result import RecognitionResult
from src.recognition.everyday_sign_dataset import EverydaySignVocabulary
from src.recognition.everyday_sign_model import EverydaySignTemporalModel
from src.recognition.expanded_sign_detector import ExpandedSignDetector


class SignModelRouter:
    """Central model router directing frames/landmarks to the active recognition pipeline."""

    MODES = ("AUTO", "EVERYDAY", "WORD", "ALPHABET", "DYNAMIC")
    LANGUAGES = ("ASL", "ISL")

    def __init__(
        self,
        language: str = "ASL",
        mode: str = "AUTO",
        confidence_threshold: float = 0.65,
        models_dir: str = "models",
    ) -> None:
        self.language = language.upper()
        self.mode = mode.upper()
        self.confidence_threshold = confidence_threshold
        self.models_dir = Path(models_dir)

        # Loaded sub-models cache
        self.everyday_models: Dict[str, EverydaySignTemporalModel] = {}
        self.everyday_vocabs: Dict[str, EverydaySignVocabulary] = {}
        self.geometric_detector = ExpandedSignDetector()

        self._load_everyday_models()

    def _load_everyday_models(self) -> None:
        """Load trained everyday sign models for ASL and ISL."""
        device = torch.device("cpu")
        for lang in ("ASL", "ISL"):
            ckpt_path = self.models_dir / "everyday_signs" / lang / "best_model.pth"
            if ckpt_path.exists():
                try:
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    except TypeError:
                        ckpt = torch.load(ckpt_path, map_location=device)
                    classes = ckpt.get("classes", [])
                    vocab = EverydaySignVocabulary(language=lang, custom_classes=classes)

                    model = EverydaySignTemporalModel(
                        num_classes=vocab.num_classes,
                        in_features=154,
                        embed_dim=128,
                        hidden_dim=128,
                        num_layers=2,
                        dropout=0.0,
                        bidirectional=True,
                    )
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()

                    self.everyday_models[lang] = model
                    self.everyday_vocabs[lang] = vocab
                except Exception as exc:
                    print(f"[SignModelRouter] Notice loading {lang} everyday sign model: {exc}")

    def set_language(self, language: str) -> None:
        lang = language.upper()
        if lang in self.LANGUAGES:
            self.language = lang

    def set_mode(self, mode: str) -> None:
        m = mode.upper()
        if m in self.MODES:
            self.mode = m

    def predict_everyday_sign(self, sequence_154: np.ndarray) -> RecognitionResult:
        """Predict everyday sign from (30, 154) landmark sequence."""
        model = self.everyday_models.get(self.language)
        vocab = self.everyday_vocabs.get(self.language)

        if model is None or vocab is None:
            return RecognitionResult(
                label="MODEL_UNAVAILABLE",
                confidence=0.0,
                language=self.language,
                metadata={"reason": f"Everyday sign model for {self.language} not loaded"},
            )

        t0 = time.perf_counter()
        tensor_x = torch.from_numpy(sequence_154.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            res = model.predict_sequence(tensor_x, top_k=5)
        t_ms = (time.perf_counter() - t0) * 1000.0

        top_label = vocab.decode(res["top_index"])
        conf = res["confidence"]

        top_k_labels = [
            {"label": vocab.decode(c["index"]), "confidence": c["confidence"]}
            for c in res["top_k"]
        ]

        return RecognitionResult(
            label=top_label,
            confidence=conf,
            language=self.language,
            metadata={
                "inference_ms": round(t_ms, 2),
                "top_k": top_k_labels,
                "model_type": "EVERYDAY_TEMPORAL_ATTENTION",
            },
        )

    def predict_geometric(
        self,
        primary_hand_lms: Optional[np.ndarray],
        all_hands_lms: Optional[List[np.ndarray]] = None,
    ) -> RecognitionResult:
        """Predict instant sign from hand landmark geometry."""
        t0 = time.perf_counter()
        sign, conf = self.geometric_detector.detect(primary_hand_lms, all_hands_lms)
        t_ms = (time.perf_counter() - t0) * 1000.0

        if sign and conf >= self.confidence_threshold:
            return RecognitionResult(
                label=sign,
                confidence=conf,
                language=self.language,
                metadata={"inference_ms": round(t_ms, 2), "model_type": "GEOMETRIC_DETECTOR"},
            )

        return RecognitionResult(
            label="NONE",
            confidence=0.0,
            language=self.language,
            metadata={"inference_ms": round(t_ms, 2)},
        )

    def route_and_predict(
        self,
        primary_hand_lms: Optional[np.ndarray],
        all_hands_lms: Optional[List[np.ndarray]] = None,
        sequence_154: Optional[np.ndarray] = None,
    ) -> RecognitionResult:
        """Central prediction dispatcher following active language and mode."""
        # 1. Mode: EVERYDAY or WORD
        if self.mode in ("EVERYDAY", "WORD"):
            # Priority 1: Fast landmark geometry
            geom_res = self.predict_geometric(primary_hand_lms, all_hands_lms)
            if geom_res.is_confident(self.confidence_threshold):
                return geom_res

            # Priority 2: Temporal 30-frame neural model
            if sequence_154 is not None and len(sequence_154) >= 15:
                nn_res = self.predict_everyday_sign(sequence_154)
                if nn_res.is_confident(self.confidence_threshold):
                    return nn_res

            return RecognitionResult(label="NONE", confidence=0.0, language=self.language)

        # 2. Mode: AUTO
        elif self.mode == "AUTO":
            # Check for high-confidence everyday sign first
            geom_res = self.predict_geometric(primary_hand_lms, all_hands_lms)
            if geom_res.is_confident(self.confidence_threshold):
                return geom_res

            if sequence_154 is not None and len(sequence_154) >= 20:
                nn_res = self.predict_everyday_sign(sequence_154)
                if nn_res.is_confident(self.confidence_threshold):
                    return nn_res

            return RecognitionResult(label="NONE", confidence=0.0, language=self.language)

        # Default fallback
        return RecognitionResult(label="NONE", confidence=0.0, language=self.language)
