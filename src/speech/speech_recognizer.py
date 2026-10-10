"""Speech-to-Text recognizer using OpenAI Whisper via Hugging Face transformers.

Supports multiple Whisper model sizes (tiny, base, small, medium, large-v3).
Uses GPU when available with automatic CPU fallback.
"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.recognition_result import RecognitionResult


class SpeechRecognizer:
    """Whisper-based speech recognizer with lazy model loading and async inference.

    Parameters
    ----------
    model_name : str
        Hugging Face model ID. Default ``openai/whisper-base``.
    language : str | None
        Force language for transcription. ``None`` = auto-detect.
    device : str
        ``"cuda"`` or ``"cpu"``. Auto-detects GPU availability.
    enabled : bool
        When False the recognizer is a no-op.
    """

    # Model metadata for documentation/evaluation
    MODEL_INFO = {
        "openai/whisper-tiny": {"params": "39M", "speed": "~32x", "wer_librispeech": "~7.6%"},
        "openai/whisper-base": {"params": "74M", "speed": "~16x", "wer_librispeech": "~5.0%"},
        "openai/whisper-small": {"params": "244M", "speed": "~6x", "wer_librispeech": "~3.4%"},
        "openai/whisper-medium": {"params": "769M", "speed": "~2x", "wer_librispeech": "~2.9%"},
        "openai/whisper-large-v3": {"params": "1.5B", "speed": "~1x", "wer_librispeech": "~2.0%"},
    }

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        language: Optional[str] = "en",
        device: str = "auto",
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.enabled = enabled
        self._pipeline = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

        # Resolve device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Lazily load the Whisper pipeline from Hugging Face."""
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            device_idx = 0 if self.device == "cuda" else -1
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                torch_dtype=torch_dtype,
                device=device_idx,
            )
            self._loaded = True
            print(f"[SpeechRecognizer] Loaded {self.model_name} on {self.device}")

        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._loaded = False
            print(f"[SpeechRecognizer] Failed to load model: {self._load_error}")

    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""
        return self._loaded and self._pipeline is not None

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> RecognitionResult:
        """Transcribe an audio segment.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 audio waveform, normalized to [-1, 1].
        sample_rate : int
            Sample rate of the audio. Whisper expects 16 kHz.

        Returns
        -------
        RecognitionResult
            Transcription result with text, confidence, and metadata.
        """
        if not self.enabled or not self.is_loaded():
            return RecognitionResult(
                label="MODEL_UNAVAILABLE",
                confidence=0.0,
                language="speech",
                mode="SPEECH",
                timestamp=time.time(),
                source="speech_unavailable",
                metadata={"reason": self._load_error or "Speech model not loaded."},
            )

        if audio is None or len(audio) == 0:
            return RecognitionResult(
                label="",
                confidence=0.0,
                language="speech",
                mode="SPEECH",
                timestamp=time.time(),
                source="speech_empty",
                metadata={"reason": "No audio provided."},
            )

        # Ensure correct dtype and shape
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.flatten()

        inference_start = time.perf_counter()

        try:
            with self._lock:
                generate_kwargs = {}
                if self.language:
                    generate_kwargs["language"] = self.language

                result = self._pipeline(
                    {"raw": audio, "sampling_rate": sample_rate},
                    generate_kwargs=generate_kwargs,
                    return_timestamps=False,
                )

            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            text = result.get("text", "").strip()

            # Whisper doesn't provide per-segment confidence directly;
            # we use a heuristic based on text length vs audio duration
            audio_duration_sec = len(audio) / sample_rate
            confidence = min(1.0, 0.85 + 0.05 * min(len(text.split()), 5)) if text else 0.0

            return RecognitionResult(
                label=text,
                confidence=confidence,
                language="speech",
                mode="SPEECH",
                timestamp=time.time(),
                source=f"whisper_{self.model_name.split('/')[-1]}",
                metadata={
                    "model": self.model_name,
                    "inference_ms": inference_ms,
                    "audio_duration_sec": round(audio_duration_sec, 2),
                    "device": self.device,
                    "language_setting": self.language,
                },
            )

        except Exception as exc:
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            return RecognitionResult(
                label="",
                confidence=0.0,
                language="speech",
                mode="SPEECH",
                timestamp=time.time(),
                source="speech_error",
                metadata={
                    "error": f"{type(exc).__name__}: {exc}",
                    "inference_ms": inference_ms,
                },
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for documentation."""
        info = self.MODEL_INFO.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "loaded": self._loaded,
            "device": self.device,
            "language": self.language,
            "load_error": self._load_error,
            **info,
        }
