from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from src.recognition.recognition_result import RecognitionResult


class BaseSignRecognizer(ABC):
    """Abstract interface for ASL/ISL model-backed recognition."""

    @abstractmethod
    def predict(self, frames: List[Any]) -> RecognitionResult:
        raise NotImplementedError
