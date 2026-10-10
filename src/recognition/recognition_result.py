from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RecognitionResult:
    label: str
    confidence: float
    language: str
    mode: str
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_k(self) -> list:
        return self.metadata.get("top_k", [])

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "language": self.language,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "source": self.source,
            "top_k": self.top_k,
            "metadata": self.metadata,
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
