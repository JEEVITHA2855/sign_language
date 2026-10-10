"""Confidence gating utilities for live sign recognition."""

from __future__ import annotations


class ConfidenceFilter:
    """Reject low-confidence predictions before they reach the temporal stabilizer."""

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = float(threshold)

    def accepts(self, confidence: float) -> bool:
        return float(confidence) >= self.threshold

    def filter(self, label: str, confidence: float):
        if label in (None, "", "UNKNOWN", "NONE"):
            return None
        if self.accepts(confidence):
            return label
        return None
