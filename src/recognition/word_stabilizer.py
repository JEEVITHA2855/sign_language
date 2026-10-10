"""Stabilization and debouncing for Word / Dynamic Sign Recognition."""

from __future__ import annotations

from typing import Optional


class WordStabilizer:
    """Stabilizes word recognition predictions and prevents continuous duplicate word commits."""

    def __init__(self, window: int = 1, min_confidence: float = 0.35) -> None:
        self.window = max(1, int(window))
        self.min_confidence = float(min_confidence)
        self.last_committed: Optional[str] = None
        self.hand_released: bool = True
        self.consecutive_count: int = 0
        self.current_candidate: Optional[str] = None

    def update(self, label: str, confidence: float) -> Optional[str]:
        """Evaluate a word candidate from the temporal recognizer.

        Parameters
        ----------
        label : str
            Predicted word/sign gloss (e.g. 'HELLO', 'THANK YOU').
        confidence : float
            Confidence score [0.0, 1.0].

        Returns
        -------
        Optional[str]
            Committed word if confirmed, else None.
        """
        if not label or label in ("MODEL_UNAVAILABLE", "NONE", "UNKNOWN", "UNCERTAIN") or confidence < self.min_confidence:
            return None

        clean_label = label.strip().upper()

        # If user performs the same sign without releasing hand/pausing
        if clean_label == self.last_committed and not self.hand_released:
            return None

        # Check consecutive window if window > 1
        if clean_label != self.current_candidate:
            self.current_candidate = clean_label
            self.consecutive_count = 1
        else:
            self.consecutive_count += 1

        if self.consecutive_count >= self.window:
            self.last_committed = clean_label
            self.hand_released = False
            self.current_candidate = None
            self.consecutive_count = 0
            return clean_label

        return None

    def mark_released(self) -> None:
        """Mark that a sign gesture was completed and hand returned to rest."""
        self.hand_released = True
        self.current_candidate = None
        self.consecutive_count = 0

    def reset(self) -> None:
        """Hard reset of word stabilizer state."""
        self.mark_released()
        self.last_committed = None