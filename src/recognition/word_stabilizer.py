"""Stabilization and debouncing for Word / Dynamic Sign Recognition."""

from __future__ import annotations

from typing import Optional


class WordStabilizer:
    """Stabilizes word recognition predictions and prevents continuous duplicate word commits."""

    def __init__(self, window: int = 6, min_confidence: float = 0.80, cooldown_seconds: float = 1.0) -> None:
        self.window = max(1, int(window))
        self.min_confidence = float(min_confidence)
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_committed: Optional[str] = None
        self.last_commit_time: float = 0.0
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
        import time

        if not label or label in ("MODEL_UNAVAILABLE", "NONE", "UNKNOWN", "UNCERTAIN") or confidence < self.min_confidence:
            # Decay consecutive count on uncertain/missing frames to prevent noise accumulation
            if self.consecutive_count > 0:
                self.consecutive_count -= 1
            if self.consecutive_count == 0:
                self.current_candidate = None
            return None

        clean_label = label.strip().upper()
        now = time.time()

        # Cooldown check: prevent rapid-fire word chatter
        if self.cooldown_seconds > 0 and self.last_commit_time > 0 and not self.hand_released:
            if (now - self.last_commit_time) < self.cooldown_seconds:
                return None

        # If user performs the same sign without releasing hand/pausing
        if clean_label == self.last_committed and not self.hand_released:
            return None

        # Check consecutive window
        if clean_label != self.current_candidate:
            self.current_candidate = clean_label
            self.consecutive_count = 1
        else:
            self.consecutive_count += 1

        if self.consecutive_count >= self.window:
            self.last_committed = clean_label
            self.last_commit_time = now
            self.hand_released = False
            self.current_candidate = None
            self.consecutive_count = 0
            return clean_label

        return None

    def mark_released(self) -> None:
        """Mark that a sign gesture was completed and hand returned to rest."""
        self.hand_released = True
        self.last_commit_time = 0.0
        self.current_candidate = None
        self.consecutive_count = 0

    def reset(self) -> None:
        """Hard reset of word stabilizer state."""
        self.mark_released()
        self.last_committed = None