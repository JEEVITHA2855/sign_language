"""Temporal smoothing and duplicate suppression for continuous sign recognition."""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Optional

CONFIDENCE_THRESHOLD = 0.80
STABLE_FRAME_COUNT = 5
PREDICTION_WINDOW_SIZE = 5
COOLDOWN_FRAMES = 3


class TemporalSmoother:
    """Stabilize per-frame predictions while preventing duplicate letter insertion."""

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        stable_frame_count: int = 5,
        prediction_window_size: int = 5,
        cooldown_frames: int = 3,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.stable_frame_count = int(stable_frame_count)
        self.prediction_window_size = int(prediction_window_size)
        self.cooldown_frames = int(cooldown_frames)

        self.recent_predictions: Deque[str] = deque(maxlen=self.prediction_window_size)
        self.last_accepted: Optional[str] = None
        self.frames_since_accept = 0

    def reset_after_change(self) -> None:
        self.recent_predictions.clear()
        self.last_accepted = None
        self.frames_since_accept = 0

    def update(self, label: str, confidence: float) -> Optional[str]:
        if label in (None, "", "UNKNOWN", "NONE"):
            self.recent_predictions.clear()
            self.frames_since_accept = 0
            return None

        if confidence < self.confidence_threshold:
            return None

        self.recent_predictions.append(label)
        if len(self.recent_predictions) < self.prediction_window_size:
            return None

        counts = Counter(self.recent_predictions)
        top_label, top_count = counts.most_common(1)[0]
        if top_count < self.stable_frame_count:
            return None

        if self.last_accepted is None:
            self.last_accepted = top_label
            self.frames_since_accept = 0
            return top_label

        if top_label == self.last_accepted:
            self.frames_since_accept += 1
            if self.frames_since_accept < self.cooldown_frames:
                return None
            self.frames_since_accept = 0
            return top_label

        self.last_accepted = top_label
        self.frames_since_accept = 0
        return top_label
