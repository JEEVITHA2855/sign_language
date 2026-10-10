from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignSegmenter:
    """Simple state machine for detecting sign start/end boundaries."""

    min_sign_frames: int = 8
    max_sign_frames: int = 64
    movement_threshold: float = 0.015
    pause_frames: int = 8

    def __post_init__(self) -> None:
        self.state = "IDLE"
        self.sign_frames = 0
        self.pause_count = 0

    def reset(self) -> None:
        self.state = "IDLE"
        self.sign_frames = 0
        self.pause_count = 0

    def update(self, movement: float, confidence: float, timestamp: float | None = None) -> str:
        if self.state == "IDLE":
            if movement >= self.movement_threshold and confidence >= 0.5:
                self.state = "SIGNING"
                self.sign_frames = 1
                self.pause_count = 0
                return "SIGN_STARTED"
            return "IDLE"

        if self.state == "SIGNING":
            if movement >= self.movement_threshold:
                self.sign_frames += 1
                if self.sign_frames >= self.min_sign_frames:
                    self.state = "SIGNING"
                return "SIGNING"

            self.pause_count += 1
            if self.pause_count >= self.pause_frames:
                self.state = "SIGN_ENDED"
                return "SIGN_ENDED"
            return "SIGNING"

        if self.state == "SIGN_ENDED":
            self.reset()
            return "IDLE"

        return self.state
