from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional, Tuple
import time


class VideoBuffer:
    """Fixed-size ring buffer for temporal sign recognition."""

    def __init__(
        self,
        max_frames: int = 64,
        target_fps: int = 15,
        min_frames: int = 8,
        sequence_length: int = 32,
    ) -> None:
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.min_frames = min_frames
        self.sequence_length = sequence_length
        self._frames: Deque[Tuple[Any, float, Optional[Any]]] = deque(maxlen=max_frames)

    def add(
        self,
        frame: Any,
        timestamp: Optional[float] = None,
        landmarks: Optional[Any] = None,
    ) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        self._frames.append((frame, ts, landmarks))

    def get_sequence(self) -> List[Any]:
        return [item[0] for item in list(self._frames)]

    def get_landmarks(self) -> List[Any]:
        lms = [item[2] for item in list(self._frames)]
        if all(lm is None for lm in lms):
            return []
        return lms

    def get_data(self) -> Tuple[List[Any], List[Any]]:
        seq = self.get_sequence()
        lms = self.get_landmarks()
        return seq, lms

    def clear(self) -> None:
        self._frames.clear()

    def is_ready(self) -> bool:
        return len(self._frames) >= self.min_frames

    def duration(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        start = self._frames[0][1]
        end = self._frames[-1][1]
        return max(0.0, end - start)

    def frame_count(self) -> int:
        return len(self._frames)
