"""Video capture abstraction with camera fallback and synthetic testing frames."""

from __future__ import annotations

import time
from typing import Tuple, Optional
import cv2
import numpy as np


class VideoCaptureWrapper:
    """Wrapper for OpenCV VideoCapture supporting live webcam and mock stream."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mock_mode: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.mock_mode = mock_mode

        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._consecutive_read_failures = 0
        self._start_time = time.time()

        if not self.mock_mode:
            # On Windows, cv2.CAP_DSHOW provides fast camera initialization
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(camera_index)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            else:
                print(f"[Capture Warning] Could not open webcam at index {camera_index}. Falling back to mock feed.")
                self.mock_mode = True

    def read(self) -> Tuple[bool, np.ndarray]:
        """Read the next video frame.

        Returns
        -------
        Tuple[bool, np.ndarray]
            (success_flag, bgr_frame)
        """
        self._frame_count += 1

        if not self.mock_mode and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self._consecutive_read_failures = 0
                    # Mirror frame horizontally for natural webcam user interaction
                    return True, cv2.flip(frame, 1)
            except Exception as exc:
                print(f"[Capture Error] Reading camera frame: {exc}")

            self._consecutive_read_failures += 1
            if self._consecutive_read_failures >= 3:
                print(f"[Capture Warning] Camera read failed {self._consecutive_read_failures} consecutive times. Activating mock camera feed.")
                self.mock_mode = True

        # Mock mode frame generator (dark aesthetic canvas with time indicator)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (25, 22, 20)  # Dark slate background

        # Subtle grid pattern
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (35, 30, 28), 1)
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (35, 30, 28), 1)

        cv2.putText(
            frame,
            "MOCK CAMERA FEED (No physical webcam detected)",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        return True, frame

    def release(self) -> None:
        """Release camera hardware."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
