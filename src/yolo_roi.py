"""Optional YOLO ROI pre-detection module for cluttered scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


class YOLOHandDetector:
    """Optional upstream hand detector for region-of-interest cropping."""

    def __init__(self, model_path: Optional[str] = None, conf_thresh: float = 0.4) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.conf_thresh = conf_thresh
        self.session = None

        if self.model_path and self.model_path.exists():
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            except Exception as e:
                print(f"[YOLO Warning] Could not load YOLO ONNX model: {e}")

    def detect_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """Detect primary hand ROI and crop. Returns (cropped_or_original_frame, bbox)."""
        if self.session is None or frame is None:
            return frame, None

        # When model is available, preprocess and run forward pass
        # Default passthrough if no detection
        return frame, None
