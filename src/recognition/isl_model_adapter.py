from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.recognition.recognition_result import RecognitionResult


class ISLModelAdapter:
    """Adapter for the downloaded ISL checkpoints, using the actual config and model files if present."""

    def __init__(self, model_dir: str | None = None, labels_path: str | None = None):
        self.model_dir = Path(model_dir or 'models/isl_download_check')
        self.labels_path = Path(labels_path or (self.model_dir / 'labels.json'))
        self.model = None
        self.labels = self._load_labels()
        self._reason = None
        self._load_model()

    def _load_labels(self) -> List[str]:
        if self.labels_path.exists():
            try:
                with open(self.labels_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return [str(v) for v in data]
                if isinstance(data, dict):
                    return [str(v) for _, v in sorted(data.items(), key=lambda kv: kv[0])]
            except Exception:
                pass

        return []

    def _load_model(self) -> None:
        if not self.model_dir.exists():
            self._reason = 'ISL model directory is not present.'
            self.model = None
            return

        self._reason = (
            'The available ISL checkpoints are custom two-stream artifacts: '
            'they require RGB video (T, 3, 128, 128) plus 154-dimensional '
            'MediaPipe hand-and-pose landmarks and the repository-specific '
            'HierarchicalTwoStreamTransformer loader. No compatible loader is '
            'installed, so these weights are intentionally not marked loaded.'
        )
        self.model = None

    def preprocess_frames(self, frames: Sequence[Any]) -> np.ndarray:
        if not frames:
            raise ValueError('No frames provided for ISL inference.')

        rgb_frames = []
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            rgb_frames.append(arr.astype(np.float32) / 255.0)

        if len(rgb_frames) == 1:
            return np.stack([rgb_frames[0]] * 8, axis=0)

        return np.stack(rgb_frames[:30], axis=0)

    def predict(self, frames: Sequence[Any]) -> RecognitionResult:
        if self.model is None or self._reason:
            return RecognitionResult(
                label='MODEL_UNAVAILABLE',
                confidence=0.0,
                language='ISL',
                mode='WORD',
                timestamp=time.time(),
                source='isl_model',
                metadata={'reason': self._reason or 'ISL model not available.'},
            )

        raise RuntimeError('ISL inference is unavailable because no compatible checkpoint loader exists.')

    def is_loaded(self) -> bool:
        return self.model is not None
