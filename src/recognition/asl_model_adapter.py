from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

from src.recognition.recognition_result import RecognitionResult


class MaxPool3dSamePadding(nn.MaxPool3d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        if stride is None:
            stride = kernel_size
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode=False)
        self._same_kernel_size = _triple(kernel_size)
        self._same_stride = _triple(stride)
        self._same_dilation = _triple(dilation)

    def forward(self, x):
        padding = []
        for size, kernel, stride, dilation in zip(
            x.shape[-3:], self._same_kernel_size, self._same_stride, self._same_dilation
        ):
            output_size = (size + stride - 1) // stride
            total = max(
                (output_size - 1) * stride
                + (kernel - 1) * dilation
                + 1
                - size,
                0,
            )
            padding.extend([total // 2, total - total // 2])
        time_before, time_after, height_before, height_after, width_before, width_after = padding
        x = nn.functional.pad(
            x,
            [width_before, width_after, height_before, height_after, time_before, time_after],
        )
        return super().forward(x)


class Unit3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, activation='relu', use_bias=False, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation == 'relu':
            x = self.relu(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        self.name = name
        self.branch0 = Unit3D(in_channels, out_channels[0], kernel_size=(1, 1, 1))
        self.branch1 = Unit3D(in_channels, out_channels[1], kernel_size=(1, 1, 1))
        self.branch1_3x3 = Unit3D(out_channels[1], out_channels[2], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.branch2 = Unit3D(in_channels, out_channels[3], kernel_size=(1, 1, 1))
        self.branch2_3x3 = Unit3D(out_channels[3], out_channels[4], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.branch3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
        self.branch3_1x1 = Unit3D(in_channels, out_channels[5], kernel_size=(1, 1, 1))

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1_3x3(self.branch1(x))
        branch2 = self.branch2_3x3(self.branch2(x))
        branch3 = self.branch3_1x1(self.branch3(x))
        outputs = [branch0, branch1, branch2, branch3]
        return torch.cat(outputs, dim=1)


class InceptionI3d(nn.Module):
    def __init__(self, num_classes=2000, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.conv3d_1a_7x7 = Unit3D(in_channels, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))
        self.maxpool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.conv3d_2b_1x1 = Unit3D(64, 64, kernel_size=(1, 1, 1))
        self.conv3d_2c_3x3 = Unit3D(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.maxpool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32], 'mixed_3b')
        self.mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], 'mixed_3c')
        self.maxpool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)
        self.mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64], 'mixed_4b')
        self.mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64], 'mixed_4c')
        self.mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64], 'mixed_4d')
        self.mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64], 'mixed_4e')
        self.mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128], 'mixed_4f')
        self.maxpool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128], 'mixed_5b')
        self.mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128], 'mixed_5c')
        self.avg_pool = nn.AvgPool3d((2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.logits = Unit3D(
            1024,
            num_classes,
            kernel_size=(1, 1, 1),
            activation='relu',
            use_bias=True,
            use_bn=False,
        )

    def forward(self, x):
        x = self.conv3d_1a_7x7(x)
        x = self.maxpool3d_2a_3x3(x)
        x = self.conv3d_2b_1x1(x)
        x = self.conv3d_2c_3x3(x)
        x = self.maxpool3d_3a_3x3(x)
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3d_4a_3x3(x)
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool3d_5a_2x2(x)
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        x = x.squeeze(3).squeeze(3)
        return x.mean(2)


class ASLModelAdapter:
    """Loads the real WLASL I3D weights and runs video-frame inference."""

    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None, device: str = 'cpu', sequence_length: int = 32):
        self.model_path = Path(model_path or 'models/asl/pytorch_model.bin')
        self.labels_path = Path(labels_path or 'models/asl/labels.json')
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.sequence_length = int(sequence_length)
        self.num_classes = 2000
        self.labels = self._load_labels()
        self.model: Optional[nn.Module] = None
        self._load_model()

    def _load_labels(self) -> List[str]:
        if self.labels_path.exists():
            try:
                with open(self.labels_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return [str(item) for item in data]
                if isinstance(data, dict):
                    return [str(v) for _, v in sorted(data.items(), key=lambda kv: kv[0])]
            except Exception:
                pass
        return [f'CLASS_{idx}' for idx in range(self.num_classes)]

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.model = None
            return

        model = InceptionI3d(num_classes=2000, in_channels=3)
        try:
            state = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(str(self.model_path), map_location=self.device)
        if not isinstance(state, dict) or not any('conv3d.weight' in key for key in state.keys()):
            raise TypeError(f'Unsupported checkpoint format in {self.model_path}.')

        mapped_state = self._map_checkpoint_keys(state)
        load_result = model.load_state_dict(mapped_state, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                'ASL checkpoint architecture mismatch: '
                f'{len(load_result.missing_keys)} missing keys and '
                f'{len(load_result.unexpected_keys)} unexpected keys.'
            )
        model.to(self.device)
        model.eval()
        self.model = model
        self.num_classes = 2000

    @staticmethod
    def _map_checkpoint_keys(state: Dict[str, Any]) -> Dict[str, Any]:
        """Translate official pytorch_i3d names to this module's names."""
        branch_names = {
            '.b0.': '.branch0.',
            '.b1a.': '.branch1.',
            '.b1b.': '.branch1_3x3.',
            '.b2a.': '.branch2.',
            '.b2b.': '.branch2_3x3.',
            '.b3b.': '.branch3_1x1.',
        }
        mapped: Dict[str, Any] = {}
        for key, value in state.items():
            if key.startswith('Conv3d_'):
                key = key[0].lower() + key[1:]
            elif key.startswith('Mixed_'):
                key = key[0].lower() + key[1:]
                for source, target in branch_names.items():
                    key = key.replace(source, target)
            mapped[key] = value
        return mapped

    def preprocess_frames(self, frames: Sequence[Any]) -> torch.Tensor:
        if not frames:
            raise ValueError('No frames provided for ASL inference.')

        processed: List[np.ndarray] = []
        for frame in frames:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[-1] != 3:
                arr = arr[:, :, :3]
            arr = arr.astype(np.float32) / 127.5 - 1.0
            arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_LINEAR)
            arr = np.asarray(arr)
            processed.append(arr)

        if len(processed) > self.sequence_length:
            processed = processed[:self.sequence_length]
        elif len(processed) < self.sequence_length:
            pad_needed = self.sequence_length - len(processed)
            for _ in range(pad_needed):
                processed.append(processed[-1].copy())

        clip = np.stack(processed, axis=0)  # T, H, W, C
        if clip.shape[-1] == 3:
            clip = np.transpose(clip, (3, 0, 1, 2))
        else:
            clip = clip[:, :, :, :3]
            clip = np.transpose(clip, (3, 0, 1, 2))
        clip = clip[np.newaxis, :, :, :, :]
        return torch.from_numpy(clip).to(self.device)

    def predict(self, frames: Sequence[Any]) -> RecognitionResult:
        if self.model is None:
            return RecognitionResult(
                label='MODEL_UNAVAILABLE',
                confidence=0.0,
                language='ASL',
                mode='WORD',
                timestamp=time.time(),
                source='asl_model',
                metadata={'reason': f'ASL model weights not found at {self.model_path}.'},
            )

        inference_start = time.perf_counter()
        tensor = self.preprocess_frames(frames)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            top_idx = int(probs.argmax().item())
            conf = float(probs[top_idx].item())
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        label = self.labels[top_idx] if top_idx < len(self.labels) else f'CLASS_{top_idx}'
        return RecognitionResult(
            label=label,
            confidence=conf,
            language='ASL',
            mode='WORD',
            timestamp=time.time(),
            source='asl_i3d_model',
            metadata={
                'num_classes': self.num_classes,
                'input_shape': list(tensor.shape),
                'top_index': top_idx,
                'inference_ms': inference_ms,
                'weights_loaded': True,
            },
        )

    def is_loaded(self) -> bool:
        return self.model is not None
