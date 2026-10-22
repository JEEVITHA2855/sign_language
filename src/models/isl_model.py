"""ISL Hierarchical Two-Stream Transformer Model & Inference Engine.

Pretrained checkpoint: Abs6187/isl-models on Hugging Face.
Supports both monolithic (8 classes) and hierarchical gating (28 classes across 3 groups).
"""

from __future__ import annotations

import os
import sys
import json
import time
import types
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows PosixPath unpickling compatibility
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# Module mock for unpickling custom config classes saved with checkpoints
class _AutoModule(types.ModuleType):
    def __getattr__(self, name):
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls

for _mod in ("configs", "configs.config", "configs.gating_config"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _AutoModule(_mod)


# ============================================================================
# Pure PyTorch Model Architecture (Zero External Dependencies)
# ============================================================================

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs: int, reduced_chs: int):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, 1, 0, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, 1, 0, bias=True)
        self.gate_fn = nn.Hardsigmoid(inplace=True) if hasattr(nn, "Hardsigmoid") else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel: int = 3,
        stride: int = 1,
        exp_chs: Optional[int] = None,
        act_layer: Any = nn.Hardswish,
        has_se: bool = False,
        se_reduced_chs: Optional[int] = None,
    ):
        super().__init__()
        self.has_residual = (in_chs == out_chs and stride == 1)
        exp_chs = exp_chs or in_chs
        self.has_exp = exp_chs != in_chs

        if self.has_exp:
            self.conv_pw = nn.Conv2d(in_chs, exp_chs, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(exp_chs)
            self.act1 = act_layer(inplace=True)

        self.conv_dw = nn.Conv2d(
            exp_chs, exp_chs, dw_kernel, stride, dw_kernel // 2, groups=exp_chs, bias=False
        )
        self.bn2 = nn.BatchNorm2d(exp_chs)
        self.act2 = act_layer(inplace=True)

        self.se = (
            SqueezeExcite(exp_chs, se_reduced_chs)
            if has_se and se_reduced_chs
            else nn.Identity()
        )

        self.conv_pwl = nn.Conv2d(exp_chs, out_chs, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        if self.has_exp:
            x = self.act1(self.bn1(self.conv_pw(x)))
        x = self.act2(self.bn2(self.conv_dw(x)))
        x = self.se(x)
        x = self.bn3(self.conv_pwl(x))
        if self.has_residual:
            x = x + res
        return x


class Block0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_dw = nn.Conv2d(16, 16, 3, 1, 1, groups=16, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_pw = nn.Conv2d(16, 16, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act1(self.bn1(self.conv_dw(x)))
        x = self.bn2(self.conv_pw(x))
        return x + res


class Block6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(160, 960, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(960)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn1(self.conv(x)))


class MobileNetV3Large(nn.Module):
    """Exact MobileNetV3Large architecture matching weights in Abs6187/isl-models."""

    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.Hardswish(inplace=True)

        HS = nn.Hardswish
        RE = nn.ReLU

        self.blocks = nn.Sequential(
            # Block 0
            nn.Sequential(Block0()),
            # Block 1
            nn.Sequential(
                InvertedResidual(16, 24, dw_kernel=3, stride=2, exp_chs=64, act_layer=RE, has_se=False),
                InvertedResidual(24, 24, dw_kernel=3, stride=1, exp_chs=72, act_layer=RE, has_se=False),
            ),
            # Block 2
            nn.Sequential(
                InvertedResidual(24, 40, dw_kernel=5, stride=2, exp_chs=72, act_layer=RE, has_se=True, se_reduced_chs=24),
                InvertedResidual(40, 40, dw_kernel=5, stride=1, exp_chs=120, act_layer=RE, has_se=True, se_reduced_chs=32),
                InvertedResidual(40, 40, dw_kernel=5, stride=1, exp_chs=120, act_layer=RE, has_se=True, se_reduced_chs=32),
            ),
            # Block 3
            nn.Sequential(
                InvertedResidual(40, 80, dw_kernel=3, stride=2, exp_chs=240, act_layer=HS, has_se=False),
                InvertedResidual(80, 80, dw_kernel=3, stride=1, exp_chs=200, act_layer=HS, has_se=False),
                InvertedResidual(80, 80, dw_kernel=3, stride=1, exp_chs=184, act_layer=HS, has_se=False),
                InvertedResidual(80, 80, dw_kernel=3, stride=1, exp_chs=184, act_layer=HS, has_se=False),
            ),
            # Block 4
            nn.Sequential(
                InvertedResidual(80, 112, dw_kernel=3, stride=1, exp_chs=480, act_layer=HS, has_se=True, se_reduced_chs=120),
                InvertedResidual(112, 112, dw_kernel=3, stride=1, exp_chs=672, act_layer=HS, has_se=True, se_reduced_chs=168),
            ),
            # Block 5
            nn.Sequential(
                InvertedResidual(112, 160, dw_kernel=5, stride=2, exp_chs=672, act_layer=HS, has_se=True, se_reduced_chs=168),
                InvertedResidual(160, 160, dw_kernel=5, stride=1, exp_chs=960, act_layer=HS, has_se=True, se_reduced_chs=240),
                InvertedResidual(160, 160, dw_kernel=5, stride=1, exp_chs=960, act_layer=HS, has_se=True, se_reduced_chs=240),
            ),
            # Block 6
            nn.Sequential(Block6()),
        )

        self.conv_head = nn.Conv2d(960, 1280, 1, 1, 0, bias=True)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act2(self.conv_head(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 30):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class VisualStream(nn.Module):
    def __init__(self, num_transformer_layers: int = 1):
        super().__init__()
        self.cnn = MobileNetV3Large()
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.LayerNorm(256),
        )
        self.positional_encoding = PositionalEncoding(256, 30)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = self.feature_projection(feats)
        feats = feats.view(B, T, -1)
        feats = self.positional_encoding(feats)
        out = self.transformer(feats)
        return out.mean(dim=1)


class PoseMultiScaleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_branch: int,
        kernel_large: int = 7,
        has_residual: bool = False,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.has_residual = has_residual
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, out_branch, kernel_size=1),
            nn.BatchNorm1d(out_branch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_branch, out_branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_branch),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, out_branch, kernel_size=1),
            nn.BatchNorm1d(out_branch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_branch, out_branch, kernel_size=kernel_large, padding=kernel_large // 2, groups=out_branch),
            nn.BatchNorm1d(out_branch),
        )
        if has_residual and out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.branch_small(x)
        l = self.branch_large(x)
        out = torch.cat([s, l], dim=1)
        if self.has_residual:
            out = out + self.residual_proj(x)
        return F.relu(out)


class PoseStream(nn.Module):
    def __init__(self, num_transformer_layers: int = 1):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(154, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.feature_extractor = nn.Sequential(
            PoseMultiScaleBlock(128, 64, kernel_large=7, has_residual=False),
            PoseMultiScaleBlock(128, 128, kernel_large=7, has_residual=True, out_channels=256),
        )
        self.to_transformer = nn.Linear(256, 256)
        self.positional_encoding = PositionalEncoding(256, 30)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 154) -> transpose to (B, 154, T)
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.feature_extractor(x)
        # transpose back to (B, T, 256)
        x = x.transpose(1, 2)
        x = self.to_transformer(x)
        x = self.positional_encoding(x)
        out = self.transformer(x)
        return out.mean(dim=1)


class FusionModule(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.visual_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.pose_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.fusion_layer = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        gv = self.visual_gate(v)
        gp = self.pose_gate(p)
        fused = gv * v + gp * p
        return self.fusion_layer(fused)


class HierarchicalTwoStreamTransformer(nn.Module):
    """Hierarchical Two-Stream Transformer (Visual + MediaPipe Pose/Hand)."""

    def __init__(self, num_classes: int = 8, num_transformer_layers: int = 1):
        super().__init__()
        self.visual_stream = VisualStream(num_transformer_layers=num_transformer_layers)
        self.pose_stream = PoseStream(num_transformer_layers=num_transformer_layers)
        self.fusion = FusionModule(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_frames: torch.Tensor, pose_landmarks: torch.Tensor) -> torch.Tensor:
        v = self.visual_stream(video_frames)
        p = self.pose_stream(pose_landmarks)
        fused = self.fusion(v, p)
        return self.classifier(fused)


def _safe_torch_load(path: Union[str, Path], map_location: Any) -> Any:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


# ============================================================================
# Inference Pipeline (SignLanguageInference)
# ============================================================================

class SignLanguageInference:
    """Documented inference class for Abs6187/isl-models.

    Usage:
        inference = SignLanguageInference(
            model_path='models/isl/best_model_new.pth',
            metadata_path='models/isl/metadata.json'
        )
        result = inference.predict(frames)
    """

    def __init__(
        self,
        model_path: str = "models/isl/best_model_new.pth",
        metadata_path: Optional[str] = "models/isl/metadata.json",
        device: str = "auto",
        use_hierarchical: bool = True,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.use_hierarchical = use_hierarchical

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_frames = 30
        self.image_size = (128, 128)
        self.class_names: List[str] = []

        # Normalization constants (ImageNet standard)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.model: Optional[HierarchicalTwoStreamTransformer] = None
        self.gating_model: Optional[HierarchicalTwoStreamTransformer] = None
        self.group_models: Dict[int, HierarchicalTwoStreamTransformer] = {}
        self.group_class_names: Dict[int, List[str]] = {}

        self.is_ready = False
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load checkpoint(s) and metadata."""
        # 1. Load metadata if available
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.class_names = [c.strip() for c in meta.get("class_names", [])]
            except Exception as exc:
                print(f"[ISL SignLanguageInference] Metadata notice: {exc}")

        # 2. Check for hierarchical checkpoints
        base_dir = self.model_path.parent
        checkpoints_dir = base_dir / "checkpoints"
        has_hierarchical = (
            self.use_hierarchical
            and checkpoints_dir.exists()
            and (checkpoints_dir / "gating_model" / "best_model.pth").exists()
        )

        if has_hierarchical:
            try:
                # Load gating model (3 classes: Group_0, Group_1, Group_2)
                gate_p = checkpoints_dir / "gating_model" / "best_model.pth"
                gate_ckpt = _safe_torch_load(gate_p, map_location=self.device)
                self.gating_model = HierarchicalTwoStreamTransformer(
                    num_classes=3, num_transformer_layers=2
                ).to(self.device)
                self.gating_model.load_state_dict(gate_ckpt["model_state_dict"], strict=True)
                self.gating_model.eval()

                # Load group models
                group_configs = [
                    (0, checkpoints_dir / "group_0" / "best_model.pth", 10),
                    (1, checkpoints_dir / "group_1" / "best_model.pth", 10),
                    (2, checkpoints_dir / "group_2" / "best_model.pth", 8),
                ]

                all_classes = []
                for g_idx, g_path, n_cls in group_configs:
                    if g_path.exists():
                        ckpt = _safe_torch_load(g_path, map_location=self.device)
                        m = HierarchicalTwoStreamTransformer(
                            num_classes=n_cls, num_transformer_layers=1
                        ).to(self.device)
                        m.load_state_dict(ckpt["model_state_dict"], strict=True)
                        m.eval()
                        self.group_models[g_idx] = m
                        c_names = [c.strip() for c in ckpt.get("class_names", [])]
                        self.group_class_names[g_idx] = c_names
                        all_classes.extend(c_names)

                self.class_names = sorted(list(set(all_classes)))
                self.is_ready = True
                print(
                    f"[ISL SignLanguageInference] Hierarchical pipeline loaded: "
                    f"Gating + {len(self.group_models)} groups ({len(self.class_names)} total classes) on {self.device}"
                )
                return
            except Exception as exc:
                print(f"[ISL SignLanguageInference] Hierarchical load fallback: {exc}")

        # 3. Fallback to monolithic root checkpoint
        if self.model_path.exists():
            ckpt = _safe_torch_load(self.model_path, map_location=self.device)
            raw_classes = ckpt.get("class_names", [])
            if raw_classes:
                self.class_names = [c.strip() for c in raw_classes]
            num_classes = len(self.class_names) if self.class_names else 8

            self.model = HierarchicalTwoStreamTransformer(
                num_classes=num_classes, num_transformer_layers=1
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            self.model.eval()
            self.is_ready = True
            print(
                f"[ISL SignLanguageInference] Monolithic model loaded: "
                f"{num_classes} classes on {self.device}"
            )
        else:
            raise FileNotFoundError(f"ISL checkpoint not found at: {self.model_path}")

    def preprocess_video_frames(
        self, frames: Union[Sequence[np.ndarray], str, Path]
    ) -> torch.Tensor:
        """Process image frames into normalized (1, 30, 3, 128, 128) tensor."""
        # If video file path passed
        if isinstance(frames, (str, Path)):
            cap = cv2.VideoCapture(str(frames))
            raw_frames = []
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                raw_frames.append(f)
            cap.release()
            frames = raw_frames

        if not frames:
            return torch.zeros((1, self.num_frames, 3, 128, 128), dtype=torch.float32, device=self.device)

        # Temporal sampling/interpolation to 30 frames
        num_src = len(frames)
        if num_src == self.num_frames:
            sampled_indices = list(range(num_src))
        else:
            sampled_indices = np.linspace(0, num_src - 1, self.num_frames, dtype=int)

        processed = []
        for idx in sampled_indices:
            frame = frames[idx]
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame

            # Resize to (128, 128)
            resized = cv2.resize(rgb, self.image_size)

            # Normalize to [0, 1] then ImageNet standardization
            arr = resized.astype(np.float32) / 255.0
            arr = (arr - self.mean) / self.std

            # HWC -> CHW
            chw = np.transpose(arr, (2, 0, 1))
            processed.append(chw)

        # Shape: (1, 30, 3, 128, 128)
        tensor = torch.from_numpy(np.stack(processed)).unsqueeze(0).to(self.device, dtype=torch.float32)
        return tensor

    def preprocess_pose_landmarks(
        self, pose_input: Optional[Union[np.ndarray, torch.Tensor, List[Any]]] = None
    ) -> torch.Tensor:
        """Prepare (1, 30, 154) landmark tensor."""
        if pose_input is None:
            return torch.zeros((1, self.num_frames, 154), dtype=torch.float32, device=self.device)

        if isinstance(pose_input, torch.Tensor):
            if pose_input.ndim == 2:
                pose_input = pose_input.unsqueeze(0)
            return pose_input.to(self.device, dtype=torch.float32)

        arr = np.array(pose_input, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)

        # Pad or interpolate to 30 frames
        B, T, D = arr.shape
        if T != self.num_frames:
            if T == 0:
                arr = np.zeros((B, self.num_frames, 154), dtype=np.float32)
            else:
                indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
                arr = arr[:, indices, :]

        # Pad or trim features to 154
        if arr.shape[2] < 154:
            pad = np.zeros((B, self.num_frames, 154 - arr.shape[2]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=2)
        elif arr.shape[2] > 154:
            arr = arr[:, :, :154]

        return torch.from_numpy(arr).to(self.device, dtype=torch.float32)

    def predict(
        self,
        video_input: Union[Sequence[np.ndarray], str, Path, torch.Tensor],
        pose_input: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run ISL sign recognition on temporal video sequence and pose landmarks."""
        if not self.is_ready:
            return {
                "class": "MODEL_UNAVAILABLE",
                "label": "MODEL_UNAVAILABLE",
                "confidence": 0.0,
                "top_k": [],
                "top_prediction": {"class": "MODEL_UNAVAILABLE", "confidence": 0.0},
                "inference_ms": 0.0,
            }

        t0 = time.perf_counter()

        raw_frames = None
        # Format input tensors
        if isinstance(video_input, (str, Path)):
            cap = cv2.VideoCapture(str(video_input))
            frames_list = []
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                frames_list.append(f)
            cap.release()
            raw_frames = frames_list
            video_tensor = self.preprocess_video_frames(raw_frames)
        elif isinstance(video_input, torch.Tensor) and video_input.ndim == 5:
            video_tensor = video_input.to(self.device, dtype=torch.float32)
        else:
            raw_frames = video_input
            video_tensor = self.preprocess_video_frames(video_input)

        # If pose_input omitted, attempt feature extraction from raw frames
        if pose_input is None and raw_frames is not None and len(raw_frames) > 0:
            try:
                from src.landmark_extractor import LandmarkExtractor
                extractor = LandmarkExtractor()
                pose_input = [extractor.extract_isl_features(f) for f in raw_frames]
                extractor.close()
            except Exception:
                pose_input = None

        pose_tensor = self.preprocess_pose_landmarks(pose_input)

        diag_meta: Dict[str, Any] = {}

        with torch.no_grad():
            if self.gating_model is not None and self.group_models:
                # 2-Stage Hierarchical Gating
                gate_logits = self.gating_model(video_tensor, pose_tensor)
                gate_probs = F.softmax(gate_logits, dim=-1).cpu().numpy()[0]
                best_group_idx = int(np.argmax(gate_probs))
                group_confidence = float(gate_probs[best_group_idx])

                diag_meta["gating_group"] = f"Group_{best_group_idx}"
                diag_meta["gating_confidence"] = round(group_confidence, 4)
                diag_meta["gating_probs"] = {
                    f"Group_{i}": round(float(p), 4) for i, p in enumerate(gate_probs)
                }

                # Stage 2: Target group classification
                target_model = self.group_models.get(best_group_idx)
                target_classes = self.group_class_names.get(best_group_idx, [])

                if target_model and target_classes:
                    group_logits = target_model(video_tensor, pose_tensor)
                    group_probs = F.softmax(group_logits, dim=-1).cpu().numpy()[0]

                    # Mathematical joint probability: P(Group) * P(Class | Group)
                    joint_probs = group_probs * (0.5 + 0.5 * group_confidence)

                    sorted_idx = np.argsort(joint_probs)[::-1]
                    top_k = [
                        {
                            "class": target_classes[i],
                            "label": target_classes[i],
                            "confidence": round(float(joint_probs[i]), 4),
                        }
                        for i in sorted_idx[:5]
                    ]

                    best_class = target_classes[sorted_idx[0]]
                    best_conf = float(joint_probs[sorted_idx[0]])
                    diag_meta["within_group_confidence"] = round(float(group_probs[sorted_idx[0]]), 4)
                else:
                    best_class = f"Group_{best_group_idx}"
                    best_conf = group_confidence
                    top_k = [{"class": best_class, "label": best_class, "confidence": best_conf}]
            else:
                # Monolithic Model
                logits = self.model(video_tensor, pose_tensor)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                sorted_idx = np.argsort(probs)[::-1]

                top_k = [
                    {
                        "class": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                        "label": self.class_names[i] if i < len(self.class_names) else f"Class_{i}",
                        "confidence": round(float(probs[i]), 4),
                    }
                    for i in sorted_idx[:min(5, len(sorted_idx))]
                ]

                best_class = top_k[0]["class"]
                best_conf = top_k[0]["confidence"]

        inference_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "class": best_class,
            "label": best_class,
            "confidence": round(best_conf, 4),
            "top_k": top_k,
            "top_prediction": {
                "class": best_class,
                "confidence": round(best_conf, 4),
            },
            "inference_ms": round(inference_ms, 1),
            "diagnostics": diag_meta,
        }


# ============================================================================
# Model Loader Helper
# ============================================================================

def load_isl_model(
    model_id: str = "Abs6187/isl-models",
    checkpoint_name: str = "best_model_new.pth",
    metadata_name: str = "metadata.json",
    cache_dir: Optional[str] = "models/isl",
    device: str = "auto",
    use_hierarchical: bool = True,
) -> SignLanguageInference:
    """Download/verify and load the pretrained ISL model."""
    target_dir = Path(cache_dir or "models/isl")
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / checkpoint_name
    meta_path = target_dir / metadata_name

    # Download if not present
    from huggingface_hub import hf_hub_download
    import shutil

    if not model_path.exists():
        try:
            print(f"[ISL Loader] Downloading {checkpoint_name} from {model_id}...")
            downloaded = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
            shutil.copy2(downloaded, model_path)
        except Exception as exc:
            print(f"[ISL Loader] Warning: Could not download {checkpoint_name}: {exc}")

    if not meta_path.exists():
        try:
            print(f"[ISL Loader] Downloading {metadata_name} from {model_id}...")
            downloaded = hf_hub_download(repo_id=model_id, filename=metadata_name)
            shutil.copy2(downloaded, meta_path)
        except Exception as exc:
            print(f"[ISL Loader] Warning: Could not download {metadata_name}: {exc}")

    if use_hierarchical:
        h_files = [
            "checkpoints/gating_model/best_model.pth",
            "checkpoints/group_0/best_model.pth",
            "checkpoints/group_1/best_model.pth",
            "checkpoints/group_2/best_model.pth",
        ]
        for rel_file in h_files:
            dest_file = target_dir / rel_file
            if not dest_file.exists():
                try:
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[ISL Loader] Downloading {rel_file} from {model_id}...")
                    dl = hf_hub_download(repo_id=model_id, filename=rel_file)
                    shutil.copy2(dl, dest_file)
                except Exception as exc:
                    print(f"[ISL Loader] Warning: Could not download {rel_file}: {exc}")

    return SignLanguageInference(
        model_path=str(model_path),
        metadata_path=str(meta_path) if meta_path.exists() else None,
        device=device,
        use_hierarchical=use_hierarchical,
    )
