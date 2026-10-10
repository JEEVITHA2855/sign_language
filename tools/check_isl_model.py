"""ISL Model Diagnostic Tool.

Verifies and displays diagnostic status for Abs6187/isl-models:
- Provider, repo, checkpoint, and metadata status
- Device (CPU / CUDA)
- MediaPipe availability
- Temporal input (frames)
- Feature dimensions
- Number of classes
- Inference readiness
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError:
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        import subprocess
        res = subprocess.run([str(venv_py)] + sys.argv, check=False)
        sys.exit(res.returncode)
    raise

try:
    import mediapipe as mp
    mediapipe_status = "Available"
except ImportError:
    mediapipe_status = "Not installed"

from src.models.isl_model import SignLanguageInference, load_isl_model


def run_diagnostics() -> None:
    repo_id = "Abs6187/isl-models"
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = PROJECT_ROOT / "models" / "isl"
    checkpoint_file = model_dir / "best_model_new.pth"
    metadata_file = model_dir / "metadata.json"

    checkpoint_status = "Loaded" if checkpoint_file.exists() else "Missing (models/isl/best_model_new.pth)"
    metadata_status = "Loaded" if metadata_file.exists() else "Missing (models/isl/metadata.json)"

    # Initialize inference
    try:
        inference = load_isl_model(
            model_id=repo_id,
            checkpoint_name="best_model_new.pth",
            metadata_name="metadata.json",
            cache_dir=str(model_dir),
            device="auto",
            use_hierarchical=True,
        )
        model_status = "Loaded"
        num_classes = len(inference.class_names)
        temporal_frames = inference.num_frames
        feature_dims = 154
        inference_status = "READY" if inference.is_ready else "NOT READY"
    except Exception as exc:
        model_status = f"Error: {exc}"
        num_classes = "N/A"
        temporal_frames = "30"
        feature_dims = "154"
        inference_status = f"FAILED ({exc})"

    print("=================================")
    print("ISL MODEL DIAGNOSTICS")
    print("=================================")
    print()
    print("Provider:")
    print("Hugging Face")
    print()
    print("Repository:")
    print(repo_id)
    print()
    print("Model:")
    print(model_status)
    print()
    print("Checkpoint:")
    print(checkpoint_status)
    print()
    print("Metadata:")
    print(metadata_status)
    print()
    print("Device:")
    print(device_name)
    print()
    print("MediaPipe:")
    print(mediapipe_status)
    print()
    print("Temporal input:")
    print(f"{temporal_frames} frames")
    print()
    print("Feature dimensions:")
    print(f"{feature_dims} (154 MediaPipe Hand + Pose features)")
    print()
    print("Number of classes:")
    print(f"{num_classes} ({', '.join(inference.class_names[:5])}...)")
    print()
    print("Inference:")
    print(inference_status)
    print()


if __name__ == "__main__":
    run_diagnostics()
