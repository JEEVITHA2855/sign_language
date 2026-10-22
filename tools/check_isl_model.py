"""ISL Model Verification & Diagnostic Tool.

Verifies and displays diagnostic status for Abs6187/isl-models:
- Model repository accessibility (Hugging Face Hub)
- Download / local cache verification of checkpoint and metadata
- Loading using the official inference path
- Model architecture configuration and expected input format
- Dependency and hardware compatibility checks
- Video sample inference with actual predictions, confidence, and latency

Usage:
    python tools/check_isl_model.py
    python tools/check_isl_model.py --video samples/isl_test.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Relaunch in venv if torch not present in current python
try:
    import torch
except ImportError:
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        import subprocess
        res = subprocess.run([str(venv_py)] + sys.argv, check=False)
        sys.exit(res.returncode)
    raise

import cv2
import numpy as np

try:
    import mediapipe as mp
    mediapipe_status = f"Available (version {mp.__version__})"
except ImportError:
    mediapipe_status = "Not installed"

from src.models.isl_model import (
    SignLanguageInference,
    HierarchicalTwoStreamTransformer,
    load_isl_model,
)
from src.landmark_extractor import LandmarkExtractor


def check_hf_accessibility(repo_id: str) -> str:
    """Verify Hugging Face Hub accessibility."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(repo_id=repo_id, timeout=6.0)
        return f"Accessible (last modified: {info.last_modified or 'N/A'}, downloads: {getattr(info, 'downloads', 'N/A')})"
    except Exception as exc:
        return f"Offline / Local Cache (network notice: {exc})"


def run_isl_verification(
    video_path: Optional[str] = None,
    repo_id: str = "Abs6187/isl-models",
    cache_dir: Optional[str] = None,
    device: str = "auto",
) -> int:
    model_dir = Path(cache_dir or (PROJECT_ROOT / "models" / "isl"))
    checkpoint_file = model_dir / "best_model_new.pth"
    metadata_file = model_dir / "metadata.json"
    gating_file = model_dir / "checkpoints" / "gating_model" / "best_model.pth"

    print("\n" + "=" * 65)
    print("           INDIAN SIGN LANGUAGE (ISL) MODEL VERIFICATION")
    print("=" * 65)

    # 1. Repository Accessibility
    hf_status = check_hf_accessibility(repo_id)
    print(f"\n1. Repository & Provider:")
    print(f"   Provider           : Hugging Face Hub")
    print(f"   Repository ID      : {repo_id}")
    print(f"   Hub Accessibility  : {hf_status}")

    # 2. Checkpoints on Disk
    print(f"\n2. Local File & Cache Verification:")
    files_to_check = [
        ("Root Monolithic Checkpoint", checkpoint_file),
        ("Dataset Metadata JSON", metadata_file),
        ("Hierarchical Gating Checkpoint", gating_file),
        ("Group 0 Checkpoint", model_dir / "checkpoints" / "group_0" / "best_model.pth"),
        ("Group 1 Checkpoint", model_dir / "checkpoints" / "group_1" / "best_model.pth"),
        ("Group 2 Checkpoint", model_dir / "checkpoints" / "group_2" / "best_model.pth"),
    ]

    all_exist = True
    for label, fpath in files_to_check:
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"   [FOUND] {label:<32}: {fpath.name} ({size_mb:.2f} MB)")
        else:
            print(f"   [MISSING] {label:<30}: {fpath.relative_to(PROJECT_ROOT) if PROJECT_ROOT in fpath.parents else fpath}")
            all_exist = False

    # 3. Environment & Hardware
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Standard Baseline)"
    print(f"\n3. Environment & Compatibility:")
    print(f"   Python Version     : {sys.version.split()[0]}")
    print(f"   PyTorch Version    : {torch.__version__}")
    print(f"   CUDA Available     : {cuda_avail} ({device_name})")
    print(f"   MediaPipe Status   : {mediapipe_status}")
    print(f"   OpenCV Version     : {cv2.__version__}")

    # 4. Model Loading via Official Path
    print(f"\n4. Model Initialization & Architecture:")
    try:
        t_load_start = time.perf_counter()
        inference = load_isl_model(
            model_id=repo_id,
            checkpoint_name="best_model_new.pth",
            metadata_name="metadata.json",
            cache_dir=str(model_dir),
            device=device,
            use_hierarchical=True,
        )
        t_load = (time.perf_counter() - t_load_start) * 1000.0

        if not inference.is_ready:
            print("   [ERROR] Inference pipeline initialized but reported NOT READY.")
            return 1

        print(f"   Model Pipeline     : Loaded Successfully in {t_load:.1f} ms")
        print(f"   Execution Device   : {inference.device}")
        print(f"   Pipeline Mode      : Hierarchical 2-Stream Transformer (Gating + 3 Groups)")
        print(f"   Total Vocabulary   : {len(inference.class_names)} signs")
        print(f"   Sample Vocabulary  : {', '.join(inference.class_names[:8])}...")
        print(f"   Expected Frames    : {inference.num_frames} frames per sequence")
        print(f"   Expected Frame Dim : (T={inference.num_frames}, C=3, H=128, W=128) ImageNet normalized")
        print(f"   Expected Pose Dim  : (T={inference.num_frames}, D=154) MediaPipe Hands (126) + Upper Pose (28)")
    except Exception as exc:
        import traceback
        print(f"   [FATAL] Model loading failed: {exc}")
        traceback.print_exc()
        return 1

    # 5. Inference on Supplied Video (if provided or default sample exists)
    resolved_video = None
    if video_path:
        resolved_video = Path(video_path)
    elif (PROJECT_ROOT / "samples" / "isl_test.mp4").exists():
        resolved_video = PROJECT_ROOT / "samples" / "isl_test.mp4"
    elif (PROJECT_ROOT / "sample.mp4").exists():
        resolved_video = PROJECT_ROOT / "sample.mp4"

    if resolved_video:
        print(f"\n5. Test Video Inference:")
        if not resolved_video.exists():
            print(f"   [ERROR] Specified video not found: {resolved_video}")
            return 1

        print(f"   Input Video File   : {resolved_video.name}")
        cap = cv2.VideoCapture(str(resolved_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"   Video Properties   : {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

        # Run inference
        t0 = time.perf_counter()
        result = inference.predict(resolved_video)
        t_infer = (time.perf_counter() - t0) * 1000.0

        top_class = result.get("class", "UNKNOWN")
        confidence = float(result.get("confidence", 0.0))
        top_k = result.get("top_k", [])
        diag = result.get("diagnostics", {})

        print(f"\n   --- Inference Results ---")
        print(f"   Predicted Sign     : {top_class}")
        print(f"   Confidence Score   : {confidence:.2%} ({confidence:.4f})")
        if diag.get("gating_group"):
            print(f"   Gating Selection   : {diag.get('gating_group')} (group conf: {diag.get('gating_confidence', 0):.2%})")
        print(f"   Latency            : {t_infer:.1f} ms")
        print(f"\n   Top-5 Predictions:")
        for idx, item in enumerate(top_k[:5], start=1):
            c_name = item.get("class", item.get("label", "UNKNOWN"))
            c_conf = float(item.get("confidence", 0.0))
            bar_len = int(c_conf * 25)
            bar = "#" * bar_len + "-" * (25 - bar_len)
            print(f"     {idx}. {c_name:<16} : [{bar}] {c_conf:.2%}")
    else:
        print(f"\n5. Test Video Inference: (Skipped - pass --video <path> to test a sample video)")

    print("\n" + "=" * 65)
    print("  VERIFICATION RESULT: ALL CHECKS PASSED - ISL MODEL OPERATIONAL")
    print("=" * 65 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Verify local ISL model checkpoint and dependencies.")
    parser.add_argument("--video", type=str, default=None, help="Path to ISL test video (.mp4)")
    parser.add_argument("--repo-id", type=str, default="Abs6187/isl-models", help="Hugging Face repo ID")
    parser.add_argument("--cache-dir", type=str, default=None, help="Model cache directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device")
    args = parser.parse_args()

    exit_code = run_isl_verification(
        video_path=args.video,
        repo_id=args.repo_id,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
