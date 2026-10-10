"""Top-level compatibility module matching Abs6187/isl-models documented usage:

    from infer import SignLanguageInference

    inference = SignLanguageInference(
        model_path='best_model.pth',
        metadata_path='metadata.json'
    )
    result = inference.predict(video_or_frames)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.isl_model import (
    HierarchicalTwoStreamTransformer,
    SignLanguageInference,
    load_isl_model,
)

__all__ = ["SignLanguageInference", "HierarchicalTwoStreamTransformer", "load_isl_model"]


if __name__ == "__main__":
    import numpy as np

    print("\n" + "=" * 60)
    print("  ISL SignLanguageInference Compatibility Runner")
    print("=" * 60)
    inf = SignLanguageInference(
        model_path="models/isl/best_model_new.pth",
        metadata_path="models/isl/metadata.json",
        use_hierarchical=True,
    )
    print(f"Loaded {len(inf.class_names)} classes on {inf.device}")
    dummy_seq = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(30)]
    res = inf.predict(dummy_seq)
    print(f"Top prediction: {res['top_prediction']['class']} ({res['top_prediction']['confidence']:.1%})")
    print(f"Top 5: {res['top_k']}")
    print("=" * 60 + "\n")
