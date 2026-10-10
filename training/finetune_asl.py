"""Reproducible fine-tuning script for American Sign Language (ASL).

Supports fine-tuning on landmark coordinates (MediaPipe 63-dim features) or video clips.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_asl_landmark_model(
    data_path: str,
    output_model: str = "models/asl_landmark_classifier.joblib",
    output_labels: str = "models/asl_landmark_labels.json",
) -> None:
    """Train MLP classifier on normalized landmark arrays (.npy / .npz format)."""
    print(f"\n[ASL Training] Loading landmark data from {data_path}...")
    data = np.load(data_path)
    X = data["X"]  # Shape: (N, 63)
    y = data["y"]  # Shape: (N,)

    print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
    classes = sorted(list(set(y)))
    print(f"Classes ({len(classes)}): {classes[:10]}...")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=200,
        random_state=42,
        early_stopping=True,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_val, y_pred, zero_division=0))

    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_model)
    with open(output_labels, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    print(f"Saved model to {output_model} and labels to {output_labels}")


def main():
    parser = argparse.ArgumentParser(description="Train ASL Landmark Model")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz file containing 'X' and 'y'")
    parser.add_argument("--output", type=str, default="models/asl_landmark_classifier.joblib")
    args = parser.parse_args()

    train_asl_landmark_model(args.data, output_model=args.output)


if __name__ == "__main__":
    main()
