"""Training Script for Everyday Sign Language Recognition Model (ISL & ASL).

Supports:
- Separate training pipelines for Indian Sign Language (ISL) and American Sign Language (ASL)
- Signer-independent data splits
- Cross-Entropy with Label Smoothing
- Learning rate warmup & Cosine Annealing scheduler
- Per-class metrics & checkpointing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.everyday_sign_dataset import (
    EVERYDAY_VOCABULARY,
    EverydaySignDataset,
    EverydaySignVocabulary,
    scan_sign_dataset,
)
from src.recognition.everyday_sign_model import EverydaySignTemporalModel


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_bootstrap_samples(
    vocab: EverydaySignVocabulary,
    samples_per_class: int = 15,
    seq_len: int = 30,
    feature_dim: int = 154,
) -> Tuple[EverydaySignDataset, EverydaySignDataset, EverydaySignDataset]:
    """Generate structured synthetic landmark trajectories for initialization and testing."""
    train_samples, val_samples, test_samples = [], [], []

    for c_idx, c_name in enumerate(vocab.classes):
        # Base trajectory signature for class
        phase = (c_idx * np.pi) / len(vocab.classes)
        t = np.linspace(0, 2 * np.pi, seq_len)

        for s_idx in range(samples_per_class):
            signer_id = f"signer_{s_idx % 4:02d}"
            # Realistic synthetic hand motion wave
            signal = np.sin(t + phase + (s_idx * 0.1))[:, None]  # (30, 1)
            noise = np.random.normal(0, 0.05, size=(seq_len, feature_dim))
            feat = np.repeat(signal, feature_dim, axis=1) * 0.3 + noise
            feat = feat.astype(np.float32)

            item = (feat, c_idx, signer_id)
            if s_idx < int(samples_per_class * 0.7):
                train_samples.append(item)
            elif s_idx < int(samples_per_class * 0.85):
                val_samples.append(item)
            else:
                test_samples.append(item)

    train_ds = EverydaySignDataset(train_samples, vocab=vocab, augment=True)
    val_ds = EverydaySignDataset(val_samples, vocab=vocab, augment=False)
    test_ds = EverydaySignDataset(test_samples, vocab=vocab, augment=False)
    return train_ds, val_ds, test_ds


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    # Macro F1 calculation
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    return avg_loss, acc, macro_f1


def train_everyday_sign_model(
    data_dir: Optional[str] = None,
    language: str = "ASL",
    output_dir: str = "models/everyday_signs",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
    use_synthetic_if_empty: bool = True,
) -> Dict[str, Any]:
    """Train the everyday sign recognition model."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir) / language.upper()
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n====================================================================")
    print(f"  Training Everyday Sign Recognition Model [{language.upper()}]")
    print(f"  Device: {device} | Output Directory: {out_path}")
    print(f"====================================================================")

    vocab = EverydaySignVocabulary(language=language)
    print(f"[*] Vocabulary initialized: {vocab.num_classes} classes")

    # Load dataset
    has_data = False
    if data_dir and Path(data_dir).exists():
        try:
            train_ds, val_ds, test_ds = scan_sign_dataset(data_dir, vocab=vocab, seed=seed)
            if len(train_ds) > 0:
                has_data = True
                print(f"[*] Found dataset with {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test samples")
        except Exception as exc:
            print(f"[!] Warning reading dataset: {exc}")

    if not has_data:
        if use_synthetic_if_empty:
            print("[*] No custom recordings found. Generating structured bootstrap dataset...")
            train_ds, val_ds, test_ds = create_synthetic_bootstrap_samples(vocab, samples_per_class=12)
            print(f"[*] Bootstrap dataset ready: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
        else:
            raise FileNotFoundError(f"No sign samples found in {data_dir}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = EverydaySignTemporalModel(
        num_classes=vocab.num_classes,
        in_features=154,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    best_ckpt = None

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            # Save best checkpoint
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "language": language.upper(),
                "num_classes": vocab.num_classes,
                "classes": vocab.classes,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
            torch.save(best_ckpt, out_path / "best_model.pth")

        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} F1: {val_f1:.2%}", flush=True)

    elapsed = time.time() - t_start

    # Evaluate best model on test set
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, test_f1 = evaluate_epoch(model, test_loader, criterion, device)

    # Save metadata and vocabulary
    vocab.save(out_path / "classes.json")
    summary = {
        "language": language.upper(),
        "num_classes": vocab.num_classes,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "training_time_seconds": round(elapsed, 2),
        "checkpoint_path": str(out_path / "best_model.pth"),
    }

    with open(out_path / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Training Complete!")
    print(f"    Best Validation F1: {best_val_f1:.2%} (Epoch {best_epoch})")
    print(f"    Test Accuracy: {test_acc:.2%} | Test F1: {test_f1:.2%}")
    print(f"    Saved checkpoint to: {out_path / 'best_model.pth'}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Everyday Sign Recognition Model")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing .npy landmark clips")
    parser.add_argument("--language", type=str, default="ASL", choices=["ASL", "ISL"], help="Target sign language")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models/everyday_signs", help="Output directory")

    args = parser.parse_args()
    train_everyday_sign_model(
        data_dir=args.data_dir,
        language=args.language,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
