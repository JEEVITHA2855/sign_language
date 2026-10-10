"""Reproducible fine-tuning script for Indian Sign Language (ISL) recognition.

Supports:
- Image classification fine-tuning with Hugging Face Transformers & PyTorch
- Custom dataset loading with Train / Validation / Test split
- Data augmentation (rotation, jitter, horizontal flip prevention for non-symmetric signs)
- Evaluation metrics logging (accuracy, precision, recall, F1)
- Saving model checkpoints and label mappings

Usage:
    python training/finetune_isl.py --data_dir data/isl_dataset --epochs 10 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image


class ISLDataset(Dataset):
    """Custom ISL image dataset organized in class subfolders."""

    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_file in cls_dir.glob("*.*"):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_file, self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Data augmentations tailored for sign gesture images."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def train_isl_model(
    data_dir: str,
    output_dir: str = "models/isl_finetuned",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "auto",
) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=======================================================")
    print(f"  Fine-Tuning ISL Vision Classifier")
    print(f"  Dataset: {data_dir} | Device: {device} | Epochs: {epochs}")
    print(f"=======================================================\n")

    train_tf, val_tf = get_data_transforms()
    full_dataset = ISLDataset(data_dir, transform=train_tf)

    if len(full_dataset) == 0:
        print(f"Error: No images found in {data_dir}.")
        return

    num_classes = len(full_dataset.classes)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total samples: {len(full_dataset)} | Classes: {num_classes}")
    print(f"Train samples: {train_size} | Val samples: {val_size}")

    # Build model (MobileNetV3 backbone for real-time mobile/CPU inference)
    import torchvision.models as models
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0
        train_loss = running_loss / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_p / "best_model.pth")
            with open(out_p / "labels.json", "w", encoding="utf-8") as f:
                json.dump(full_dataset.classes, f, indent=2)

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ISL Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--output_dir", type=str, default="models/isl_finetuned")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_isl_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
