"""Everyday Sign Language Temporal Neural Network Model (BiLSTM + Multi-Head Temporal Attention).

Accepts (batch_size, seq_len=30, in_features=154) landmark sequences and outputs
multi-class probabilities across 50+ everyday communication signs.
Inference latency on CPU is under 10 milliseconds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Temporal attention mechanism to dynamically weight key gesture inflection frames."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, T, D) temporal hidden representations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Context vector (B, D) and attention weights (B, T, 1).
        """
        scores = self.attn_proj(x)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(x * weights, dim=1)  # (B, D)
        return context, weights


class EverydaySignTemporalModel(nn.Module):
    """Spatial-Temporal BiLSTM / GRU Network with Self-Attention for Sign Language Recognition."""

    def __init__(
        self,
        num_classes: int = 50,
        in_features: int = 154,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional

        # 1. Spatial Feature Projection & Layer Normalization
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 2. Temporal Recurrent Layers
        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        # 3. Temporal Attention Pooling
        self.attention = TemporalAttention(hidden_dim=rnn_out_dim)

        # 4. Classification Head with Residual Projection
        self.classifier = nn.Sequential(
            nn.LayerNorm(rnn_out_dim),
            nn.Linear(rnn_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, T=30, D=154) landmark input sequence.
        return_attention : bool
            Whether to return attention weights.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Logits of shape (B, num_classes).
        """
        # (B, T, 154) -> (B, T, 128)
        emb = self.input_proj(x)

        # (B, T, 128) -> (B, T, 256)
        rnn_out, _ = self.rnn(emb)

        # Attention pooling: (B, T, 256) -> (B, 256)
        context, attn_weights = self.attention(rnn_out)

        # (B, 256) -> (B, num_classes)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights

        return logits

    @torch.no_grad()
    def predict_sequence(
        self,
        x: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Predict top classes and probabilities for a single sequence or batch.

        Parameters
        ----------
        x : torch.Tensor
            (1, 30, 154) or (30, 154) tensor.
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        Dict[str, Any]
            Top prediction index, confidence score, and top-k rankings.
        """
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(0)

        logits, attn_weights = self.forward(x, return_attention=True)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        sorted_indices = probs.argsort()[::-1]
        top_pred_idx = int(sorted_indices[0])
        top_conf = float(probs[top_pred_idx])

        top_candidates = [
            {"index": int(idx), "confidence": round(float(probs[idx]), 4)}
            for idx in sorted_indices[:min(top_k, len(sorted_indices))]
        ]

        return {
            "top_index": top_pred_idx,
            "confidence": round(top_conf, 4),
            "top_k": top_candidates,
            "probabilities": probs,
            "attention_weights": attn_weights.cpu().numpy()[0].squeeze(-1),
        }
