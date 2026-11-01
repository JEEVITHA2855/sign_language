"""Test Suite for Everyday Sign Language Recognition Pipeline.

Tests:
1. Vocabulary integrity, category organization, and duplicate removal
2. Spatial-Temporal Landmark normalization and missing-frame interpolation
3. Temporal BiLSTM + Self-Attention model forward pass and prediction
4. Signer-independent dataset splitting
5. SignModelRouter multimodal mode switching
6. Common Communication Text Layer schema compliance (ISO-8601, source, confidence)
"""

from __future__ import annotations

import datetime
from pathlib import Path
import numpy as np
import pytest
import torch

from src.recognition.everyday_sign_dataset import (
    EVERYDAY_VOCABULARY,
    EVERYDAY_VOCABULARY_CATEGORIES,
    EverydaySignDataset,
    EverydaySignVocabulary,
)
from src.recognition.everyday_sign_model import EverydaySignTemporalModel
from src.recognition.landmark_feature_pipeline import LandmarkFeaturePipeline
from src.recognition.sign_model_router import SignModelRouter
from src.communication.common_text_layer import CommonTextLayer, TextEntry
from training.train_everyday_sign import create_synthetic_bootstrap_samples


class TestEverydaySignVocabulary:
    """Test vocabulary curation and encoding/decoding."""

    def test_vocabulary_size_and_no_duplicates(self):
        vocab = EverydaySignVocabulary(language="ASL")
        assert vocab.num_classes >= 50, f"Expected >= 50 classes, got {vocab.num_classes}"
        assert len(vocab.classes) == len(set(vocab.classes)), "Vocabulary contains duplicates!"

    def test_category_coverage(self):
        assert "Greetings & Communication" in EVERYDAY_VOCABULARY_CATEGORIES
        assert "Personal & People" in EVERYDAY_VOCABULARY_CATEGORIES
        assert "Education" in EVERYDAY_VOCABULARY_CATEGORIES
        assert "Healthcare & Emergency" in EVERYDAY_VOCABULARY_CATEGORIES
        assert "Everyday Activities" in EVERYDAY_VOCABULARY_CATEGORIES

        # Key signs from user specification must be present
        required_signs = [
            "HELLO", "BYE", "THANK YOU", "SORRY", "PLEASE", "WELCOME", "YES", "NO", "HELP",
            "MY NAME", "WHAT IS YOUR NAME", "I", "YOU", "MOTHER", "FATHER", "FRIEND",
            "SCHOOL", "COLLEGE", "TEACHER", "STUDENT", "BOOK", "READ", "WRITE",
            "HOSPITAL", "DOCTOR", "MEDICINE", "PAIN", "WATER", "FOOD",
            "EAT", "DRINK", "SLEEP", "GO", "COME", "HOME", "WORK"
        ]
        vocab_asl = EverydaySignVocabulary(language="ASL")
        for sign in required_signs:
            assert sign in vocab_asl.classes, f"Required sign '{sign}' missing from vocabulary!"

    def test_encode_decode(self):
        vocab = EverydaySignVocabulary(language="ISL")
        idx = vocab.encode("HOSPITAL")
        assert isinstance(idx, int)
        assert vocab.decode(idx) == "HOSPITAL"

    def test_unknown_label_raises_error(self):
        vocab = EverydaySignVocabulary(language="ASL")
        with pytest.raises(KeyError):
            vocab.encode("NON_EXISTENT_SIGN_123")


class TestLandmarkFeaturePipeline:
    """Test feature normalization and sequence processing."""

    def test_hand_landmark_normalization_wrist_centering(self):
        pipeline = LandmarkFeaturePipeline(target_seq_len=30, feature_dim=154)
        raw_lms = np.random.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)
        norm_lms = pipeline.normalize_hand_landmarks(raw_lms)

        # Wrist (index 0) must be centered at (0, 0, 0)
        assert np.allclose(norm_lms[0], [0.0, 0.0, 0.0], atol=1e-5)
        assert norm_lms.shape == (21, 3)

    def test_interpolate_missing_frames(self):
        pipeline = LandmarkFeaturePipeline(target_seq_len=30, feature_dim=154)
        seq = np.ones((10, 154), dtype=np.float32)
        seq[0, :] = 2.0
        seq[9, :] = 4.0
        # Zero out middle frames
        seq[1:9, :] = 0.0

        interpolated = pipeline.interpolate_missing_frames(seq)
        # Middle frames should be linearly interpolated, not zero
        assert np.all(interpolated[5, :] > 2.5)
        assert np.all(interpolated[5, :] < 3.5)

    def test_process_sequence_shape(self):
        pipeline = LandmarkFeaturePipeline(target_seq_len=30, feature_dim=154)
        # Simulate 15 dummy frames
        dummy_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(15)]
        features = pipeline.process_sequence(dummy_frames)
        assert features.shape == (30, 154)


class TestEverydaySignTemporalModel:
    """Test BiLSTM + Temporal Attention architecture."""

    def test_forward_pass_and_attention_shapes(self):
        model = EverydaySignTemporalModel(
            num_classes=51,
            in_features=154,
            embed_dim=128,
            hidden_dim=128,
            num_layers=2,
            bidirectional=True,
        )
        dummy_input = torch.randn(4, 30, 154)
        logits, attn = model(dummy_input, return_attention=True)

        assert logits.shape == (4, 51)
        assert attn.shape == (4, 30, 1)
        # Attention weights across time must sum to 1.0
        attn_sums = torch.sum(attn, dim=1).squeeze(-1)
        assert torch.allclose(attn_sums, torch.ones(4), atol=1e-5)

    def test_predict_sequence_output_contract(self):
        model = EverydaySignTemporalModel(num_classes=51, in_features=154)
        dummy_seq = torch.randn(30, 154)
        res = model.predict_sequence(dummy_seq, top_k=5)

        assert "top_index" in res
        assert "confidence" in res
        assert "top_k" in res
        assert len(res["top_k"]) == 5
        assert 0.0 <= res["confidence"] <= 1.0


class TestSignerIndependentDataset:
    """Test dataset and signer splitting."""

    def test_synthetic_bootstrap_signer_splits(self):
        vocab = EverydaySignVocabulary(language="ASL")
        train_ds, val_ds, test_ds = create_synthetic_bootstrap_samples(vocab, samples_per_class=6)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

        # Verify sample extraction
        x, y, meta = train_ds[0]
        assert x.shape == (30, 154)
        assert isinstance(y.item(), int)
        assert "signer_id" in meta


class TestSignModelRouter:
    """Test central multi-language and multi-mode router."""

    def test_router_initialization_and_mode_switching(self):
        router = SignModelRouter(language="ASL", mode="AUTO")
        assert router.language == "ASL"
        assert router.mode == "AUTO"

        router.set_language("ISL")
        assert router.language == "ISL"

        router.set_mode("EVERYDAY")
        assert router.mode == "EVERYDAY"

    def test_router_geometric_detection_fallback(self):
        router = SignModelRouter(language="ASL", mode="EVERYDAY")
        # Empty inputs should return label "NONE"
        res = router.route_and_predict(None, None, None)
        assert res.label == "NONE"
        assert res.confidence == 0.0


class TestCommonCommunicationTextLayerSchema:
    """Test unified text stream and JSON schema specification."""

    def test_text_entry_to_dict_matches_specification(self):
        entry = TextEntry(
            text="Thank you",
            source="ISL",
            mode="WORD",
            confidence=0.92,
        )
        d = entry.to_dict()

        assert d["text"] == "Thank you"
        assert d["language"] == "ISL"
        assert d["confidence"] == 0.92
        assert d["source"] == "SIGN_RECOGNITION"
        assert "timestamp" in d

        # Validate ISO-8601 parseable
        parsed_dt = datetime.datetime.fromisoformat(d["timestamp"])
        assert parsed_dt is not None
