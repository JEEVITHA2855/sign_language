"""Comprehensive Unit Tests for ISL Word Recognition Integration (Abs6187/isl-models).

Validates:
1. ISL model loading (checkpoint + metadata + gating/monolithic)
2. ISL preprocessing (video frames 128x128 standardization + 154-dim landmarks)
3. ISL prediction (inference output structure, confidence, top_k)
4. ISL word debouncing (hold sign -> 1 commit; release -> re-commit allowed)
5. ISL word buffer (word addition, sentence assembly, backspace)
6. ISL -> common text layer ([ISL] source attribution, mode attribution, deduplication)
"""

from __future__ import annotations

import time
import numpy as np
import pytest
import torch

from src.models.isl_model import (
    SignLanguageInference,
    HierarchicalTwoStreamTransformer,
    load_isl_model,
)
from src.recognition.isl_word_recognizer import ISLWordRecognizer
from src.recognition.isl_recognizer import ISLRecognizer
from src.recognition.word_stabilizer import WordStabilizer
from src.recognition.word_buffer import WordBuffer
from src.communication.common_text_layer import CommonTextLayer


# ============================================================================
# 1. ISL Model Loading Tests
# ============================================================================

def test_isl_model_loading_and_architecture():
    """Verify that the Hugging Face model loads cleanly with hierarchical gating and 28 classes."""
    inference = load_isl_model(
        model_id="Abs6187/isl-models",
        checkpoint_name="best_model_new.pth",
        metadata_name="metadata.json",
        cache_dir="models/isl",
        device="cpu",
        use_hierarchical=True,
    )

    assert inference is not None
    assert inference.is_ready is True
    assert inference.num_frames == 30
    assert inference.image_size == (128, 128)
    # Check that vocabulary contains 28 signs
    assert len(inference.class_names) == 28
    assert "Monday" in inference.class_names
    assert "Thank You" in inference.class_names or "Thank You" in [c.strip() for c in inference.class_names]
    assert "Happy" in inference.class_names
    assert "Dog" in inference.class_names


def test_isl_word_recognizer_initialization():
    """Verify ISLWordRecognizer wrapper loads and reports vocabulary correctly."""
    recognizer = ISLWordRecognizer(
        model_id="Abs6187/isl-models",
        model_dir="models/isl",
        confidence_threshold=0.35,
        device="cpu",
        use_hierarchical=True,
        enabled=True,
    )

    assert recognizer.is_loaded() is True
    vocab = recognizer.get_vocab()
    assert len(vocab) == 28
    assert recognizer.load_error is None


# ============================================================================
# 2. ISL Preprocessing Tests
# ============================================================================

def test_isl_preprocessing_video_frames():
    """Verify video frames preprocessing converts sequence to (1, 30, 3, 128, 128) tensor."""
    inference = load_isl_model(use_hierarchical=True, device="cpu")

    # Simulate variable-length sequence of 18 frames
    dummy_frames = [np.ones((240, 320, 3), dtype=np.uint8) * 120 for _ in range(18)]
    tensor = inference.preprocess_video_frames(dummy_frames)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 30, 3, 128, 128)
    assert tensor.dtype == torch.float32


def test_isl_preprocessing_pose_landmarks():
    """Verify pose landmarks preprocessing formats variable inputs into (1, 30, 154) tensor."""
    inference = load_isl_model(use_hierarchical=True, device="cpu")

    # Test None input -> zero tensor
    t_zero = inference.preprocess_pose_landmarks(None)
    assert t_zero.shape == (1, 30, 154)
    assert torch.all(t_zero == 0.0)

    # Test list of variable length with 154 features
    dummy_poses = [np.random.randn(154).astype(np.float32) for _ in range(15)]
    t_pose = inference.preprocess_pose_landmarks(dummy_poses)
    assert t_pose.shape == (1, 30, 154)


# ============================================================================
# 3. ISL Prediction Tests
# ============================================================================

def test_isl_prediction_output_format():
    """Verify that inference prediction outputs label, confidence, top_k, and latency."""
    recognizer = ISLWordRecognizer(
        confidence_threshold=0.30,
        device="cpu",
        use_hierarchical=True,
    )

    dummy_seq = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(30)]
    dummy_pose = np.zeros((30, 154), dtype=np.float32)

    result = recognizer.predict(dummy_seq, pose_landmarks=dummy_pose)

    assert result is not None
    assert result.language == "ISL"
    assert result.mode == "WORD"
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

    # Test both attribute and dict-like access
    assert result.label != ""
    assert result["label"] == result.label
    assert result["confidence"] == result.confidence
    assert isinstance(result.top_k, list)
    assert len(result.top_k) > 0
    assert "class" in result.top_k[0]
    assert "confidence" in result.top_k[0]


def test_isl_recognizer_mode_switching():
    """Verify ISLRecognizer switches between ALPHABET and WORD modes cleanly."""
    isl_rec = ISLRecognizer(enabled=True, mode="WORD", device="cpu")
    assert isl_rec.get_mode() == "WORD"
    assert isl_rec.is_loaded() is True

    isl_rec.set_mode("ALPHABET")
    assert isl_rec.get_mode() == "ALPHABET"

    isl_rec.set_mode("WORD")
    assert isl_rec.get_mode() == "WORD"

    # Invalid mode raises error
    with pytest.raises(ValueError):
        isl_rec.set_mode("INVALID_MODE")


# ============================================================================
# 4. ISL Word Debouncing & State Machine Tests
# ============================================================================

def test_isl_word_debouncing_same_sign_held():
    """Verify holding the same sign produces only a single word commit."""
    stabilizer = WordStabilizer(window=1, min_confidence=0.30)

    # Frame 1: Initial detection of HELLO
    commit1 = stabilizer.update("HELLO", confidence=0.85)
    assert commit1 == "HELLO"

    # Frame 2-5: Held continuously without release -> must produce None!
    for _ in range(4):
        commit_held = stabilizer.update("HELLO", confidence=0.88)
        assert commit_held is None


def test_isl_word_debouncing_release_and_repeat():
    """Verify sign -> release -> sign produces two distinct commits."""
    stabilizer = WordStabilizer(window=1, min_confidence=0.30)

    # First sign
    c1 = stabilizer.update("HELLO", confidence=0.85)
    assert c1 == "HELLO"

    # Hand released
    stabilizer.mark_released()

    # Second identical sign
    c2 = stabilizer.update("HELLO", confidence=0.87)
    assert c2 == "HELLO"


def test_isl_word_debouncing_low_confidence_rejected():
    """Verify predictions below min_confidence are rejected."""
    stabilizer = WordStabilizer(window=1, min_confidence=0.50)

    c = stabilizer.update("HELLO", confidence=0.35)
    assert c is None


# ============================================================================
# 5. ISL Word Buffer Tests
# ============================================================================

def test_isl_word_buffer_operations():
    """Verify WordBuffer stores ISL words, constructs sentence, and supports deletion."""
    wb = WordBuffer()
    assert wb.get_sentence() == ""

    wb.add_word("HELLO")
    wb.add_word("THANK YOU")
    assert wb.get_sentence() == "HELLO THANK YOU"
    assert wb.get_words() == ["HELLO", "THANK YOU"]

    # Delete last word
    wb.delete_last_word()
    assert wb.get_sentence() == "HELLO"

    wb.clear()
    assert wb.get_sentence() == ""


# ============================================================================
# 6. ISL -> Common Communication Text Layer Tests
# ============================================================================

def test_isl_to_common_text_layer_attribution():
    """Verify common text layer attributes ISL words correctly and separates from ASL/Speech."""
    ctl = CommonTextLayer(dedup_window_sec=1.0)

    # Add ISL word
    entry1 = ctl.add_entry("THANK YOU", source="ISL", mode="WORD", confidence=0.92)
    assert entry1 is not None
    assert entry1.source == "ISL"
    assert entry1.mode == "WORD"
    assert entry1.text == "THANK YOU"

    # Add ASL alphabet
    entry2 = ctl.add_entry("A", source="ASL", mode="ALPHABET", confidence=0.98)
    assert entry2 is not None
    assert entry2.source == "ASL"

    # Add Speech
    entry3 = ctl.add_entry("HOW ARE YOU", source="SPEECH", mode="SPEECH", confidence=0.95)
    assert entry3 is not None
    assert entry3.source == "SPEECH"

    # Transcript formatting
    transcript = ctl.get_attributed_transcript()
    assert "[ISL] THANK YOU" in transcript
    assert "[ASL] A" in transcript
    assert "[SPEECH] HOW ARE YOU" in transcript

    # Source filtering
    isl_entries = ctl.get_entries_by_source("ISL")
    assert len(isl_entries) == 1
    assert isl_entries[0].text == "THANK YOU"


def test_isl_common_text_layer_duplicate_suppression():
    """Verify common text layer deduplicates identical rapid submissions within window."""
    ctl = CommonTextLayer(dedup_window_sec=2.0, enable_deduplication=True)

    e1 = ctl.add("HELLO", source="ISL", mode="WORD", confidence=0.90)
    assert e1 is not None

    # Immediate duplicate within window should be suppressed
    e2 = ctl.add("HELLO", source="ISL", mode="WORD", confidence=0.90)
    assert e2 is None

    assert ctl.entry_count == 1


# ============================================================================
# 7. Robustness, Edge Cases & Mode Routing Tests (Requirement 12)
# ============================================================================

def test_isl_video_file_input():
    """Verify that a valid video file path completes inference through the recognizer."""
    rec = ISLWordRecognizer(device="cpu", use_hierarchical=True)
    sample_file = "samples/isl_test.mp4"
    result = rec.predict(sample_file)

    assert result is not None
    assert result.language == "ISL"
    assert result.mode == "WORD"
    assert result.label != "MODEL_UNAVAILABLE"
    assert result.confidence > 0.0
    assert result.metadata["vocab_size"] == 28


def test_isl_invalid_inputs_handled_gracefully():
    """Verify that nonexistent files, empty frames, and insufficient frames do not crash."""
    rec = ISLWordRecognizer(device="cpu", use_hierarchical=True)

    # 1. Non-existent file
    res_missing = rec.predict("samples/missing_video_12345.mp4")
    assert res_missing.label in ("ERROR", "MODEL_UNAVAILABLE", "UNCERTAIN")
    assert res_missing.confidence == 0.0

    # 2. Empty frame list
    res_empty = rec.predict([])
    assert res_empty.label in ("UNCERTAIN", "MODEL_UNAVAILABLE")

    # 3. None input
    res_none = rec.predict(None)
    assert res_none.label in ("UNCERTAIN", "MODEL_UNAVAILABLE")

    # 4. Very few frames (e.g. 2 frames) - must not crash, should interpolate to 30
    two_frames = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(2)]
    res_short = rec.predict(two_frames)
    assert res_short is not None
    assert isinstance(res_short.confidence, float)


def test_isl_vs_asl_pipeline_routing():
    """Verify ISL mode uses ISL vocabulary while ASL mode preserves ASL behavior."""
    from src.recognition.asl_recognizer import ASLRecognizer

    isl_rec = ISLRecognizer(mode="WORD", device="cpu")
    asl_rec = ASLRecognizer(enabled=True)

    dummy_frames = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(30)]

    isl_res = isl_rec.predict(dummy_frames)
    assert isl_res.language == "ISL"
    assert isl_res.mode == "WORD"

    # ASL recognizer preserves ASL language attribution
    asl_res = asl_rec.predict(dummy_frames)
    assert asl_res.language == "ASL"
    assert asl_res.mode == "WORD"

