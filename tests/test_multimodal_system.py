"""Unit tests for the Multimodal Sign Language & Speech-to-Text System."""

import time
import pytest
import numpy as np

from src.communication.common_text_layer import CommonTextLayer, CommonTextEntry
from src.recognition.isl_recognizer import ISLRecognizer
from src.recognition.asl_recognizer import ASLRecognizer
from src.speech.speech_recognizer import SpeechRecognizer
from src.api.server import SignLanguageAPIService


def test_common_text_layer_add_and_transcript():
    layer = CommonTextLayer(max_history=50)

    layer.add_entry(text="HELLO", source="ISL", mode="ALPHABET", confidence=0.88)
    layer.add_entry(text="WORLD", source="ASL", mode="WORD", confidence=0.92)
    layer.add_entry(text="Good morning", source="SPEECH", mode="SPEECH", confidence=0.95)

    entries = layer.get_entries()
    assert len(entries) == 3
    assert entries[0].source == "ISL"
    assert entries[0].text == "HELLO"
    assert entries[1].source == "ASL"
    assert entries[2].source == "SPEECH"

    transcript = layer.get_full_transcript()
    assert "HELLO" in transcript
    assert "WORLD" in transcript
    assert "Good morning" in transcript


def test_common_text_layer_deduplication():
    layer = CommonTextLayer(enable_deduplication=True, dedup_window_seconds=2.0)

    res1 = layer.add_entry(text="HELLO", source="ISL", confidence=0.8)
    res2 = layer.add_entry(text="HELLO", source="ISL", confidence=0.8)  # Duplicate within 2s

    assert res1 is not None
    assert res2 is None  # Dropped as duplicate
    assert len(layer.get_entries()) == 1


def test_common_text_layer_listeners():
    layer = CommonTextLayer()
    received = []

    def on_entry(entry: CommonTextEntry):
        received.append(entry.text)

    layer.add_listener(on_entry)
    layer.add_entry(text="NAMASTE", source="ISL")

    assert len(received) == 1
    assert received[0] == "NAMASTE"


def test_isl_recognizer_modes():
    recognizer = ISLRecognizer(mode="ALPHABET", confidence_threshold=0.60)
    assert recognizer.get_mode() == "ALPHABET"

    recognizer.set_mode("WORD")
    assert recognizer.get_mode() == "WORD"

    # Invalid mode raises ValueError
    with pytest.raises(ValueError):
        recognizer.set_mode("INVALID_MODE")


def test_isl_recognizer_prediction_uncertain():
    recognizer = ISLRecognizer(mode="ALPHABET", confidence_threshold=0.70)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = recognizer.predict([dummy_frame])
    assert result.language == "ISL"
    assert result.confidence < 0.70 or result.label in {"UNCERTAIN", "MODEL_UNAVAILABLE", "UNKNOWN"}


def test_speech_recognizer_interface():
    speech = SpeechRecognizer(enabled=False)
    dummy_audio = np.zeros(16000, dtype=np.float32)

    result = speech.predict(dummy_audio)
    assert result.mode == "SPEECH"
    assert result.language == "speech"


def test_api_service_status_and_config():
    service = SignLanguageAPIService()
    status = service.get_status()

    assert "status" in status
    assert status["active_language"] in {"ISL", "ASL"}
    assert "models" in status

    # Test config update
    updated = service.update_config({"language": "ASL", "mode": "WORD", "isl_threshold": 0.65})
    assert updated["active_language"] == "ASL"
    assert updated["active_mode"] == "WORD"
    assert updated["thresholds"]["isl"] == 0.65


def test_api_service_speech_processing():
    service = SignLanguageAPIService()
    res = service.process_speech(text="Thank you")

    assert res["source"] == "SPEECH"
    assert res["recognized_text"] == "Thank you"
    assert "Thank you" in service.common_layer.get_full_transcript()
