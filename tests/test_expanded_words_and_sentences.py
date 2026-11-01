"""Unit tests for the 100+ Word Recognition and AI Sentence Framing Engine."""

import time
import pytest
import numpy as np

from src.recognition.expanded_sign_detector import ExpandedSignDetector
from src.recognition.basic_sign_detector import BasicSignDetector
from src.language.sentence_framing_engine import SentenceFramingEngine
from src.language.word_builder import WordBuilder
from src.api.server import SignLanguageAPIService


def _build_hand(thumb="folded", index="folded", middle="folded", ring="folded", pinky="folded"):
    """Accurately construct all 21 MediaPipe coordinates for given finger states."""
    lms = np.zeros((21, 3), dtype=np.float32)
    # 0 is wrist
    lms[0] = (0.5, 0.75, 0.0)

    # MCP positions (base of fingers)
    lms[1] = (0.42, 0.65, 0.0)  # thumb cmc
    lms[2] = (0.40, 0.55, 0.0)  # thumb mcp
    lms[5] = (0.45, 0.50, 0.0)  # index mcp
    lms[9] = (0.50, 0.48, 0.0)  # middle mcp
    lms[13] = (0.55, 0.50, 0.0) # ring mcp
    lms[17] = (0.60, 0.53, 0.0) # pinky mcp

    # Thumb
    if thumb == "up":
        lms[3] = (0.38, 0.45, 0.0)  # ip
        lms[4] = (0.38, 0.35, 0.0)  # tip
    elif thumb == "down":
        lms[2] = (0.40, 0.60, 0.0)
        lms[3] = (0.40, 0.70, 0.0)
        lms[4] = (0.40, 0.85, 0.0)
    elif thumb == "out":
        lms[3] = (0.30, 0.55, 0.0)
        lms[4] = (0.22, 0.55, 0.0)
    else:  # folded
        lms[3] = (0.45, 0.55, 0.0)
        lms[4] = (0.48, 0.55, 0.0)

    # Fingers configuration helper
    configs = [
        (index, 5, 6, 7, 8, 0.45),
        (middle, 9, 10, 11, 12, 0.50),
        (ring, 13, 14, 15, 16, 0.55),
        (pinky, 17, 18, 19, 20, 0.60),
    ]
    for state, mcp_idx, pip_idx, dip_idx, tip_idx, x_pos in configs:
        if state == "up":
            lms[pip_idx] = (x_pos, 0.38, 0.0)
            lms[dip_idx] = (x_pos, 0.28, 0.0)
            lms[tip_idx] = (x_pos, 0.18, 0.0)
        else:  # folded
            lms[pip_idx] = (x_pos, 0.55, 0.0)
            lms[dip_idx] = (x_pos, 0.62, 0.0)
            lms[tip_idx] = (x_pos, 0.68, 0.0)

    return lms


class TestExpandedSignDetector:
    """Test suite for the 100+ word sign detector."""

    def test_vocabulary_size_and_categories(self):
        detector = ExpandedSignDetector()
        assert len(detector.ALL_WORDS) >= 100, f"Expected >= 100 words, got {len(detector.ALL_WORDS)}"
        assert len(detector.VOCABULARY_CATEGORIES) >= 8
        assert "Greetings & Courtesy" in detector.VOCABULARY_CATEGORIES
        assert "Emergency & Medical" in detector.VOCABULARY_CATEGORIES
        assert "Needs & Daily Living" in detector.VOCABULARY_CATEGORIES
        assert "Action Verbs" in detector.VOCABULARY_CATEGORIES

    def test_sign_descriptions_coverage(self):
        detector = ExpandedSignDetector()
        for word in detector.ALL_WORDS:
            assert word in detector.SIGN_DESCRIPTIONS, f"Missing description for sign: {word}"
            assert len(detector.SIGN_DESCRIPTIONS[word]) > 5

    def test_detect_hello_open_palm(self):
        detector = ExpandedSignDetector()
        hand = _build_hand(thumb="out", index="up", middle="up", ring="up", pinky="up")
        sign, conf = detector.detect(hand, [hand])
        assert sign == "HELLO"
        assert conf >= 0.90

    def test_detect_yes_thumbs_up(self):
        detector = ExpandedSignDetector()
        hand = _build_hand(thumb="up", index="folded", middle="folded", ring="folded", pinky="folded")
        sign, conf = detector.detect(hand, [hand])
        assert sign == "YES"
        assert conf >= 0.90

    def test_detect_no_thumbs_down(self):
        detector = ExpandedSignDetector()
        hand = _build_hand(thumb="down", index="folded", middle="folded", ring="folded", pinky="folded")
        sign, conf = detector.detect(hand, [hand])
        assert sign == "NO"
        assert conf >= 0.90

    def test_detect_i_love_you(self):
        detector = ExpandedSignDetector()
        hand = _build_hand(thumb="out", index="up", middle="folded", ring="folded", pinky="up")
        sign, conf = detector.detect(hand, [hand])
        assert sign == "I LOVE YOU"
        assert conf >= 0.90

    def test_detect_peace_sign(self):
        detector = ExpandedSignDetector()
        hand = _build_hand(thumb="folded", index="up", middle="up", ring="folded", pinky="folded")
        sign, conf = detector.detect(hand, [hand])
        assert sign == "PEACE"
        assert conf >= 0.90

    def test_basic_sign_detector_forwards_to_expanded(self):
        basic = BasicSignDetector()
        assert len(basic.BASIC_VOCABULARY) >= 100
        hand = _build_hand(thumb="up", index="folded", middle="folded", ring="folded", pinky="folded")
        sign, conf = basic.detect(hand, [hand])
        assert sign == "YES"


class TestSentenceFramingEngine:
    """Test suite for gloss-to-sentence grammatical reconstruction."""

    def test_phrase_patterns_reconstruction(self):
        framer = SentenceFramingEngine(auto_speak=False)

        # Pattern 1: Greeting & Inquiry
        framer.add_word("HELLO")
        framer.add_word("HOW ARE YOU")
        sent = framer.frame_sentence(flush=True)
        assert "How are you" in sent

        # Pattern 2: Medical / Hospital inquiry
        framer.clear()
        framer.add_word("WHERE")
        framer.add_word("HOSPITAL")
        sent = framer.frame_sentence(flush=True)
        assert "hospital" in sent.lower()

        # Pattern 3: Basic need (water)
        framer.clear()
        framer.add_word("I")
        framer.add_word("WANT")
        framer.add_word("WATER")
        sent = framer.frame_sentence(flush=True)
        assert "water" in sent.lower()

    def test_fallback_reconstruction(self):
        framer = SentenceFramingEngine(auto_speak=False)
        framer.add_word("WE")
        framer.add_word("LEARN")
        framer.add_word("TODAY")
        sent = framer.frame_sentence(flush=True)
        assert sent.startswith("We learn today")
        assert sent.endswith(".")

    def test_auto_frame_on_pause(self):
        framer = SentenceFramingEngine(pause_timeout=0.05, auto_speak=False)
        framer.add_word("THANK YOU")
        time.sleep(0.08)
        framed = framer.check_auto_frame()
        assert framed is not None
        assert "Thank you" in framed


class TestWordBuilderVocabulary:
    """Test WordBuilder dictionary integration."""

    def test_word_builder_fuzzy_suggestions(self):
        builder = WordBuilder(vocabulary=ExpandedSignDetector.ALL_WORDS)
        for char in "HOSP":
            builder.add_letter(char)
        suggestions = builder.get_suggestions(k=3)
        assert any("HOSPITAL" in s for s in suggestions)


class TestAPIServiceIntegration:
    """Test API service endpoints for vocabulary and sentence framing."""

    def test_api_service_vocabulary_endpoint(self):
        service = SignLanguageAPIService()
        vocab = service.get_vocabulary()
        assert "total_words" in vocab
        assert vocab["total_words"] >= 100
        assert "Emergency & Medical" in vocab["categories"]
        assert "HOSPITAL" in vocab["descriptions"]

    def test_api_service_frame_sentence(self):
        service = SignLanguageAPIService()
        service.sentence_framer.add_word("HELLO")
        service.sentence_framer.add_word("FRIEND")
        res = service.frame_sentence(force=True)
        assert "sentence" in res
        assert "Hello" in res["sentence"]

    def test_api_service_process_frame_payload_structure(self):
        service = SignLanguageAPIService()
        res = service.process_frame("")
        assert "active_words" in res
        assert "framed_sentence" in res
        assert "auto_framed" in res
        assert "transcript_summary" in res
