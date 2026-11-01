"""Basic & Expanded Sign & Gesture Detector based on Hand Landmark Geometry.

Recognizes 100+ high-frequency everyday sign language gestures with high precision and zero latency:
- Greetings & Courtesy (HELLO, GOODBYE, PLEASE, THANK YOU, SORRY, HOW ARE YOU...)
- Affirmation & Logic (YES, NO, OK, AGREE, TRUE, FALSE...)
- Emergency & Medical (HELP, HOSPITAL, DOCTOR, SICK, PAIN...)
- Needs & Daily Living (WATER, FOOD, EAT, DRINK, RESTROOM, SLEEP...)
- Questions (WHAT, WHERE, WHO, WHY, HOW...)
- People & Pronouns (I/ME, YOU, WE, FRIEND, FAMILY...)
- Actions & Verbs (STOP, GO, COME, WORK, LEARN, PLAY, LOVE...)
- Emotions (HAPPY, SAD, ANGRY, PEACE...)
- Objects & Tech (PHONE, BOOK, COMPUTER, CALL ME...)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.recognition.expanded_sign_detector import ExpandedSignDetector


class BasicSignDetector:
    """Accurately detects 100+ sign language words from 21 MediaPipe landmarks and dual-hand geometry."""

    # Full 100+ vocabulary list
    BASIC_VOCABULARY = ExpandedSignDetector.ALL_WORDS
    VOCABULARY_CATEGORIES = ExpandedSignDetector.VOCABULARY_CATEGORIES
    SIGN_DESCRIPTIONS = ExpandedSignDetector.SIGN_DESCRIPTIONS

    def __init__(self, confidence: float = 0.85) -> None:
        self.default_confidence = confidence
        self._detector = ExpandedSignDetector(confidence=confidence)

    def detect(
        self,
        primary_hand_lms: Optional[np.ndarray],
        all_hands_lms: Optional[List[np.ndarray]] = None,
        pose_lms: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float]:
        """Classify landmarks into a recognized sign from the 100+ lexicon.

        Parameters
        ----------
        primary_hand_lms : Optional[np.ndarray]
            (21, 3) normalized Cartesian landmarks of primary hand.
        all_hands_lms : Optional[List[np.ndarray]]
            List of (21, 3) landmarks for all detected hands.
        pose_lms : Optional[np.ndarray]
            Optional (33, 3) pose coordinates.

        Returns
        -------
        Tuple[Optional[str], float]
            (sign_name, confidence) or (None, 0.0)
        """
        return self._detector.detect(
            primary_hand_lms=primary_hand_lms,
            all_hands_lms=all_hands_lms,
            pose_lms=pose_lms,
        )

    def get_vocabulary(self) -> List[str]:
        """Return the complete list of 100+ supported words."""
        return self._detector.get_vocabulary()

    def get_description(self, sign_name: str) -> str:
        """Return instructions on how to perform a sign."""
        return self._detector.get_description(sign_name)
