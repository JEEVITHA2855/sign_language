"""Intelligent Sentence Framing Engine for Sign Language Conversations.

Accumulates recognized words/signs into an active sequence and automatically frames them
into grammatically sound, fluent English sentences (e.g., ['WHERE', 'HOSPITAL'] -> 'Where is the hospital?').
Seamlessly coordinates with the Local Neural Flan-T5 model and TTS speech synthesis.
"""

from __future__ import annotations

import re
import time
from typing import Callable, List, Optional
from src.nlp.huggingface_corrector import HuggingFaceCorrector


class SentenceFramingEngine:
    """Manages word accumulation, grammar reconstruction, and sentence framing."""

    def __init__(
        self,
        corrector: Optional[HuggingFaceCorrector] = None,
        auto_speak: bool = True,
        on_sentence_framed: Optional[Callable[[str], None]] = None,
        pause_timeout: float = 1.8,
        tts: Optional[Any] = None,
        hf_corrector: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.corrector = corrector or hf_corrector or HuggingFaceCorrector(enabled=True)
        self.auto_speak = auto_speak
        self.on_sentence_framed = on_sentence_framed
        self.pause_timeout = pause_timeout
        self.tts = tts

        self.words: List[str] = []
        self.last_word: Optional[str] = None
        self.last_word_time: float = 0.0
        self.last_framed_sentence: str = ""
        self.framed_history: List[str] = []

        # High-frequency grammatical mapping rules for instantaneous phrase framing
        self.PHRASE_PATTERNS = [
            (r"\b(where|where is)\s+(hospital|doctor|toilet|restroom|home)\b", r"Where is the \2?"),
            (r"\b(how|how are)\s+you\b", r"How are you?"),
            (r"\bwhat\s+(your|ur)?\s*name\b", r"What is your name?"),
            (r"\bmy\s+name\s+([a-zA-Z]+)\b", r"My name is \1."),
            (r"\bi\s+want\s+(water|food|drink|sleep|rest|help)\b", r"I want \1, please."),
            (r"\bi\s+need\s+(help|doctor|medicine|water|hospital)\b", r"I need \1."),
            (r"\bplease\s+help\s+(me)?\b", r"Please help me."),
            (r"\bthank\s+you\s+(much|very)?\b", r"Thank you very much."),
            (r"\bnice\s+(to)?\s*meet\s+you\b", r"Nice to meet you."),
            (r"\bgood\s+morning\b", r"Good morning!"),
            (r"\bgood\s+night\b", r"Good night!"),
            (r"\bi\s+love\s+you\b", r"I love you."),
            (r"\bcall\s+me\s*(later)?\b", r"Call me later."),
            (r"\bstop\s+it\b", r"Stop it."),
            (r"\bdoctor\s+come\b", r"The doctor is coming."),
            (r"\bsee\s+you\s*(later)?\b", r"See you later."),
        ]

    def add_word(self, word: str) -> str:
        """Add a recognized word to the current sentence sequence.

        Parameters
        ----------
        word : str
            Recognized sign gloss (e.g. 'HELLO', 'WATER', 'HOSPITAL').

        Returns
        -------
        str
            The current active sentence string.
        """
        cleaned = (word or "").strip().upper()
        if not cleaned or cleaned in ("NONE", "UNCERTAIN", "UNKNOWN", "MODEL_UNAVAILABLE"):
            return self.get_current_raw_phrase()

        now = time.time()
        # Suppress duplicate word if committed within 0.8s
        if cleaned == self.last_word and (now - self.last_word_time) < 0.8:
            return self.get_current_raw_phrase()

        self.last_word = cleaned
        self.last_word_time = now

        # Boundary keywords that trigger automatic sentence finalization
        if cleaned in ("STOP", "FINISH", "PERIOD"):
            if self.words:
                return self.frame_sentence(flush=True)
            return ""

        self.words.append(cleaned)
        return self.get_current_raw_phrase()

    def get_active_words(self) -> List[str]:
        """Return the list of words currently in the active sentence buffer."""
        return list(self.words)

    def get_current_raw_phrase(self) -> str:
        """Return raw space-separated words."""
        return " ".join(self.words).strip()

    def rule_based_frame(self, raw_phrase: str) -> str:
        """Instant rule-based grammar and phrasing pass."""
        text = raw_phrase.strip().lower()
        if not text:
            return ""

        # Normalize common sign glosses
        text = re.sub(r"\b(i / me|i/me)\b", "i", text)
        text = re.sub(r"\b(he / she|he/she)\b", "they", text)
        text = re.sub(r"\bhow_are_you_\b", "how are you", text)
        text = re.sub(r"\bthank_you\b", "thank you", text)
        text = re.sub(r"\bcall_me\b", "call me", text)
        text = re.sub(r"\bgood_morning\b", "good morning", text)
        text = re.sub(r"\bgood_night\b", "good night", text)

        for pattern, replacement in self.PHRASE_PATTERNS:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
                break
        else:
            # General sentence formatting
            if not text.endswith((".", "?", "!")):
                if any(text.startswith(q) for q in ("what", "where", "who", "when", "why", "how", "is", "can")):
                    text += "?"
                else:
                    text += "."

        # Capitalize first letter
        return text[0].upper() + text[1:] if text else ""

    def check_auto_frame(self) -> Optional[str]:
        """Check if pause timeout has elapsed and automatically frame sentence."""
        if not self.words:
            return None
        now = time.time()
        if (now - self.last_word_time) >= self.pause_timeout:
            return self.frame_sentence(flush=True)
        return None

    def frame_sentence(
        self,
        flush: bool = True,
        force: bool = True,
        speak: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Reconstruct words into a grammatically correct English sentence.

        Parameters
        ----------
        flush : bool
            Whether to clear the current word buffer after framing.
        force : bool
            Force framing even if below minimum word threshold.
        speak : Optional[bool]
            Whether to speak the resulting sentence via TTS.

        Returns
        -------
        str
            The final fluent English sentence.
        """
        raw = self.get_current_raw_phrase()
        if not raw:
            return ""

        # Step 1: Fast rule-based grammar pass
        rule_sentence = self.rule_based_frame(raw)

        # Step 2: Neural transformer pass if available
        final_sentence = rule_sentence
        if self.corrector and len(self.words) >= 2:
            try:
                neural_sentence = self.corrector.retransform(raw)
                if neural_sentence and len(neural_sentence.strip()) > len(raw):
                    final_sentence = neural_sentence.strip()
            except Exception:
                pass

        self.last_framed_sentence = final_sentence
        self.framed_history.append(final_sentence)

        should_speak = self.auto_speak if speak is None else speak
        if should_speak and self.tts:
            try:
                self.tts.speak_async(final_sentence)
            except Exception:
                pass

        if flush:
            self.words.clear()
            self.last_word = None

        if self.on_sentence_framed:
            self.on_sentence_framed(final_sentence)

        return final_sentence

    def backspace(self) -> str:
        """Remove the most recently added word."""
        if self.words:
            self.words.pop()
        return self.get_current_raw_phrase()

    def clear(self) -> None:
        """Clear active sentence words."""
        self.words.clear()
        self.last_word = None
