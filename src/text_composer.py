"""Sentence builder and gloss-to-text composer."""

from __future__ import annotations

from typing import List, Optional


class TextComposer:
    """Combines confirmed sign tokens into formatted sentences and phrases."""

    def __init__(self, no_hand_timeout_frames: int = 25, auto_space: bool = True) -> None:
        self.no_hand_timeout_frames = no_hand_timeout_frames
        self.auto_space = auto_space

        self.tokens: List[str] = []
        self.current_word_chars: List[str] = []
        self.raw_history: List[str] = []
        self.consecutive_no_hand_frames: int = 0
        self.completed_sentences: List[str] = []
        self.retransformed_sentence: Optional[str] = None

    def add_token(self, token: str) -> str:
        """Add a recognized token (either a letter or a full word sign).

        Parameters
        ----------
        token : str
            Recognized sign symbol (e.g. 'A', 'B', 'SPACE', 'DEL', 'HELLO').

        Returns
        -------
        str
            The updated current sentence string.
        """
        self.consecutive_no_hand_frames = 0
        cleaned = token.strip().upper()
        self.raw_history.append(cleaned)

        if cleaned == "SPACE":
            # Real space action: finish current word
            if self.current_word_chars:
                self.tokens.append("".join(self.current_word_chars))
                self.current_word_chars.clear()
            return self.get_current_sentence()

        elif cleaned in ("DEL", "DELETE", "BACKSPACE"):
            # Real delete / backspace action
            return self.backspace()

        elif len(cleaned) == 1 and cleaned.isalpha():
            # It's an alphabet letter: accumulate into current word
            self.current_word_chars.append(cleaned)
        else:
            # It's a full word sign (e.g. HELLO, THANK_YOU)
            if self.current_word_chars:
                self.tokens.append("".join(self.current_word_chars))
                self.current_word_chars.clear()

            # Clean formatting for compound words
            formatted_word = cleaned.replace("_", " ").title()
            self.tokens.append(formatted_word)

        return self.get_current_sentence()

    def mark_no_hand(self) -> Optional[str]:
        """Record a frame where no hand was visible.

        Returns
        -------
        Optional[str]
            Completed word or phrase if boundary threshold was reached.
        """
        self.consecutive_no_hand_frames += 1

        if self.consecutive_no_hand_frames == self.no_hand_timeout_frames:
            # Flush any pending spelled letters into a completed word
            if self.current_word_chars:
                completed = "".join(self.current_word_chars)
                self.tokens.append(completed)
                self.current_word_chars.clear()
                return completed

        return None

    def get_current_word(self) -> str:
        """Return the word currently being spelled, or the last completed word if chars empty."""
        if self.current_word_chars:
            return "".join(self.current_word_chars)
        if self.tokens:
            return self.tokens[-1]
        return ""

    def get_last_completed_word(self) -> Optional[str]:
        """Return the most recently finalized word from tokens list."""
        if self.tokens:
            return self.tokens[-1]
        return None

    def get_raw_stream(self) -> str:
        """Return the raw sequence of recognized symbols for AI re-transformation."""
        parts = list(self.tokens)
        if self.current_word_chars:
            parts.append("".join(self.current_word_chars))
        if parts:
            return " SPACE ".join(parts)
        return " ".join(self.raw_history)

    def set_retransformed_sentence(self, sentence: str) -> None:
        """Store the AI-retransformed / predicted fluent sentence."""
        self.retransformed_sentence = sentence.strip()

    def get_display_sentence(self) -> str:
        """Return the best sentence to display (retransformed if available, else raw composed)."""
        if self.retransformed_sentence:
            return self.retransformed_sentence
        return self.get_current_sentence()

    def get_current_sentence(self) -> str:
        """Return the running composed text as a readable string."""
        parts = list(self.tokens)
        if self.current_word_chars:
            parts.append("".join(self.current_word_chars))

        if not parts:
            return ""

        raw_text = " ".join(parts)
        # Sentence casing
        return raw_text[0].upper() + raw_text[1:] if raw_text else ""

    def commit_sentence(self) -> str:
        """Finalize the current sentence and start a fresh line."""
        sentence = self.get_display_sentence()
        if sentence:
            self.completed_sentences.append(sentence)
        self.tokens.clear()
        self.current_word_chars.clear()
        self.raw_history.clear()
        self.retransformed_sentence = None
        self.consecutive_no_hand_frames = 0
        return sentence

    def backspace(self) -> str:
        """Remove the most recently added letter or word."""
        self.retransformed_sentence = None
        if self.current_word_chars:
            self.current_word_chars.pop()
        elif self.tokens:
            self.tokens.pop()
        if self.raw_history:
            self.raw_history.pop()
        return self.get_current_sentence()

    def clear(self) -> None:
        """Clear all active composed text."""
        self.tokens.clear()
        self.current_word_chars.clear()
        self.raw_history.clear()
        self.retransformed_sentence = None
        self.consecutive_no_hand_frames = 0
