from __future__ import annotations

from typing import List


class WordBuffer:
    """Canonical word-sentence buffer that accepts both words and characters."""

    def __init__(self) -> None:
        self._words: List[str] = []
        self._current_chars: List[str] = []

    def add_word(self, word: str) -> str:
        cleaned = (word or "").strip()
        if not cleaned:
            return self.get_sentence()
        self._words.append(cleaned.upper())
        return self.get_sentence()

    def add_character(self, char: str) -> str:
        cleaned = (char or "").strip().upper()
        if not cleaned:
            return self.get_sentence()
        if len(cleaned) == 1 and cleaned.isalnum():
            self._current_chars.append(cleaned)
            return self.get_sentence()
        if cleaned == "SPACE":
            self._commit_current_chars()
            return self.get_sentence()
        if cleaned in {"DEL", "DELETE", "BACKSPACE"}:
            return self.backspace()
        return self.get_sentence()

    def _commit_current_chars(self) -> None:
        if not self._current_chars:
            return
        self._words.append("".join(self._current_chars))
        self._current_chars.clear()

    def get_words(self) -> List[str]:
        words = list(self._words)
        if self._current_chars:
            words.append("".join(self._current_chars))
        return words

    def get_sentence(self) -> str:
        return " ".join(self.get_words()).strip()

    def backspace(self, allow_word_delete: bool = False) -> str:
        if self._current_chars:
            self._current_chars.pop()
            return self.get_sentence()
        if allow_word_delete and self._words:
            self._words.pop()
            return self.get_sentence()
        return ""

    def delete_last_word(self) -> str:
        if self._words:
            self._words.pop()
        return self.get_sentence()

    def clear(self) -> None:
        self._words.clear()
        self._current_chars.clear()
