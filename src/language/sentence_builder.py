"""Sentence buffer for composing words into a typed phrase."""

from __future__ import annotations

from typing import List


class SentenceBuilder:
    def __init__(self) -> None:
        self.words: List[str] = []

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def add_word(self, word: str) -> str:
        cleaned = (word or "").strip()
        if not cleaned:
            return self.text
        self.words.append(cleaned)
        return self.text

    def backspace(self) -> str:
        if not self.words:
            return ""
        self.words.pop()
        return self.text

    def delete_word(self) -> str:
        if not self.words:
            return ""
        self.words.pop()
        return self.text

    def clear(self) -> str:
        self.words.clear()
        return self.text
