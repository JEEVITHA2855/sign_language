"""Build and manage current words from confirmed alphabet characters."""

from __future__ import annotations


class WordBuilder:
    def __init__(self) -> None:
        self.current = ""

    def append_char(self, char: str) -> str:
        cleaned = (char or "").strip().upper()
        if cleaned and len(cleaned) == 1 and cleaned.isalnum():
            self.current += cleaned
        return self.current

    def commit(self) -> str:
        word = self.current
        self.current = ""
        return word

    def clear(self) -> None:
        self.current = ""

    def backspace(self) -> str:
        if self.current:
            self.current = self.current[:-1]
        return self.current
