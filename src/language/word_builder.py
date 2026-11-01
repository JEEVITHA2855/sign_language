"""Build and manage current words from confirmed alphabet characters with 100+ words fuzzy matching."""

from __future__ import annotations

from typing import List, Optional
from src.recognition.expanded_sign_detector import ExpandedSignDetector
from src.language.fuzzy_matcher import FuzzyMatcher


class WordBuilder:
    def __init__(self, vocabulary: Optional[List[str]] = None) -> None:
        self.current = ""
        vocab = vocabulary or ExpandedSignDetector.ALL_WORDS
        self.matcher = FuzzyMatcher(vocab)

    def append_char(self, char: str) -> str:
        cleaned = (char or "").strip().upper()
        if cleaned and len(cleaned) == 1 and cleaned.isalnum():
            self.current += cleaned
        return self.current

    def add_letter(self, char: str) -> str:
        """Alias for append_char."""
        return self.append_char(char)

    def get_suggestions(self, k: int = 3) -> List[str]:
        """Return the top-k matching full words from the 100+ vocabulary."""
        if len(self.current) >= 2:
            return self.matcher.suggest(self.current, limit=k, min_similarity=0.4)
        return []

    def get_suggestion(self) -> Optional[str]:
        """Return the best matching full word from the 100+ vocabulary."""
        if len(self.current) >= 2:
            matches = self.matcher.suggest(self.current, limit=1, min_similarity=0.6)
            if matches:
                return matches[0]
        return None

    def commit(self, auto_complete: bool = False) -> str:
        """Commit the current word, optionally autocompleting with best fuzzy match."""
        if auto_complete and len(self.current) >= 3:
            sugg = self.get_suggestion()
            if sugg:
                self.current = ""
                return sugg

        word = self.current
        self.current = ""
        return word

    def clear(self) -> None:
        self.current = ""

    def backspace(self) -> str:
        if self.current:
            self.current = self.current[:-1]
        return self.current
