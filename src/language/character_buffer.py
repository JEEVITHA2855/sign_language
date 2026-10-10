"""Stateful buffer for current word and sentence composition."""

from __future__ import annotations


class CharacterBuffer:
    """Maintain the in-flight word and committed sentence while recognizing symbols."""

    def __init__(self) -> None:
        self.current_word = ""
        self.sentence = ""

    def append_char(self, char: str) -> str:
        cleaned = (char or "").strip().upper()
        if not cleaned:
            return self.current_word
        if cleaned == "SPACE":
            return self.handle_space()
        if len(cleaned) == 1 and (cleaned.isalpha() or cleaned.isdigit()):
            self.current_word += cleaned
        return self.current_word

    def handle_space(self) -> str:
        if self.current_word:
            committed = self.current_word
            self.sentence += committed + " "
            self.current_word = ""
            return committed
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
        return self.sentence

    def backspace(self) -> str:
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence = self.sentence.rstrip()
            if " " in self.sentence:
                self.sentence = self.sentence.rsplit(" ", 1)[0] + " "
            else:
                self.sentence = ""
        return self.current_word

    def delete_word(self) -> str:
        self.current_word = ""
        self.sentence = self.sentence.rstrip()
        if " " in self.sentence:
            self.sentence = self.sentence.rsplit(" ", 1)[0] + " "
        else:
            self.sentence = ""
        return self.current_word

    def clear(self) -> None:
        self.current_word = ""
        self.sentence = ""
