from src.recognition.word_buffer import WordBuffer


def test_word_buffer_adds_words_and_characters():
    buffer = WordBuffer()
    buffer.add_word("HELLO")
    buffer.add_character("J")
    buffer.add_character("E")
    buffer.add_character("E")
    buffer.add_character("V")
    assert buffer.get_words() == ["HELLO", "JEEV"]
    assert buffer.get_sentence() == "HELLO JEEV"


def test_word_buffer_backspace_and_clear():
    buffer = WordBuffer()
    buffer.add_word("HELLO")
    assert buffer.backspace() == ""
    buffer.add_character("W")
    buffer.add_character("A")
    buffer.add_character("T")
    buffer.add_character("E")
    buffer.add_character("R")
    assert buffer.get_sentence() == "HELLO WATER"
    buffer.clear()
    assert buffer.get_words() == []
    assert buffer.get_sentence() == ""
