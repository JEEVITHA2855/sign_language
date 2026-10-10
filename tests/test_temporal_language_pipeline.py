from src.recognition.confidence_filter import ConfidenceFilter
from src.recognition.temporal_smoother import TemporalSmoother
from src.language.character_buffer import CharacterBuffer
from src.language.word_builder import WordBuilder
from src.language.sentence_builder import SentenceBuilder
from src.language.fuzzy_matcher import FuzzyMatcher


def test_confidence_filter_rejects_below_threshold():
    filter_ = ConfidenceFilter(threshold=0.80)
    assert filter_.accepts(0.95) is True
    assert filter_.accepts(0.75) is False


def test_stable_prediction_uses_window_majority():
    smoother = TemporalSmoother(
        confidence_threshold=0.80,
        stable_frame_count=5,
        prediction_window_size=5,
        cooldown_frames=3,
    )

    for _ in range(4):
        assert smoother.update("H", 0.92) is None

    assert smoother.update("H", 0.92) == "H"
    assert smoother.update("H", 0.92) is None


def test_duplicate_suppression_keeps_single_letter():
    smoother = TemporalSmoother(
        confidence_threshold=0.80,
        stable_frame_count=5,
        prediction_window_size=5,
        cooldown_frames=4,
    )

    for _ in range(4):
        assert smoother.update("H", 0.92) is None
    assert smoother.update("H", 0.92) == "H"

    smoother.reset_after_change()
    for _ in range(4):
        assert smoother.update("H", 0.92) is None
    assert smoother.update("H", 0.92) == "H"


def test_space_handling_commits_word():
    buffer = CharacterBuffer()
    for ch in ["H", "E", "L", "L", "O"]:
        buffer.append_char(ch)

    committed = buffer.handle_space()
    assert committed == "HELLO"
    assert buffer.current_word == ""
    assert buffer.sentence == "HELLO "


def test_repeated_letters_after_rearm_are_allowed():
    smoother = TemporalSmoother(
        confidence_threshold=0.80,
        stable_frame_count=5,
        prediction_window_size=5,
        cooldown_frames=2,
    )

    for _ in range(4):
        assert smoother.update("L", 0.92) is None
    assert smoother.update("L", 0.92) == "L"

    smoother.reset_after_change()
    for _ in range(4):
        assert smoother.update("L", 0.92) is None
    assert smoother.update("L", 0.92) == "L"


def test_word_creation_and_sentence_creation():
    sentence = SentenceBuilder()
    word = WordBuilder()

    for ch in ["H", "E", "L", "L", "O"]:
        word.append_char(ch)
    sentence.add_word(word.commit())

    word.clear()
    for ch in ["W", "O", "R", "L", "D"]:
        word.append_char(ch)
    sentence.add_word(word.commit())

    assert sentence.text == "HELLO WORLD"


def test_backspace_delete_word_and_clear_sentence():
    sentence = SentenceBuilder()
    sentence.add_word("HELLO")
    sentence.add_word("WORLD")

    assert sentence.backspace() == "HELLO"
    assert sentence.delete_word() == ""
    assert sentence.clear() == ""


def test_fuzzy_suggestions_rank_expected_matches():
    suggestions = FuzzyMatcher(["HELLO", "HELP", "HELD"]).suggest("HELO")
    assert suggestions[0] == "HELLO"
    assert "HELLO" in suggestions
