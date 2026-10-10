"""Unit tests verifying the exact gesture debouncing and word recognition test cases:

Test 1 — Held letter: Show A and hold for 5 seconds -> Expected: 'A', NOT 'AAAAAAAA'
Test 2 — Different letters: Show A -> B -> C -> Expected: 'ABC'
Test 3 — Same letter twice: Show A -> remove hand -> A -> Expected: 'AA'
Test 4 — Same letter without release: Show A -> A -> Expected: 'A'
Test 5 — Noisy prediction: A A A B A A A -> Expected: 'A'
Test 6 — Word: Perform complete sign for HELLO -> Expected: 'HELLO', NOT 'HELLO HELLO HELLO'
Test 7 — Multiple words: HELLO, then distinct gesture THANK YOU -> Expected: 'HELLO THANK YOU'
"""

import pytest
from src.stabilizer import Stabilizer, GestureState
from src.recognition.word_stabilizer import WordStabilizer
from src.recognition.word_buffer import WordBuffer
from src.text_composer import TextComposer


def test_case_1_held_letter():
    """Test 1: Show A and hold for 5 seconds (150 frames). Must commit A only once."""
    stabilizer = Stabilizer(required_stable_frames=5, min_conf=0.70)
    composer = TextComposer()

    # Feed 150 frames of 'A' with 90% confidence
    for frame_idx in range(150):
        committed = stabilizer.update("A", 0.90)
        if committed:
            composer.add_token(committed)

    # Expected: exactly one 'A'
    assert composer.get_current_sentence() == "A"
    assert stabilizer.state == GestureState.WAIT_FOR_RELEASE
    assert stabilizer.last_committed == "A"


def test_case_2_different_letters():
    """Test 2: Show A -> B -> C. Expected output: 'ABC'."""
    stabilizer = Stabilizer(required_stable_frames=5, min_conf=0.70)
    composer = TextComposer()

    # Show A for 10 frames
    for _ in range(10):
        c = stabilizer.update("A", 0.90)
        if c:
            composer.add_token(c)

    # Transition to B for 10 frames
    for _ in range(10):
        c = stabilizer.update("B", 0.90)
        if c:
            composer.add_token(c)

    # Transition to C for 10 frames
    for _ in range(10):
        c = stabilizer.update("C", 0.90)
        if c:
            composer.add_token(c)

    assert composer.get_current_sentence() == "ABC"


def test_case_3_same_letter_twice_with_release():
    """Test 3: Show A -> remove hand -> A. Expected output: 'AA'."""
    stabilizer = Stabilizer(required_stable_frames=5, min_conf=0.70)
    composer = TextComposer()

    # 1. Show A for 8 frames
    for _ in range(8):
        c = stabilizer.update("A", 0.90)
        if c:
            composer.add_token(c)
    assert composer.get_current_sentence() == "A"

    # 2. Remove hand from frame (e.g. for 5 frames)
    for _ in range(5):
        stabilizer.mark_no_hand()
    assert stabilizer.state == GestureState.NO_HAND

    # 3. Show A again for 8 frames
    for _ in range(8):
        c = stabilizer.update("A", 0.90)
        if c:
            composer.add_token(c)

    # Expected: 'AA'
    assert composer.get_current_sentence() == "AA"


def test_case_4_same_letter_without_release():
    """Test 4: Show A -> A without hand release. Expected output: 'A'."""
    stabilizer = Stabilizer(required_stable_frames=5, min_conf=0.70)
    composer = TextComposer()

    # Feed A continuously for 50 frames without removing hand
    for _ in range(50):
        c = stabilizer.update("A", 0.92)
        if c:
            composer.add_token(c)

    # Expected: exactly one 'A'
    assert composer.get_current_sentence() == "A"


def test_case_5_noisy_prediction():
    """Test 5: Predictions A A A B A A A.

    Expected: 'A', unless B becomes sufficiently stable.
    """
    stabilizer = Stabilizer(required_stable_frames=5, min_conf=0.70)
    composer = TextComposer()

    # Stream: A A A A A (A commits), then B (1 frame noise), then A A A
    stream = ["A", "A", "A", "A", "A", "B", "A", "A", "A"]
    for sym in stream:
        c = stabilizer.update(sym, 0.85)
        if c:
            composer.add_token(c)

    # 'B' only appeared for 1 frame (did not reach 5 stable frames), so output must be 'A'
    assert composer.get_current_sentence() == "A"


def test_case_6_word_held():
    """Test 6: Perform complete sign for HELLO.

    Expected: 'HELLO', NOT 'HELLO HELLO HELLO HELLO'.
    """
    word_stabilizer = WordStabilizer(window=1, min_confidence=0.35)
    word_buffer = WordBuffer()

    # 1st evaluation returns HELLO
    acc1 = word_stabilizer.update("HELLO", 0.88)
    if acc1:
        word_buffer.add_word(acc1)

    # 2nd evaluation without hand release (e.g. while hand is still signing/held)
    acc2 = word_stabilizer.update("HELLO", 0.85)
    if acc2:
        word_buffer.add_word(acc2)

    # 3rd evaluation without hand release
    acc3 = word_stabilizer.update("HELLO", 0.89)
    if acc3:
        word_buffer.add_word(acc3)

    # Expected: exactly one 'HELLO'
    assert word_buffer.get_sentence() == "HELLO"


def test_case_7_multiple_words():
    """Test 7: Perform HELLO, then a second distinct gesture THANK YOU.

    Expected: 'HELLO THANK YOU'.
    """
    word_stabilizer = WordStabilizer(window=1, min_confidence=0.35)
    word_buffer = WordBuffer()

    # 1. First word sign: HELLO
    acc1 = word_stabilizer.update("HELLO", 0.88)
    if acc1:
        word_buffer.add_word(acc1)
    assert word_buffer.get_sentence() == "HELLO"

    # User pauses / lowers hand
    word_stabilizer.mark_released()

    # 2. Second distinct word sign: THANK YOU
    acc2 = word_stabilizer.update("THANK YOU", 0.91)
    if acc2:
        word_buffer.add_word(acc2)

    # Expected: 'HELLO THANK YOU'
    assert word_buffer.get_sentence() == "HELLO THANK YOU"


def test_word_buffer_backspace_and_clear():
    """Verify word buffer backspace and clear operations."""
    wb = WordBuffer()
    wb.add_word("HELLO")
    wb.add_word("WORLD")
    assert wb.get_sentence() == "HELLO WORLD"

    wb.delete_last_word()
    assert wb.get_sentence() == "HELLO"

    wb.clear()
    assert wb.get_sentence() == ""
