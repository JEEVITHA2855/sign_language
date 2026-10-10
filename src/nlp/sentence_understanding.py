"""Gloss-to-fluent-sentence smoothing and grammatical reconstruction."""

from __future__ import annotations

from typing import List, Dict

# Common conversational phrase mappings from sign glosses to fluent English
GLOSS_PHRASE_TEMPLATES: Dict[str, str] = {
    "HELLO": "Hello!",
    "HELLO HOW YOU": "Hello, how are you?",
    "THANK YOU": "Thank you very much.",
    "PLEASE": "Please.",
    "YES": "Yes, absolutely.",
    "NO": "No, thank you.",
    "GOOD": "That is good.",
    "PEACE": "Peace to you.",
    "LOVE": "I love you.",
    "OK": "Okay, sounds good.",
    "HELP": "Please help me.",
    "NICE MEET YOU": "Nice to meet you.",
    "HOW YOU": "How are you doing?",
    "I NEED HELP": "I need assistance.",
}


def smooth_gloss_to_sentence(gloss_tokens: List[str]) -> str:
    """Transform a sequence of raw sign glosses into a grammatically natural sentence.

    Parameters
    ----------
    gloss_tokens : List[str]
        List of recognized sign tokens, e.g. ["HELLO", "THANK_YOU"].

    Returns
    -------
    str
        Grammatically coherent sentence string.
    """
    if not gloss_tokens:
        return ""

    # Normalize tokens to uppercase space-separated key
    cleaned_tokens = [t.strip().upper().replace("_", " ") for t in gloss_tokens if t.strip()]
    raw_key = " ".join(cleaned_tokens)

    # 1. Direct template matching
    if raw_key in GLOSS_PHRASE_TEMPLATES:
        return GLOSS_PHRASE_TEMPLATES[raw_key]

    # 2. Rule-based gloss smoothing
    words = []
    for token in cleaned_tokens:
        if token == "THANK YOU":
            words.append("thank you")
        elif token == "HELLO":
            words.append("hello")
        else:
            words.append(token.lower())

    sentence = " ".join(words)
    if not sentence:
        return ""

    # Capitalize first letter
    sentence = sentence[0].upper() + sentence[1:]

    # Add terminal punctuation if absent
    if sentence[-1] not in ".!?":
        if any(q in sentence.lower() for q in ["how", "what", "where", "why", "who"]):
            sentence += "?"
        else:
            sentence += "."

    return sentence
