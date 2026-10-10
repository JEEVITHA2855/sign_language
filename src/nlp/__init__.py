"""NLP package for sentence understanding and multilingual translation."""

from .sentence_understanding import smooth_gloss_to_sentence
from .translation import translate_sentence

__all__ = ["smooth_gloss_to_sentence", "translate_sentence"]
