"""Multilingual translation module for localized sign language captions."""

from __future__ import annotations

from typing import Dict, Optional

# Offline phrase translations for common sign language vocabulary
TRANSLATION_DICTIONARIES: Dict[str, Dict[str, str]] = {
    "es": {  # Spanish
        "Hello!": "¡Hola!",
        "Hello, how are you?": "¡Hola, ¿cómo estás?",
        "Thank you very much.": "Muchas gracias.",
        "Thank You": "Gracias",
        "Please.": "Por favor.",
        "Yes, absolutely.": "Sí, absolutamente.",
        "Yes": "Sí",
        "No, thank you.": "No, gracias.",
        "No": "No",
        "That is good.": "Eso es bueno.",
        "Good": "Bueno",
        "Peace to you.": "Paz para ti.",
        "Peace": "Paz",
        "I love you.": "Te quiero.",
        "Love": "Amor",
        "Okay, sounds good.": "Está bien.",
        "Ok": "De acuerdo",
        "Please help me.": "Por favor, ayúdame.",
        "Help": "Ayuda",
    },
    "fr": {  # French
        "Hello!": "Bonjour!",
        "Hello, how are you?": "Bonjour, comment allez-vous?",
        "Thank you very much.": "Merci beaucoup.",
        "Thank You": "Merci",
        "Please.": "S'il vous plaît.",
        "Yes, absolutely.": "Oui, absolument.",
        "Yes": "Oui",
        "No, thank you.": "Non, merci.",
        "No": "Non",
        "That is good.": "C'est bien.",
        "Good": "Bien",
        "Peace to you.": "Paix à vous.",
        "Peace": "Paix",
        "I love you.": "Je t'aime.",
        "Love": "Amour",
        "Okay, sounds good.": "D'accord.",
        "Ok": "D'accord",
        "Please help me.": "Aidez-moi, s'il vous plaît.",
        "Help": "Aide",
    },
    "de": {  # German
        "Hello!": "Hallo!",
        "Hello, how are you?": "Hallo, wie geht es dir?",
        "Thank you very much.": "Vielen Dank.",
        "Thank You": "Danke",
        "Please.": "Bitte.",
        "Yes, absolutely.": "Ja, absolut.",
        "Yes": "Ja",
        "No, thank you.": "Nein, danke.",
        "No": "Nein",
        "That is good.": "Das ist gut.",
        "Good": "Gut",
        "Peace to you.": "Friede sei mit dir.",
        "Peace": "Frieden",
        "I love you.": "Ich liebe dich.",
        "Love": "Liebe",
        "Okay, sounds good.": "Alles klar.",
        "Ok": "In Ordnung",
        "Please help me.": "Bitte hilf mir.",
        "Help": "Hilfe",
    },
    "hi": {  # Hindi
        "Hello!": "नमस्ते!",
        "Hello, how are you?": "नमस्ते, आप कैसे हैं?",
        "Thank you very much.": "बहुत-बहुत धन्यवाद।",
        "Thank You": "धन्यवाद",
        "Please.": "कृपया।",
        "Yes, absolutely.": "हाँ, बिल्कुल।",
        "Yes": "हाँ",
        "No, thank you.": "नहीं, धन्यवाद।",
        "No": "नहीं",
        "That is good.": "यह अच्छा है।",
        "Good": "अच्छा",
        "Peace to you.": "शांति।",
        "Peace": "शांति",
        "I love you.": "मैं तुमसे प्यार करता हूँ।",
        "Love": "प्यार",
        "Okay, sounds good.": "ठीक है।",
        "Ok": "ठीक है",
        "Please help me.": "कृपया मेरी मदद करें।",
        "Help": "मदद",
    },
}


def translate_sentence(sentence: str, target_lang: str = "es") -> str:
    """Translate an English sentence into the target language.

    Parameters
    ----------
    sentence : str
        Input sentence in English.
    target_lang : str
        ISO language code (e.g. 'es', 'fr', 'de', 'hi').

    Returns
    -------
    str
        Translated sentence or original sentence if translation is unavailable.
    """
    sentence_clean = sentence.strip()
    if not sentence_clean:
        return ""

    lang_dict = TRANSLATION_DICTIONARIES.get(target_lang.lower())
    if lang_dict and sentence_clean in lang_dict:
        return lang_dict[sentence_clean]

    # Substring / partial lookup
    if lang_dict:
        for en_key, trans_val in lang_dict.items():
            if en_key.lower() == sentence_clean.lower():
                return trans_val

    return sentence_clean
