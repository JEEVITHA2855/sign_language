"""Gemini AI Sign Language Re-transformation and Predictive Translation Module.

Transforms raw, noisy fingerspelling streams (e.g. 'je evi th a  i s gi rl') into
clean, fluent, grammatically correct sentences (e.g. 'Jeevitha is a girl.').
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Optional, Callable


GEMINI_SYSTEM_INSTRUCTION = """You are an AI assistant specialized in Sign Language Fingerspelling Reconstruction.
Users spell out words using an ASL gesture camera. Because recognition captures poses frame-by-frame, the raw output has broken syllables, fragmented words, missing articles, or phonetic spelling.

Examples:
- "je evi th a  i s gi rl" -> "Jeevitha is a girl."
- "h e l l o  h o w  r  u" -> "Hello, how are you?"
- "m y  n a m  j e e v i t h a" -> "My name is Jeevitha."
- "n i c e  t o  m e e t  u" -> "Nice to meet you."
- "t h a n k  u  v e r y  m u c h" -> "Thank you very much."
- "w h a t  u r  n a m e" -> "What is your name?"
- "i  l i k e  c o d i n g" -> "I like coding."
- "w h e r e  i s  h o s p i t a l" -> "Where is the hospital?"

Task:
Reconstruct the raw input into a natural, grammatically correct, fluent English sentence.
- Fix broken word fragments (e.g. 'je evi th a' -> 'Jeevitha', 'gi rl' -> 'girl', 'i s' -> 'is').
- Preserve proper nouns and names accurately.
- Add appropriate capitalization and punctuation (. or ?).
- Output ONLY the final reconstructed sentence. Do NOT include explanations, markdown, or quotation marks.
"""


def local_heuristic_retransform(raw_text: str) -> str:
    """Intelligent offline fallback that cleans and merges fragmented fingerspelling tokens."""
    if not raw_text or not raw_text.strip():
        return ""

    # Normalize and strip tokens
    raw_tokens = raw_text.strip().split()
    if not raw_tokens:
        return ""

    tokens = [t.strip() for t in raw_tokens if t.strip().upper() not in ("NONE", "DEL")]
    if not tokens:
        return ""

    # Check for explicit SPACE tokens
    # Group tokens by SPACE
    chunks: list[list[str]] = []
    curr_chunk: list[str] = []
    for t in tokens:
        if t.upper() == "SPACE":
            if curr_chunk:
                chunks.append(curr_chunk)
                curr_chunk = []
        else:
            curr_chunk.append(t)
    if curr_chunk:
        chunks.append(curr_chunk)

    processed_words: list[str] = []

    # Common English words dictionary for recognition
    COMMON_WORDS = {
        "a", "i", "is", "in", "it", "to", "of", "on", "at", "my", "we", "he", "she", "am", "me", "so", "no", "go",
        "girl", "boy", "name", "hello", "good", "love", "peace", "help", "please", "thank", "you", "are",
        "water", "food", "sign", "language", "friend", "happy", "yes", "nice", "meet", "who", "what",
        "where", "why", "how", "fine", "see", "later", "welcome", "student", "teacher", "this", "that"
    }

    for chunk in chunks:
        # If all items in chunk are single letters, join them as a single word
        if all(len(item) == 1 for item in chunk):
            word = "".join(chunk).lower()
            processed_words.append(word)
        else:
            # Chunk has multi-letter items: e.g. ["Je", "Evi", "Th", "A"] or ["Hello", "World"]
            # Check if joining them forms a known word or plausible name
            joined = "".join(chunk).lower()
            if joined in COMMON_WORDS or len(chunk) > 2:
                processed_words.append(joined)
            else:
                for item in chunk:
                    processed_words.append(item.lower())

    sentence = " ".join(processed_words)

    # Contextual grammatical substitutions
    sentence = re.sub(r"\bje\s*evi\s*th\s*a\b", "jeevitha", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bgi\s*rl\b", "girl", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bi\s*s\b", "is", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bis girl\b", "is a girl", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bis boy\b", "is a boy", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bmy nam\b", "my name", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bmy name\b(?!\s+is\b)", "my name is", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bhow u\b", "how are you", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\bu\b", "you", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\br\b", "are", sentence, flags=re.IGNORECASE)

    # Format sentence casing and terminal punctuation
    sentence = sentence.strip()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            if any(sentence.lower().startswith(q) for q in ["what", "how", "where", "why", "who", "when", "is", "can"]):
                sentence += "?"
            else:
                sentence += "."

    return sentence


class GeminiCorrector:
    """Predicts and retransforms raw sign language tokens into fluent sentences using Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.api_key = (api_key or "").strip()
        self._load_env_file()
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        self.client = None
        self.is_transforming = False
        self.last_raw: str = ""
        self.last_retransformed: str = ""
        self.last_source: str = "OFFLINE"  # "GEMINI" or "OFFLINE"

        if self.enabled and self.api_key:
            self._init_client()

    def _load_env_file(self) -> None:
        """Auto-load GEMINI_API_KEY from .env file if present."""
        project_root = Path(__file__).resolve().parent.parent.parent
        for env_path in [Path(".env"), project_root / ".env"]:
            if env_path.exists():
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip("'\"")
                                if k == "GEMINI_API_KEY" and v:
                                    os.environ["GEMINI_API_KEY"] = v
                                    if not self.api_key:
                                        self.api_key = v
                except Exception:
                    pass

    def _init_client(self) -> None:
        """Initialize Google Generative AI client safely across legacy and modern SDKs."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            if hasattr(genai, "GenerativeModel"):
                self.client = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=GEMINI_SYSTEM_INSTRUCTION,
                )
                self.client_type = "modern"
                return

            if hasattr(genai, "generate_text"):
                self.client = genai
                self.client_type = "legacy"
                return

            self.client = None
            self.client_type = "unknown"
        except Exception as e:
            print(f"[Gemini Warning] Could not initialize google.generativeai: {e}")
            self.client = None
            self.client_type = "unknown"

    @staticmethod
    def _extract_response_text(response) -> str:
        """Normalize Gemini response objects from both SDK versions into plain text."""
        if response is None:
            return ""

        if isinstance(response, str):
            return response.strip().strip('"\'')

        for attr in ("text", "output_text", "result", "generated_text"):
            value = getattr(response, attr, None)
            if value is not None:
                text = str(value).strip().strip('"\'')
                if text:
                    return text

        if isinstance(response, dict):
            for key in ("text", "output_text", "result", "generated_text"):
                value = response.get(key)
                if value is not None:
                    text = str(value).strip().strip('"\'')
                    if text:
                        return text

        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                candidate_text = GeminiCorrector._extract_response_text(candidate)
                if candidate_text:
                    return candidate_text

        content = getattr(response, "content", None)
        if content is not None:
            parts = getattr(content, "parts", None)
            if parts:
                for part in parts:
                    part_text = GeminiCorrector._extract_response_text(part)
                    if part_text:
                        return part_text

        return ""

    def set_api_key(self, api_key: str) -> None:
        """Update API key dynamically at runtime."""
        self.api_key = api_key.strip()
        if self.api_key:
            self._init_client()

    def retransform(self, raw_text: str) -> str:
        """Synchronously retransform raw sign tokens into a fluent sentence."""
        raw_text = raw_text.strip()
        if not raw_text:
            return ""

        self.last_raw = raw_text

        # Try Gemini API if available
        if self.enabled and self.client is not None:
            try:
                if getattr(self, "client_type", "unknown") == "modern":
                    response = self.client.generate_content(
                        f"Reconstruct this sign language stream into a natural English sentence:\n{raw_text}"
                    )
                elif getattr(self, "client_type", "unknown") == "legacy":
                    legacy_model = self.model_name if self.model_name.startswith("models/") else "models/text-bison-001"
                    response = self.client.generate_text(
                        model=legacy_model,
                        prompt=f"Reconstruct this sign language stream into a natural English sentence:\n{raw_text}",
                    )
                else:
                    response = None

                cleaned_result = self._extract_response_text(response)
                if cleaned_result:
                    self.last_retransformed = cleaned_result
                    self.last_source = "GEMINI"
                    return cleaned_result
            except Exception as e:
                print(f"[Gemini API Error] {e}. Falling back to local heuristic.")

        # Fallback to local heuristic
        result = local_heuristic_retransform(raw_text)
        self.last_retransformed = result
        self.last_source = "OFFLINE"
        return result

    def retransform_async(
        self,
        raw_text: str,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Run retransformation in a background daemon thread so camera UI never freezes."""
        if not raw_text or not raw_text.strip() or self.is_transforming:
            return

        self.is_transforming = True

        def _worker():
            try:
                res = self.retransform(raw_text)
                if callback:
                    callback(res, self.last_source)
            finally:
                self.is_transforming = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
