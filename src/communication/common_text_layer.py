"""Common Communication Layer — unifies outputs from ISL, ASL, and Speech recognizers.

All three input sources converge into a single text stream with timestamps
and source attribution, enabling unified display, TTS, and translation.

Architecture:
    ISL ──┐
          │
    ASL ──┼──→ COMMON TEXT LAYER → Display
          │                        → TTS
    Speech┘                        → Translation
"""

from __future__ import annotations

import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Any, Callable

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.recognition_result import RecognitionResult


@dataclass
class TextEntry:
    """A single text entry in the unified communication stream."""
    text: str
    source: str               # "ISL", "ASL", "SPEECH"
    mode: str = "WORD"        # "ALPHABET", "WORD", "SPEECH"
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Backward-compatible alias
CommonTextEntry = TextEntry


class CommonTextLayer:
    """Aggregates recognition results from all sources into a unified text stream.

    This layer provides:
    - A combined transcript from ISL, ASL, and speech recognition
    - Source-attributed entries for display
    - A single output interface for TTS and translation
    - Duplicate suppression across sources
    """

    def __init__(
        self,
        max_entries: int = 200,
        max_history: Optional[int] = None,
        dedup_window_sec: float = 2.0,
        dedup_window_seconds: Optional[float] = None,
        enable_deduplication: bool = True,
    ) -> None:
        self.max_entries = max_history if max_history is not None else max_entries
        self.dedup_window_sec = dedup_window_seconds if dedup_window_seconds is not None else dedup_window_sec
        self.enable_deduplication = enable_deduplication
        self._entries: Deque[TextEntry] = deque(maxlen=self.max_entries)
        self._lock = threading.Lock()
        self._listeners: List[Callable[[TextEntry], None]] = []

    def add_entry(
        self,
        text: str,
        source: str = "UNKNOWN",
        mode: str = "WORD",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TextEntry]:
        """Add a text entry directly to the stream."""
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        source_up = source.upper()
        mode_up = mode.upper()
        now = time.time()

        with self._lock:
            if self.enable_deduplication:
                for entry in reversed(self._entries):
                    if now - entry.timestamp > self.dedup_window_sec:
                        break
                    if entry.text == clean_text and entry.source == source_up:
                        return None

            entry = TextEntry(
                text=clean_text,
                source=source_up,
                mode=mode_up,
                confidence=confidence,
                timestamp=now,
                metadata=metadata or {},
            )
            self._entries.append(entry)

        for listener in self._listeners:
            try:
                listener(entry)
            except Exception:
                pass

        return entry

    def add(
        self,
        text: str,
        source: str = "UNKNOWN",
        mode: str = "WORD",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TextEntry]:
        """Alias for add_entry."""
        return self.add_entry(
            text=text,
            source=source,
            mode=mode,
            confidence=confidence,
            metadata=metadata,
        )

    def add_result(self, result: RecognitionResult) -> Optional[TextEntry]:
        """Add a RecognitionResult from any recognizer into the common stream."""
        if not result or not result.label:
            return None

        if result.label in ("MODEL_UNAVAILABLE", "ERROR", "NONE", "UNKNOWN", "UNCERTAIN", ""):
            return None

        return self.add_entry(
            text=result.label,
            source=result.language,
            mode=result.mode,
            confidence=result.confidence,
            metadata=dict(result.metadata) if result.metadata else {},
        )

    def add_listener(self, callback: Callable[[TextEntry], None]) -> None:
        """Register a callback that fires whenever a new entry is added."""
        self._listeners.append(callback)

    def get_transcript(self, max_entries: Optional[int] = None) -> str:
        """Return the full combined transcript as a single string."""
        with self._lock:
            entries = list(self._entries)
        if max_entries:
            entries = entries[-max_entries:]
        return " ".join(e.text for e in entries)

    def get_full_transcript(self, max_entries: Optional[int] = None) -> str:
        """Alias for get_transcript."""
        return self.get_transcript(max_entries=max_entries)

    def get_attributed_transcript(self, max_entries: Optional[int] = None) -> str:
        """Return formatted transcript with source attribution, e.g. [ISL] HELLO."""
        with self._lock:
            entries = list(self._entries)
        if max_entries:
            entries = entries[-max_entries:]
        return "\n".join(f"[{e.source}] {e.text}" for e in entries)

    def get_entries(self, max_entries: Optional[int] = None) -> List[TextEntry]:
        """Return recent entries with source attribution."""
        with self._lock:
            entries = list(self._entries)
        if max_entries:
            entries = entries[-max_entries:]
        return entries

    def get_entries_by_source(self, source: str, max_entries: Optional[int] = None) -> List[TextEntry]:
        """Return entries filtered by source (ISL, ASL, SPEECH)."""
        with self._lock:
            entries = [e for e in self._entries if e.source == source.upper()]
        if max_entries:
            entries = entries[-max_entries:]
        return entries

    def get_last_entry(self) -> Optional[TextEntry]:
        """Return the most recent entry."""
        with self._lock:
            return self._entries[-1] if self._entries else None

    def get_display_lines(self, max_lines: int = 10) -> List[Dict[str, Any]]:
        """Return recent entries formatted for UI display."""
        entries = self.get_entries(max_entries=max_lines)
        return [
            {
                "text": e.text,
                "source": e.source,
                "mode": e.mode,
                "confidence": round(e.confidence * 100, 1),
                "timestamp": e.timestamp,
            }
            for e in entries
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the current stream state for JSON APIs."""
        entries = self.get_entries()
        return {
            "total_count": len(entries),
            "full_transcript": self.get_full_transcript(),
            "entries": [
                {
                    "text": e.text,
                    "source": e.source,
                    "mode": e.mode,
                    "confidence": round(e.confidence, 3),
                    "timestamp": round(e.timestamp, 3),
                    "metadata": e.metadata,
                }
                for e in entries
            ],
        }

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  OmniSign AI: Common Communication Text Layer Demo")
    print("=" * 65)

    layer = CommonTextLayer(dedup_window_sec=2.0)

    # Register a live listener callback
    def on_new_text(entry: TextEntry):
        print(f"  [STREAM EVENT] [{entry.source}] ({entry.mode}) -> '{entry.text}' (conf: {entry.confidence:.0%})")

    layer.add_listener(on_new_text)

    print("\n1. Simulating ISL input:")
    layer.add_entry("NAMASTE", source="ISL", mode="WORD", confidence=0.94)

    print("\n2. Simulating ASL fingerspelling input:")
    layer.add_entry("H", source="ASL", mode="ALPHABET", confidence=0.98)
    layer.add_entry("I", source="ASL", mode="ALPHABET", confidence=0.95)

    print("\n3. Simulating Speech input:")
    layer.add_entry("HOW ARE YOU", source="SPEECH", mode="SPEECH", confidence=0.99)

    print("\n4. Testing duplicate suppression within 2.0s window:")
    dup = layer.add_entry("HOW ARE YOU", source="SPEECH", mode="SPEECH", confidence=0.99)
    print(f"  Duplicate entry result: {dup} (Suppressed: {dup is None})")

    print("\n" + "-" * 65)
    print("  Unified Transcript:")
    print(" ", layer.get_full_transcript())
    print("-" * 65)
    print(f"  Total verified entries: {layer.entry_count}")
    print("=" * 65 + "\n")

