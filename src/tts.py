"""Asynchronous Text-to-Speech (TTS) engine with robust Windows SAPI & pyttsx3 support."""

from __future__ import annotations

import sys
import os
import threading
import queue
import time
from typing import Optional


class TTSEngine:
    """Thread-safe background speech synthesizer with native SAPI and pyttsx3 support."""

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 165,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
        mode: str = "sentence",
    ) -> None:
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.mode = mode.lower() if mode else "sentence"  # "sentence", "word", "letter"

        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=10)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_spoken = ""
        self._last_spoken_time = 0.0
        self.is_speaking = False

        # Check native Windows SAPI voice availability
        self._use_win32_sapi = (sys.platform == "win32")

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        """Worker loop handling text synthesis safely across threads."""
        sapi_voice = None
        pyttsx_engine = None

        if self._use_win32_sapi:
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                # SAPI rate: -10 to 10 (convert 165 wpm ~ 0 or 1)
                sapi_rate = int((self.rate - 150) / 15)
                sapi_voice.Rate = max(-10, min(10, sapi_rate))
                sapi_voice.Volume = int(self.volume * 100)
            except Exception as e:
                print(f"[TTS Warning] SAPI initialization failed: {e}. Falling back to pyttsx3.")
                sapi_voice = None

        if sapi_voice is None:
            try:
                import pyttsx3
                pyttsx_engine = pyttsx3.init()
                pyttsx_engine.setProperty("rate", self.rate)
                pyttsx_engine.setProperty("volume", self.volume)
                if self.voice_id:
                    pyttsx_engine.setProperty("voice", self.voice_id)
            except Exception as e:
                print(f"[TTS Error] pyttsx3 init failed: {e}")
                pyttsx_engine = None

        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:  # Exit sentinel
                break

            if self.enabled and text.strip():
                self.is_speaking = True
                try:
                    if sapi_voice is not None:
                        # 0 = Synchronous within this background worker thread, so queue stays ordered
                        sapi_voice.Speak(text, 0)
                    elif pyttsx_engine is not None:
                        pyttsx_engine.say(text)
                        pyttsx_engine.runAndWait()
                except Exception as ex:
                    print(f"[TTS Error] Playback exception: {ex}")
                finally:
                    self.is_speaking = False

            self._queue.task_done()

        if self._use_win32_sapi:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

        if pyttsx_engine:
            try:
                pyttsx_engine.stop()
            except Exception:
                pass

    def speak_async(self, text: str, min_interval_sec: float = 0.8) -> None:
        """Queue text to speak in the background without blocking the camera loop.

        Parameters
        ----------
        text : str
            Utterance to speak.
        min_interval_sec : float
            Minimum cooldown before speaking the exact same phrase again.
        """
        if not self.enabled or not text.strip():
            return

        now = time.time()
        if text == self._last_spoken and (now - self._last_spoken_time) < min_interval_sec:
            return

        self._last_spoken = text
        self._last_spoken_time = now

        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def cycle_mode(self) -> str:
        """Cycle between speech modes: sentence -> word -> letter."""
        modes = ["sentence", "word", "letter"]
        idx = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(idx + 1) % len(modes)]
        return self.mode

    def speak_token(self, token: str, is_completed_word: bool = False, is_sentence: bool = False) -> None:
        """Speak according to active TTS mode (avoids letter-by-letter spam in sentence mode)."""
        if not self.enabled or not token.strip():
            return

        if self.mode == "sentence" and is_sentence:
            self.speak_async(token)
        elif self.mode == "word" and (is_completed_word or is_sentence):
            self.speak_async(token)
        elif self.mode == "letter":
            self.speak_async(token)

    def stop(self) -> None:
        """Stop background worker."""
        self._running = False
        if self._thread and self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
                self._thread.join(timeout=1.0)
            except Exception:
                pass
