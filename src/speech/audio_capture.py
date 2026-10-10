"""Microphone audio capture for real-time speech recognition.

Uses sounddevice for cross-platform microphone access with a ring buffer
that feeds audio chunks to the speech recognizer.
"""

from __future__ import annotations

import threading
import queue
import time
from typing import Optional, Callable

import numpy as np


class AudioCapture:
    """Thread-safe microphone capture with configurable sample rate and chunk size."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_sec: float = 3.0,
        silence_threshold: float = 0.01,
        silence_duration_sec: float = 1.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_sec = chunk_duration_sec
        self.silence_threshold = silence_threshold
        self.silence_duration_sec = silence_duration_sec

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        self._is_recording = False
        self._record_thread: Optional[threading.Thread] = None
        self._buffer: list[np.ndarray] = []
        self._silence_start: Optional[float] = None
        self._stream = None

    def start(self) -> None:
        """Start recording from the default microphone."""
        if self._is_recording:
            return
        self._is_recording = True
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

    def stop(self) -> None:
        """Stop recording and flush remaining audio."""
        self._is_recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=2.0)
        # Flush any remaining buffer
        if self._buffer:
            combined = np.concatenate(self._buffer, axis=0)
            try:
                self._audio_queue.put_nowait(combined)
            except queue.Full:
                pass
            self._buffer.clear()

    def get_audio_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Get the next available audio chunk (blocks up to timeout seconds)."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def _record_loop(self) -> None:
        """Main recording loop using sounddevice."""
        try:
            import sounddevice as sd
        except ImportError:
            print("[AudioCapture] sounddevice not installed. Install with: pip install sounddevice")
            self._is_recording = False
            return

        block_size = int(self.sample_rate * 0.1)  # 100ms blocks

        def _audio_callback(indata, frames, time_info, status):
            if status:
                pass  # Ignore overflow/underflow for real-time
            audio_chunk = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
            self._process_block(audio_chunk)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=block_size,
                callback=_audio_callback,
            )
            self._stream.start()

            while self._is_recording:
                time.sleep(0.05)

        except Exception as exc:
            print(f"[AudioCapture] Error: {exc}")
        finally:
            self._is_recording = False

    def _process_block(self, audio: np.ndarray) -> None:
        """Accumulate audio blocks and emit chunks on silence boundaries."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        is_silent = rms < self.silence_threshold

        self._buffer.append(audio)

        if is_silent:
            if self._silence_start is None:
                self._silence_start = time.time()
            elif time.time() - self._silence_start >= self.silence_duration_sec:
                # Silence detected — emit buffered audio
                if len(self._buffer) > 1:
                    combined = np.concatenate(self._buffer, axis=0)
                    min_samples = int(self.sample_rate * 0.5)  # At least 0.5s
                    if len(combined) >= min_samples:
                        try:
                            self._audio_queue.put_nowait(combined)
                        except queue.Full:
                            pass
                self._buffer.clear()
                self._silence_start = None
        else:
            self._silence_start = None

        # Force-emit if buffer exceeds max duration
        total_samples = sum(len(b) for b in self._buffer)
        max_samples = int(self.sample_rate * self.chunk_duration_sec * 3)
        if total_samples >= max_samples:
            combined = np.concatenate(self._buffer, axis=0)
            try:
                self._audio_queue.put_nowait(combined)
            except queue.Full:
                pass
            self._buffer.clear()
            self._silence_start = None
