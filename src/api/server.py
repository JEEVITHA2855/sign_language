"""Web API server for the Sign Language Recognition and Speech-to-Text System.

Provides REST endpoints for:
- Frame-based sign recognition (ISL / ASL, Alphabet / Word)
- Speech-to-text audio transcription (Whisper)
- Common text layer aggregation (unified multi-modal transcript)
- TTS speech synthesis
- System configuration and status reporting

Supports both FastAPI/uvicorn (when installed) and a built-in zero-dependency
HTTP server fallback so the application runs in any environment.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.asl_recognizer import ASLRecognizer
from src.recognition.isl_recognizer import ISLRecognizer
from src.stabilizer import Stabilizer
from src.recognition.word_stabilizer import WordStabilizer
from src.speech.speech_recognizer import SpeechRecognizer
from src.communication.common_text_layer import CommonTextLayer
from src.tts import TTSEngine
from src.classifier import GestureClassifier
from src.preprocessing import normalize_landmarks
from src.recognition.video_buffer import VideoBuffer
from src.recognition.basic_sign_detector import BasicSignDetector


class SignLanguageAPIService:
    """Core state and inference coordinator for the Web API."""

    def __init__(self, frontend_dir: Optional[str] = None) -> None:
        self.frontend_dir = Path(frontend_dir or (PROJECT_ROOT / "frontend"))
        self.common_layer = CommonTextLayer(max_history=100)

        # Configuration defaults
        self.language = "ASL"  # "ASL" or "ISL"
        self.mode = "ALPHABET"  # "ALPHABET" or "WORD"
        self.isl_threshold = 0.55
        self.asl_threshold = 0.50
        self.tts_enabled = True

        # Initialize recognizers
        print("[APIService] Initializing recognizers...")
        self.isl_recognizer = ISLRecognizer(
            mode=self.mode,
            confidence_threshold=self.isl_threshold,
        )
        self.asl_recognizer = ASLRecognizer(
            model_path="models/asl/pytorch_model.bin",
            labels_path="models/asl/labels.json",
        )
        self.speech_recognizer = SpeechRecognizer(
            model_name="openai/whisper-base",
            language="en",
        )

        # Load MediaPipe Landmark Extractor & Gesture Classifier for ASL Alphabet
        self.extractor = None
        try:
            from src.landmark_extractor import LandmarkExtractor
            self.extractor = LandmarkExtractor(model_path="models/hand_landmarker.task")
        except Exception as exc:
            print(f"[APIService] LandmarkExtractor init notice: {exc}")

        self.labels = [chr(i) for i in range(ord("A"), ord("Z") + 1)] + ["SPACE", "DEL"]
        self.asl_classifier = None
        try:
            self.asl_classifier = GestureClassifier(
                model_path="models/gesture_classifier.onnx",
                labels=self.labels,
                sklearn_fallback_path="models/gesture_classifier.joblib",
            )
        except Exception as exc:
            print(f"[APIService] GestureClassifier init notice: {exc}")

        # Temporal video buffer for dynamic word signs
        self.video_buffer = VideoBuffer(max_frames=32, sequence_length=32)

        # State machine stabilizers for temporal smoothing
        self.alphabet_stabilizer = Stabilizer(
            window=5,
            required_stable_frames=5,
            min_conf=0.70,
        )
        self.word_stabilizer = WordStabilizer(
            window=1,
            min_confidence=0.20,
        )
        self.basic_sign_detector = BasicSignDetector()

        # TTS Engine
        try:
            self.tts = TTSEngine(enabled=self.tts_enabled)
        except Exception:
            self.tts = None

        print("[APIService] Initialization complete.")

    def get_status(self) -> Dict[str, Any]:
        """Return the current status of all models and pipelines."""
        return {
            "status": "online",
            "active_language": self.language,
            "active_mode": self.mode,
            "thresholds": {
                "isl": self.isl_threshold,
                "asl": self.asl_threshold,
            },
            "tts_enabled": self.tts_enabled,
            "models": {
                "isl_alphabet": {
                    "loaded": self.isl_recognizer.alphabet_adapter is not None and self.isl_recognizer.alphabet_adapter.is_loaded(),
                    "model_name": self.isl_recognizer.model_name,
                    "type": "Hugging Face Image Classification (Hemg/Indian-sign-language-classification)",
                },
                "isl_word": {
                    "loaded": (
                        self.isl_recognizer.isl_word_recognizer is not None
                        and self.isl_recognizer.isl_word_recognizer.is_loaded()
                    ) or (
                        self.isl_recognizer.word_adapter is not None
                        and self.isl_recognizer.word_adapter.is_loaded()
                    ),
                    "model_name": "Hierarchical Two-Stream Transformer (Abs6187/isl-models)",
                    "type": "Pretrained ISL Video & Pose Classifier (28 signs)",
                },
                "asl_word": {
                    "loaded": self.asl_recognizer.is_loaded(),
                    "model_name": "WLASL I3D PyTorch Checkpoint (2,000 glosses)",
                },
                "speech_whisper": {
                    "loaded": self.speech_recognizer.is_loaded(),
                    "model_name": self.speech_recognizer.model_name,
                    "type": "OpenAI Whisper ASR via Hugging Face",
                },
                "tts": {
                    "available": self.tts is not None,
                    "enabled": self.tts_enabled,
                },
            },
            "timestamp": time.time(),
        }

    def update_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update system settings dynamically."""
        if "language" in payload and payload["language"] in {"ISL", "ASL"}:
            self.language = payload["language"]
        if "mode" in payload and payload["mode"] in {"ALPHABET", "WORD"}:
            self.mode = payload["mode"]
            self.isl_recognizer.set_mode(self.mode)
        if "isl_threshold" in payload:
            self.isl_threshold = float(payload["isl_threshold"])
            self.isl_recognizer.confidence_threshold = self.isl_threshold
        if "asl_threshold" in payload:
            self.asl_threshold = float(payload["asl_threshold"])
        if "tts_enabled" in payload:
            self.tts_enabled = bool(payload["tts_enabled"])
            if self.tts:
                self.tts.enabled = self.tts_enabled

        return self.get_status()

    def process_frame(self, frame_b64: str) -> Dict[str, Any]:
        """Decode base64 image frame and perform sign recognition."""
        try:
            # Strip data URL prefix if present
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"error": "Failed to decode frame image"}
        except Exception as exc:
            return {"error": f"Frame decode error: {exc}"}

        threshold = self.isl_threshold if self.language == "ISL" else self.asl_threshold
        t0 = time.perf_counter()
        top_k: List[Dict[str, Any]] = []
        label = "NONE"
        confidence = 0.0
        smoothed_label = None

        if self.mode == "ALPHABET":
            if self.language == "ASL":
                primary_hand = None
                if self.extractor is not None:
                    primary_hand, _, _ = self.extractor.extract(frame)

                if primary_hand is not None and self.asl_classifier is not None:
                    vec = normalize_landmarks(primary_hand)
                    label, confidence, _ = self.asl_classifier.predict(vec)
                    raw_top = self.asl_classifier.get_top_k(vec, k=3)
                    top_k = [{"label": l, "confidence": round(float(c), 3)} for l, c in raw_top]
                    smoothed_label = self.alphabet_stabilizer.update(label, confidence)
                else:
                    self.alphabet_stabilizer.mark_no_hand()
                    label = "NO_HAND"
                    confidence = 0.0
            else:
                # ISL Alphabet
                result = self.isl_recognizer.predict([frame])
                label = result.label
                confidence = float(result.confidence)
                top_k = result.metadata.get("top_k", [])
                smoothed_label = self.alphabet_stabilizer.update(label, confidence)
        else:
            # WORD / DYNAMIC SIGN RECOGNITION
            primary_hand = None
            all_hands = []
            handedness = []
            if self.extractor is not None:
                primary_hand, all_hands, handedness = self.extractor.extract(frame)

            if primary_hand is not None:
                # 1. Instant Basic Sign Detection (Geometry Classifier)
                basic_sign, basic_conf = self.basic_sign_detector.detect(primary_hand, all_hands)
                if basic_sign and basic_conf >= 0.85:
                    label = basic_sign
                    confidence = basic_conf
                    top_k = [{"label": basic_sign, "confidence": basic_conf}]
                    smoothed_label = self.word_stabilizer.update(basic_sign, basic_conf)
                else:
                    isl_lms = (
                        self.extractor.extract_isl_features(frame, all_hands_xyz=all_hands, handedness_labels=handedness)
                        if self.language == "ISL" and self.extractor is not None
                        else None
                    )
                    self.video_buffer.add(frame, landmarks=isl_lms)
                    if self.video_buffer.is_ready():
                        sequence, landmarks = self.video_buffer.get_data()
                        recognizer = self.asl_recognizer if self.language == "ASL" else self.isl_recognizer
                        if self.language == "ISL":
                            result = recognizer.predict(sequence, pose_landmarks=landmarks)
                        else:
                            result = recognizer.predict(sequence)
                        label = result.label
                        confidence = float(result.confidence)
                        top_k = result.metadata.get("top_k", [])
                        smoothed_label = self.word_stabilizer.update(label, confidence)
                        self.video_buffer.clear()
                    else:
                        label = f"BUFFERING ({self.video_buffer.frame_count()}/32)"
                        confidence = self.video_buffer.frame_count() / 32.0
            else:
                self.word_stabilizer.mark_released()
                if self.video_buffer.frame_count() >= 8:
                    sequence, landmarks = self.video_buffer.get_data()
                    recognizer = self.asl_recognizer if self.language == "ASL" else self.isl_recognizer
                    if self.language == "ISL":
                        result = recognizer.predict(sequence, pose_landmarks=landmarks)
                    else:
                        result = recognizer.predict(sequence)
                    label = result.label
                    confidence = float(result.confidence)
                    top_k = result.metadata.get("top_k", [])
                    smoothed_label = self.word_stabilizer.update(label, confidence)
                else:
                    label = "NO_HAND"
                    confidence = 0.0
                self.video_buffer.clear()

        inference_ms = (time.perf_counter() - t0) * 1000.0

        is_uncertain = confidence < threshold or label in {"UNCERTAIN", "MODEL_UNAVAILABLE", "UNKNOWN", "ERROR", "NO_HAND"} or "BUFFERING" in label
        display_text = label if not is_uncertain else ("Sign not recognized. Please try again." if label not in ("NO_HAND", "NONE") and "BUFFERING" not in label else "--")

        # If a stable sign or word is confirmed, push to common text layer
        if smoothed_label:
            self.common_layer.add_entry(
                text=smoothed_label,
                source=self.language,
                mode=self.mode,
                confidence=confidence,
                metadata={"inference_ms": inference_ms},
            )
            if self.tts_enabled and self.tts:
                self.tts.speak_async(smoothed_label)

        return {
            "language": self.language,
            "mode": self.mode,
            "raw_label": label,
            "display_label": display_text,
            "confidence": round(confidence, 3),
            "threshold": threshold,
            "is_uncertain": is_uncertain,
            "smoothed_label": smoothed_label,
            "inference_ms": round(inference_ms, 1),
            "top_k": top_k,
            "transcript_summary": self.common_layer.get_full_transcript(),
        }

    def process_speech(self, text: Optional[str] = None, audio_b64: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe speech audio or post direct speech text."""
        recognized_text = text or ""
        confidence = 0.95

        if audio_b64 and not recognized_text:
            try:
                # If audio data provided, transcribe with Whisper
                if "," in audio_b64:
                    audio_b64 = audio_b64.split(",", 1)[1]
                audio_bytes = base64.b64decode(audio_b64)
                # Convert raw PCM/WAV to float32
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                result = self.speech_recognizer.predict(audio_np, sample_rate=16000)
                recognized_text = result.label
                confidence = result.confidence
            except Exception as exc:
                return {"error": f"Speech processing error: {exc}"}

        if recognized_text and recognized_text.strip():
            self.common_layer.add_entry(
                text=recognized_text.strip(),
                source="SPEECH",
                mode="SPEECH",
                confidence=confidence,
            )
            if self.tts_enabled and self.tts:
                self.tts.speak_async(recognized_text)

        return {
            "source": "SPEECH",
            "recognized_text": recognized_text,
            "confidence": confidence,
            "transcript_summary": self.common_layer.get_full_transcript(),
        }

    def speak_text(self, text: str) -> Dict[str, Any]:
        """Trigger TTS speech output."""
        if not text:
            return {"error": "No text provided"}
        if self.tts:
            self.tts.speak_async(text)
            return {"success": True, "text": text}
        return {"error": "TTS engine not available"}


# Global service instance
SERVICE = SignLanguageAPIService()


class APIRequestHandler(SimpleHTTPRequestHandler):
    """Zero-dependency HTTP Request Handler implementing the REST API and static frontend."""

    def __init__(self, *args, **kwargs):
        directory = str(PROJECT_ROOT / "frontend")
        super().__init__(*args, directory=directory, **kwargs)

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(SERVICE.get_status())
        elif parsed.path == "/api/common-text":
            self._send_json(SERVICE.common_layer.to_dict())
        elif parsed.path.startswith("/api/"):
            self._send_json({"error": "Endpoint not found"}, status=404)
        else:
            # Serve static frontend files
            super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if parsed.path == "/api/config":
            self._send_json(SERVICE.update_config(payload))
        elif parsed.path == "/api/recognize/frame":
            frame_b64 = payload.get("frame", "")
            self._send_json(SERVICE.process_frame(frame_b64))
        elif parsed.path == "/api/speech/transcribe":
            text = payload.get("text")
            audio_b64 = payload.get("audio")
            self._send_json(SERVICE.process_speech(text=text, audio_b64=audio_b64))
        elif parsed.path == "/api/common-text/clear":
            SERVICE.common_layer.clear()
            self._send_json({"success": True, "message": "Transcript cleared"})
        elif parsed.path == "/api/tts/speak":
            text = payload.get("text", "")
            self._send_json(SERVICE.speak_text(text))
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)


def run_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Start the REST API and frontend server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, APIRequestHandler)
    print(f"\n=======================================================")
    print(f"  AI Sign Language & Speech-to-Text Web Server")
    print(f"  Open in Browser: http://{host}:{port}")
    print(f"=======================================================\n")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sign Language Web API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
