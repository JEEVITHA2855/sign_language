"""Real-Time AI-Based Sign Language Detection, Gemini Re-transformation, and Translation Application."""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture import VideoCaptureWrapper
from src.preprocessing import normalize_landmarks
from src.classifier import GestureClassifier
from src.stabilizer import Stabilizer
from src.text_composer import TextComposer
from src.tts import TTSEngine
from src.nlp.huggingface_corrector import HuggingFaceCorrector
from src.nlp.translation import translate_sentence
from src.recognition.asl_recognizer import ASLRecognizer
from src.recognition.isl_recognizer import ISLRecognizer
from src.recognition.sign_segmenter import SignSegmenter
from src.recognition.video_buffer import VideoBuffer
from src.recognition.word_buffer import WordBuffer
from src.recognition.word_stabilizer import WordStabilizer
from src.communication.common_text_layer import CommonTextLayer

# Complete ASL Alphabet Hand Action and Posture Guide
ALPHABET_ACTIONS = {
    "A": ("Fist, thumb on side", "Make a fist with thumb upright against side of index"),
    "B": ("Flat 4 fingers up", "Hold 4 fingers straight up touching, thumb folded across palm"),
    "C": ("Curved C-cup", "Curve all fingers and thumb into a C cup shape"),
    "D": ("Index pointing up", "Index points straight up, middle/ring/pinky curl to touch thumb"),
    "E": ("Fingertips on thumb", "Curl all fingertips down tightly, resting on top of tucked thumb"),
    "F": ("OK shape (index+thumb)", "Touch index and thumb tips in circle, hold other 3 fingers up"),
    "G": ("Index points sideways", "Extend index horizontally forward, thumb parallel above it"),
    "H": ("Index+middle horizontal", "Extend index and middle together forward, thumb tucked"),
    "I": ("Pinky straight up", "Make a fist, extend only pinky finger upright"),
    "J": ("Pinky traces 'J'", "Extend pinky and trace a J-hook curve in the air"),
    "K": ("V-shape + thumb middle", "Index up, middle at 45 deg, thumb between them"),
    "L": ("Right-angle L-shape", "Extend thumb and index finger at 90 degrees"),
    "M": ("3 fingers over thumb", "Drape index, middle, ring fingers over tucked thumb"),
    "N": ("2 fingers over thumb", "Drape index and middle fingers over tucked thumb"),
    "O": ("All fingers in 'O'", "Curve all fingers to touch thumb tip, forming an O"),
    "P": ("Downward pointing K", "Form 'K' hand and tilt wrist so index points downward"),
    "Q": ("Downward pointing G", "Form 'G' hand and point index and thumb downwards"),
    "R": ("Crossed index & middle", "Cross middle finger over index finger"),
    "S": ("Fist, thumb across", "Make a tight fist, wrap thumb across front of fingers"),
    "T": ("Thumb between 1st & 2nd", "Fist with thumb inserted between index and middle"),
    "U": ("Index & middle together", "Extend index and middle straight up touching, thumb tucked"),
    "V": ("Peace sign (V-shape)", "Extend index and middle in a V shape spread apart"),
    "W": ("3 fingers up (W)", "Extend index, middle, ring fingers spread upright"),
    "X": ("Index finger hook", "Curl index into a hook, tuck other fingers into fist"),
    "Y": ("Thumb & pinky out", "Extend thumb and pinky out ('hang loose'), curl middle 3"),
    "Z": ("Index traces 'Z'", "Extend index finger and draw a Z path in the air"),
    "SPACE": ("Open flat hand", "Hold flat open hand horizontal to insert word space"),
    "DEL": ("Backspace / delete", "Pinch gesture or backspace key to delete last letter"),
}

ALPHABET_GRID = [chr(i) for i in range(ord("A"), ord("Z") + 1)] + ["SPACE", "DEL"]


class SignLanguageApp:
    """Main application loop coordinating Vision, ML, Gemini AI, TTS, and the Alphabet HUD."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        camera_override: Optional[int] = None,
        mock_mode: bool = False,
        disable_tts: bool = False,
        mode_override: Optional[str] = None,
    ) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        cam_cfg = self.cfg.get("camera", {})
        cam_idx = camera_override if camera_override is not None else cam_cfg.get("index", 0)

        self.labels = self.cfg.get("labels", [])
        recognition_cfg = self.cfg.get("recognition", {})
        requested_mode = str(mode_override or recognition_cfg.get("mode", "FINGERSPELLING")).upper()
        self.mode = requested_mode if requested_mode in {"WORD", "FINGERSPELLING", "AUTO"} else "FINGERSPELLING"
        self.language = str(recognition_cfg.get("language", "ASL")).upper()
        stab_cfg = self.cfg.get("stabilizer", {})
        tts_cfg = self.cfg.get("tts", {})
        nlp_cfg = self.cfg.get("nlp", {})
        gem_cfg = self.cfg.get("gemini", {})

        print("\n" + "=" * 68)
        print("  Real-Time AI Sign Language System (Gemini AI + ASL Alphabet)")
        print("=" * 68)
        print(f"  Camera Index   : {cam_idx} (Mock Mode: {mock_mode})")
        print(f"  Vocabulary     : {len(self.labels)} classes (A-Z + SPACE + DEL)")
        print(f"  TTS Mode       : {tts_cfg.get('mode', 'sentence').upper()} (Enabled: {not disable_tts})")
        print(f"  Gemini AI      : {gem_cfg.get('model', 'gemini-1.5-flash')} (Auto: {gem_cfg.get('auto_trigger_on_finish', True)})")
        print("=" * 68 + "\n")

        # Layout dimensions
        self.cam_w = cam_cfg.get("width", 640)
        self.cam_h = cam_cfg.get("height", 480)
        self.sidebar_w = 400  # Dedicated right-side Alphabet & Gemini panel
        self.bottom_h = 100  # Bottom ribbon for raw stream, AI sentence, and translation
        self.total_w = self.cam_w + self.sidebar_w
        self.total_h = self.cam_h + self.bottom_h

        # Initialize pipeline components
        self.capture = VideoCaptureWrapper(
            camera_index=cam_idx,
            width=self.cam_w,
            height=self.cam_h,
            fps=cam_cfg.get("fps", 30),
            mock_mode=mock_mode,
        )

        model_paths = self.cfg.get("model_paths", {})
        if mock_mode:
            self.extractor = None
        else:
            from src.landmark_extractor import LandmarkExtractor

            self.extractor = LandmarkExtractor(
                model_path=model_paths.get("hand_landmarker", "models/hand_landmarker.task")
            )

        self.classifier = GestureClassifier(
            model_path=model_paths.get("classifier", "models/gesture_classifier.onnx"),
            labels=self.labels,
            sklearn_fallback_path=model_paths.get("sklearn_classifier", "models/gesture_classifier.joblib"),
        )

        self.asl_recognizer = ASLRecognizer(
            enabled=True,
            model_path=model_paths.get("asl_model"),
            labels_path=model_paths.get("asl_labels"),
        )
        self.isl_recognizer = ISLRecognizer(enabled=True)
        self.video_buffer = VideoBuffer(
            max_frames=recognition_cfg.get("max_frames", 64),
            target_fps=recognition_cfg.get("target_fps", 15),
            min_frames=recognition_cfg.get("min_frames", 8),
            sequence_length=recognition_cfg.get("sequence_length", 32),
        )
        self.sign_segmenter = SignSegmenter(
            min_sign_frames=recognition_cfg.get("min_sign_frames", 8),
            max_sign_frames=recognition_cfg.get("max_sign_frames", 64),
            movement_threshold=recognition_cfg.get("movement_threshold", 0.015),
            pause_frames=recognition_cfg.get("pause_frames", 8),
        )
        self.word_stabilizer = WordStabilizer(
            window=recognition_cfg.get("word_stabilizer_window", 1),
            min_confidence=recognition_cfg.get("word_min_confidence", 0.35),
        )
        self.word_buffer = WordBuffer()
        self._previous_hand = None
        self.word_prediction = "NONE"
        self.word_confidence = 0.0
        self.word_inference_ms = 0.0
        self.word_model_status = (
            "LOADED" if self._word_recognizer().is_loaded() else "NOT AVAILABLE"
        )

        self.stabilizer = Stabilizer(
            window=stab_cfg.get("window", 5),
            min_ratio=stab_cfg.get("min_ratio", 0.70),
            min_conf=stab_cfg.get("min_conf", 0.70),
            required_stable_frames=stab_cfg.get("required_stable_frames", 5),
            debounce_frames=stab_cfg.get("debounce_frames", 10),
        )

        composer_cfg = self.cfg.get("text_composer", {})
        self.composer = TextComposer(
            no_hand_timeout_frames=composer_cfg.get("no_hand_timeout_frames", 45),
            auto_space=composer_cfg.get("auto_space", True),
        )
        self.common_layer = CommonTextLayer(max_history=100)

        self.tts = TTSEngine(
            enabled=(not disable_tts and tts_cfg.get("enabled", True)),
            rate=tts_cfg.get("rate", 165),
            volume=tts_cfg.get("volume", 1.0),
            voice_id=tts_cfg.get("voice_id"),
            mode=tts_cfg.get("mode", "sentence"),
        )

        # AI language corrector
        hf_key = os.environ.get("HF_TOKEN", "")
        hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen3-8B")

        self.huggingface = HuggingFaceCorrector(
            api_key=hf_key,
            model_name=hf_model,
            enabled=True,
        )

        self.ai_auto_trigger = gem_cfg.get(
            "auto_trigger_on_finish",
            True,
        )
        self.ai_status = "AI READY"
        self.ai_sentence = ""

        self.enable_nlp = nlp_cfg.get("enable_sentence_understanding", True)
        self.enable_trans = nlp_cfg.get("enable_translation", False)
        self.target_lang = nlp_cfg.get("target_language", "es")

        self.is_paused = False
        self.fps = 0.0
        self._last_frame_time = time.time()
        self._status_message = "Ready - Spell ASL letters to camera"
        self._status_time = time.time()

    def set_status(self, msg: str) -> None:
        self._status_message = msg
        self._status_time = time.time()

    def _active_pipeline(self) -> str:
        if self.mode == "AUTO":
            return "WORD" if self._word_recognizer().is_loaded() else "FINGERSPELLING"
        return self.mode

    def _word_recognizer(self):
        return self.asl_recognizer if self.language == "ASL" else self.isl_recognizer

    def _process_word_frame(
        self,
        frame: np.ndarray,
        primary_hand: Optional[np.ndarray],
        all_hands: Optional[List[np.ndarray]] = None,
        handedness: Optional[List[str]] = None,
    ) -> None:
        if primary_hand is None:
            movement = 0.0
            hand_confidence = 0.0
            # Hand left frame: mark release so identical subsequent words can be accepted
            self.word_stabilizer.mark_released()

            # If user had started signing and now removed hand, finalize gesture:
            if self.video_buffer.frame_count() >= self.video_buffer.min_frames:
                sequence, landmarks = self.video_buffer.get_data()
                if self.video_buffer.is_ready():
                    if self.language == "ISL":
                        result = self._word_recognizer().predict(sequence, pose_landmarks=landmarks)
                    else:
                        result = self._word_recognizer().predict(sequence)
                    self.word_prediction = result.label
                    self.word_confidence = result.confidence
                    self.word_inference_ms = float(result.metadata.get("inference_ms", 0.0))
                    accepted = self.word_stabilizer.update(result.label, result.confidence)
                    if accepted:
                        self.word_buffer.add_word(accepted)
                        self.common_layer.add_entry(accepted, source=self.language, mode="WORD", confidence=self.word_confidence)
                        self.set_status(f"Confirmed word: [{accepted}]")
                        if self.tts.mode == "word":
                            self.tts.speak_token(accepted, is_completed_word=True, is_sentence=False)
                self.video_buffer.clear()
                self.sign_segmenter.reset()
                self._previous_hand = None
                return
        else:
            if self._previous_hand is None:
                # A held static sign has no inter-frame movement; hand appearance
                # itself is the segment onset signal.
                movement = self.sign_segmenter.movement_threshold
            else:
                movement = float(np.mean(np.linalg.norm(primary_hand - self._previous_hand, axis=1)))
            hand_confidence = 1.0
            self._previous_hand = primary_hand.copy()

        event = self.sign_segmenter.update(movement, hand_confidence, time.time())
        if event in {"SIGN_STARTED", "SIGNING"} and primary_hand is not None:
            isl_lms = None
            if self.language == "ISL" and self.extractor is not None:
                isl_lms = self.extractor.extract_isl_features(frame, all_hands_xyz=all_hands, handedness_labels=handedness)
            self.video_buffer.add(frame, landmarks=isl_lms)

        if event == "SIGN_ENDED":
            sequence, landmarks = self.video_buffer.get_data()
            if self.video_buffer.is_ready():
                if self.language == "ISL":
                    result = self._word_recognizer().predict(sequence, pose_landmarks=landmarks)
                else:
                    result = self._word_recognizer().predict(sequence)
                self.word_prediction = result.label
                self.word_confidence = result.confidence
                self.word_inference_ms = float(result.metadata.get("inference_ms", 0.0))
                accepted = self.word_stabilizer.update(result.label, result.confidence)
                if accepted:
                    self.word_buffer.add_word(accepted)
                    self.common_layer.add_entry(accepted, source=self.language, mode="WORD", confidence=self.word_confidence)
                    self.set_status(f"Confirmed word: [{accepted}]")
                    if self.tts.mode == "word":
                        self.tts.speak_token(accepted, is_completed_word=True, is_sentence=False)
            self.video_buffer.clear()
            self.sign_segmenter.reset()
            self._previous_hand = None

    def trigger_ai_retransform(
        self,
        speak_after: bool = True,
    ) -> None:
        """Query the active AI language corrector to refine raw fingerspelling into a fluent sentence."""
        raw_stream = (
            self.word_buffer.get_sentence()
            if self._active_pipeline() == "WORD"
            else self.composer.get_raw_stream()
        )

        if not raw_stream or not raw_stream.strip():
            return

        if len(raw_stream.strip()) < 2:
            return

        self.ai_status = "AI TRANSFORMING..."
        self.set_status("AI reconstructing sentence...")

        def _on_retransformed(
            result: str,
            source: str,
        ):
            if result:
                self.ai_sentence = result
                self.composer.set_retransformed_sentence(result)
                self.ai_status = f"{source} READY"
                self.set_status(f"AI Reconstructed: '{result}'")
                if speak_after and self.tts.enabled:
                    self.tts.speak_async(result)
            else:
                self.ai_status = "READY"

        self.huggingface.retransform_async(
            raw_stream,
            callback=_on_retransformed,
        )

    def cycle_translation(self) -> None:
        langs = ["es", "fr", "de", "hi", None]
        curr_idx = langs.index(self.target_lang) if self.target_lang in langs else -1
        next_lang = langs[(curr_idx + 1) % len(langs)]
        if next_lang is None:
            self.enable_trans = False
            self.target_lang = "es"
            self.set_status("Translation: Disabled")
        else:
            self.enable_trans = True
            self.target_lang = next_lang
            self.set_status(f"Translation: Enabled ({next_lang.upper()})")

    def run(self, max_frames: Optional[int] = None) -> None:
        """Execute the real-time processing loop."""
        frame_idx = 0

        try:
            while True:
                if max_frames and frame_idx >= max_frames:
                    break

                now = time.time()
                dt = now - self._last_frame_time
                self._last_frame_time = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                ret, frame = self.capture.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                disp_cam = cv2.resize(frame, (self.cam_w, self.cam_h))

                best_label = "NONE"
                conf = 0.0
                top_k: List[Tuple[str, float]] = []

                if not self.is_paused:
                    # 1. Vision stage: MediaPipe Hand Landmarker
                    if self.extractor is None:
                        primary_hand, all_hands, handedness = None, [], []
                    else:
                        primary_hand, all_hands, handedness = self.extractor.extract(frame)

                    if self._active_pipeline() == "WORD":
                        if self.extractor is not None and all_hands:
                            self.extractor.draw_landmarks(disp_cam, all_hands, handedness)
                        self._process_word_frame(frame, primary_hand, all_hands, handedness)
                        best_label = self.word_prediction
                        conf = self.word_confidence
                    elif primary_hand is not None:
                        # Draw aesthetic hand skeleton
                        self.extractor.draw_landmarks(disp_cam, all_hands, handedness)

                        # 2. Preprocessing & Normalization
                        feature_vec = normalize_landmarks(primary_hand)

                        # 3. Machine Learning Classification
                        best_label, conf, _ = self.classifier.predict(feature_vec)
                        top_k = self.classifier.get_top_k(feature_vec, k=3)

                        # 4. Temporal Prediction Stabilization
                        committed = self.stabilizer.update(best_label, conf)
                        if committed:
                            # 5. Text Composition
                            self.composer.add_token(committed)
                            self.common_layer.add_entry(committed, source=self.language, mode="ALPHABET", confidence=conf)
                            self.set_status(f"Confirmed Sign: [{committed}]")

                            # 6. Audio TTS: uses mode (in sentence mode, individual letters are silent!)
                            self.tts.speak_token(committed, is_completed_word=False, is_sentence=False)
                    else:
                        # Hand not detected in this frame
                        self.stabilizer.mark_no_hand()
                        boundary_word = self.composer.mark_no_hand()
                        if boundary_word:
                            if self.ai_auto_trigger:
                                raw_stream = self.composer.get_raw_stream()
                                if raw_stream and len(raw_stream.strip()) >= 2:
                                    self.trigger_ai_retransform(speak_after=True)
                            elif self.enable_nlp:
                                fluent = self.composer.get_current_sentence()
                                if fluent:
                                    self.set_status(f"Completed phrase: {fluent}")
                                    self.tts.speak_async(fluent)
                else:
                    cv2.putText(
                        disp_cam,
                        "PAUSED - Press [SPACE] to resume",
                        (self.cam_w // 2 - 200, self.cam_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Assemble Full Composite Canvas (Camera + Alphabet Action Guide + Bottom Ribbon)
                canvas = self._render_full_dashboard(disp_cam, best_label, conf, top_k)

                # Interactive window display
                cv2.imshow("Real-Time AI Sign Language System", canvas)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q")):  # ESC or Q to quit
                    print("\nExiting application...")
                    break
                elif key in (ord("g"), ord("G"), ord("h"), ord("H")):  # G/H to trigger AI re-transformation
                    self.trigger_ai_retransform(speak_after=True)
                elif key in (ord("s"), ord("S")):  # S to speak sentence
                    spoken_text = self.composer.get_display_sentence()
                    if spoken_text:
                        self.tts.speak_async(spoken_text)
                        self.set_status(f"Speaking: '{spoken_text}'")
                elif key in (ord("m"), ord("M")):  # M to cycle TTS mode
                    new_mode = self.tts.cycle_mode()
                    self.set_status(f"TTS Audio Mode: {new_mode.upper()}")
                elif key == ord("1"):
                    self.mode = "FINGERSPELLING"
                    self.set_status("Mode: ALPHABET (FINGERSPELLING)")
                elif key == ord("2"):
                    self.mode = "WORD"
                    self.set_status("Mode: WORD / SIGN (DYNAMIC)")
                elif key == ord("3"):
                    self.mode = "AUTO"
                    self.set_status(f"Mode: AUTO ({self._active_pipeline()})")
                elif key in (ord("l"), ord("L")):
                    self.language = "ISL" if self.language == "ASL" else "ASL"
                    self.set_status(f"Language: {self.language}")
                elif key == ord(" "):  # Space to pause/resume
                    self.is_paused = not self.is_paused
                    self.set_status("Paused" if self.is_paused else "Resumed")
                elif key in (ord("c"), ord("C")):  # C to clear
                    self.composer.clear()
                    self.stabilizer.reset()
                    self.word_buffer.clear()
                    self.word_stabilizer.reset()
                    self.video_buffer.clear()
                    self.sign_segmenter.reset()
                    self.ai_sentence = ""
                    self.set_status("Buffers cleared")
                elif key == 8:  # Backspace
                    if self._active_pipeline() == "WORD":
                        self.word_buffer.backspace()
                        self.set_status("Deleted last word")
                    else:
                        self.composer.backspace()
                        self.set_status("Deleted last sign")
                elif key in (ord("t"), ord("T")):  # T to cycle translation
                    self.cycle_translation()

        except Exception as exc:
            import traceback
            print(f"\n[App Runtime Exception] Encountered unexpected error: {exc}")
            traceback.print_exc()
        finally:
            self.cleanup()

    def _render_full_dashboard(
        self,
        cam_frame: np.ndarray,
        active_label: str,
        confidence: float,
        top_k: List[Tuple[str, float]],
    ) -> np.ndarray:
        """Compose camera frame with dedicated 400px Alphabet Action Guide & Bottom Ribbon."""
        canvas = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 22)  # Sleek dark graphite theme

        # 1. Place camera frame on left
        canvas[0 : self.cam_h, 0 : self.cam_w] = cam_frame

        # Camera Header Bar
        cv2.rectangle(canvas, (0, 0), (self.cam_w, 42), (12, 12, 16), -1)
        cv2.putText(
            canvas,
            "AI SIGN LANGUAGE RECOGNITION",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 240, 255),
            2,
            cv2.LINE_AA,
        )
        mode_str = "MOCK" if self.capture.mock_mode else "LIVE"
        cv2.putText(
            canvas,
            f"{mode_str} | {self._active_pipeline()} | {self.fps:.1f} FPS",
            (self.cam_w - 145, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 160),
            1,
            cv2.LINE_AA,
        )

        # 2. Render Right Sidebar: ASL Alphabet Action Guide & Gemini Dashboard
        self._render_action_chart(canvas, active_label, confidence, top_k)

        # 3. Render Bottom Ribbon: Raw Stream, Gemini Predicted Sentence, Translation
        self._render_bottom_ribbon(canvas)

        return canvas

    def _render_action_chart(
        self,
        canvas: np.ndarray,
        active_label: str,
        confidence: float,
        top_k: List[Tuple[str, float]],
    ) -> None:
        """Draw the interactive right-side ASL Alphabet Action Guide with real-time pose descriptions."""
        sx = self.cam_w
        sw = self.sidebar_w

        # Vertical divider line
        cv2.line(canvas, (sx, 0), (sx, self.total_h), (50, 50, 60), 2)

        # Sidebar Header
        cv2.rectangle(canvas, (sx, 0), (sx + sw, 42), (24, 24, 30), -1)
        cv2.putText(
            canvas,
            "ASL ALPHABET & ACTION GUIDE",
            (sx + 16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Card 1: Active Gesture & Posture Instruction
        cy = 48
        cv2.rectangle(canvas, (sx + 10, cy), (sx + sw - 10, cy + 85), (28, 28, 36), -1)
        cv2.rectangle(canvas, (sx + 10, cy), (sx + sw - 10, cy + 85), (60, 60, 75), 1)

        disp_sign = active_label if active_label != "NONE" else "--"
        sign_col = (0, 255, 180) if confidence >= 0.75 else (0, 180, 255)
        cv2.putText(canvas, "ACTIVE DETECTED SIGN", (sx + 20, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1)
        cv2.putText(canvas, disp_sign, (sx + 20, cy + 48), cv2.FONT_HERSHEY_DUPLEX, 0.95, sign_col, 2, cv2.LINE_AA)

        # Retrieve action description for active sign
        action_info = ALPHABET_ACTIONS.get(active_label, ("Resting hand", "Show an ASL alphabet sign to the camera"))
        cv2.putText(canvas, f"Action: {action_info[0]}", (sx + 85, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 220), 1)
        cv2.putText(canvas, f"{action_info[1][:42]}", (sx + 85, cy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 180), 1)

        # Confidence Bar
        bar_x, bar_y, bar_w, bar_h = sx + 20, cy + 68, sw - 80, 8
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 50), -1)
        fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
        bar_col = (0, 240, 120) if confidence >= 0.75 else ((0, 180, 255) if confidence >= 0.50 else (0, 70, 255))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        cv2.putText(canvas, f"{confidence * 100:.0f}%", (bar_x + bar_w + 6, bar_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Section 1: Keyboard Controls & Shortcuts
        cy = 145
        cv2.putText(canvas, f"KEYBOARD CONTROLS [{self.language} | {self._active_pipeline()}]", (sx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)
        shortcuts = [
            ("[1 / 2]", f"Mode: [1] Alphabet, [2] Word ({self._active_pipeline()})"),
            ("[L]", f"Language: Toggle ISL / ASL (Active: {self.language})"),
            ("[G/H]", "AI Reconstruct Sentence (Hugging Face)"),
            ("[S]", "Speak Fluent Sentence via TTS"),
            ("[M]", f"Cycle TTS Mode ({self.tts.mode.upper()})"),
            ("[SPACE]", "Pause / Resume Stream"),
            ("[BKSP]", "Delete Last Sign / Letter"),
            ("[C]", "Clear All Buffers"),
            ("[T]", "Cycle Translation Language"),
            ("[Q]", "Quit Application"),
        ]
        sc_y = cy + 18
        for key, desc in shortcuts:
            cv2.putText(canvas, f"{key:<8} : {desc}", (sx + 16, sc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 200), 1)
            sc_y += 15

        # Section 2: Alphabet Quick Reference Grid (A–Z, SPACE, DEL)
        cy = 295
        cv2.putText(canvas, "ASL ALPHABET GRID (A-Z, SPACE, DEL)", (sx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1)

        grid_start_x = sx + 14
        grid_start_y = cy + 20
        cell_w, cell_h = 26, 22

        for idx, letter in enumerate(ALPHABET_GRID):
            row = idx // 14
            col = idx % 14
            bx = grid_start_x + col * cell_w
            by = grid_start_y + row * cell_h

            is_letter_active = (active_label == letter)
            if is_letter_active:
                cv2.rectangle(canvas, (bx, by - 15), (bx + cell_w - 3, by + 5), (0, 255, 180), -1)
                l_color = (0, 0, 0)
            else:
                cv2.rectangle(canvas, (bx, by - 15), (bx + cell_w - 3, by + 5), (32, 32, 40), -1)
                cv2.rectangle(canvas, (bx, by - 15), (bx + cell_w - 3, by + 5), (55, 55, 65), 1)
                l_color = (190, 190, 200)

            disp_l = letter[:2] if len(letter) > 1 else letter
            cv2.putText(canvas, disp_l, (bx + 3, by), cv2.FONT_HERSHEY_SIMPLEX, 0.36, l_color, 1)

        # Section 3: Alphabet Action Inspector Box
        cy = 360
        cv2.putText(canvas, "ACTION DETAIL FOR DETECTED SIGN", (sx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1)
        cv2.rectangle(canvas, (sx + 12, cy + 8), (sx + sw - 12, cy + 56), (24, 26, 32), -1)
        cv2.rectangle(canvas, (sx + 12, cy + 8), (sx + sw - 12, cy + 56), (50, 50, 65), 1)

        ref_sign = active_label if active_label in ALPHABET_ACTIONS else "A"
        ref_action = ALPHABET_ACTIONS.get(ref_sign, ("--", "--"))
        cv2.putText(canvas, f"Sign [{ref_sign}]: {ref_action[0]}", (sx + 20, cy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1)
        cv2.putText(canvas, f"{ref_action[1][:48]}", (sx + 20, cy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 190), 1)

        # Section 4: Gemini AI Status & TTS Indicator Card
        cy = 430
        tts_active = getattr(self.tts, "is_speaking", False)
        tts_bg = (0, 80, 40) if tts_active else (28, 28, 36)
        tts_border = (0, 255, 120) if tts_active else (60, 60, 70)
        cv2.rectangle(canvas, (sx + 12, cy), (sx + sw - 12, cy + 40), tts_bg, -1)
        cv2.rectangle(canvas, (sx + 12, cy), (sx + sw - 12, cy + 40), tts_border, 1)

        # AI status pill
        ai_color = (0, 255, 200) if "READY" in self.ai_status else (0, 180, 255)
        cv2.putText(canvas, f"AI: [{self.ai_status}]", (sx + 20, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ai_color, 1)

        # TTS pill
        tts_text = f"TTS: [SPEAKING...]" if tts_active else f"TTS: [{self.tts.mode.upper()} MODE]"
        cv2.putText(canvas, tts_text, (sx + 20, cy + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 160) if tts_active else (160, 160, 170), 1)

        cy = 472
        cv2.putText(canvas, "RECOGNITION DEBUG PANEL", (sx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)
        debug_lines = [
            f"Current Prediction: {active_label}",
            f"Confidence: {confidence * 100:.0f}% (Threshold: {self.stabilizer.min_conf * 100:.0f}%)",
            f"Stable Frames: {self.stabilizer.stable_count} / {self.stabilizer.required_stable_frames}",
            f"Hand Detected: {'YES' if self.stabilizer.hand_detected else 'NO'}",
            f"State: {self.stabilizer.state}",
            f"Last Committed: {self.stabilizer.last_committed or 'NONE'}",
            f"Recognition Mode: {self._active_pipeline()} [{self.language}]",
            f"Buffer Frames: {self.video_buffer.frame_count()} (Segment: {self.sign_segmenter.state})",
            f"ASL Word Model: {'READY (2000 words)' if self.asl_recognizer.is_loaded() else 'UNAVAILABLE'}",
        ]
        for line in debug_lines:
            cy += 12
            cv2.putText(canvas, line, (sx + 16, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (185, 185, 195), 1)

    def _render_bottom_ribbon(self, canvas: np.ndarray) -> None:
        """Render the bottom ribbon displaying Raw Sign Stream, Gemini Predicted Sentence, and Translation."""
        rx = 0
        ry = self.cam_h
        rw = self.total_w

        # Horizontal divider
        cv2.line(canvas, (0, ry), (rw, ry), (50, 50, 60), 2)
        cv2.rectangle(canvas, (0, ry), (rw, self.total_h), (14, 14, 18), -1)

        # Line 1: Raw Recognition Stream
        raw_stream = (
            self.word_buffer.get_sentence()
            if self._active_pipeline() == "WORD"
            else self.composer.get_raw_stream()
        )
        if not raw_stream:
            raw_stream = "[Spell ASL letters to compose phrase]"
        cv2.putText(canvas, "RAW SIGN STREAM:", (16, ry + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 160), 1)
        disp_raw = raw_stream if len(raw_stream) < 55 else "..." + raw_stream[-52:]
        cv2.putText(canvas, disp_raw, (175, ry + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        # Line 2: Gemini AI Predicted Fluent Sentence
        display_sentence = self.composer.get_display_sentence()
        if not display_sentence:
            display_sentence = "[AI reconstructed sentence will appear here]"
        cv2.putText(canvas, "AI FLUENT SENTENCE:", (16, ry + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 200), 1)
        cv2.putText(canvas, display_sentence, (195, ry + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        # Line 3: Translation or Live System Status
        if self.enable_trans:
            trans_text = translate_sentence(display_sentence, self.target_lang)
            cv2.putText(
                canvas,
                f"[{self.target_lang.upper()} TRANSLATION]: {trans_text}",
                (16, ry + 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                f"STATUS: {self._status_message}",
                (16, ry + 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )

    def cleanup(self) -> None:
        """Safely release all hardware and thread resources."""
        self.capture.release()
        if self.extractor is not None:
            self.extractor.close()
        self.tts.stop()
        cv2.destroyAllWindows()
        print("\nShutdown complete. Resources freed.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Detection Application.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--camera", type=int, default=None, help="Override camera index")
    parser.add_argument("--mock", action="store_true", help="Run with mock camera stream for headless testing")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice audio synthesis")
    parser.add_argument("--mode", choices=["WORD", "FINGERSPELLING", "AUTO"], default=None, help="Recognition pipeline mode")
    parser.add_argument("--frames", type=int, default=None, help="Run for N frames then exit (for automated testing)")
    args = parser.parse_args()

    app = SignLanguageApp(
        config_path=args.config,
        camera_override=args.camera,
        mock_mode=args.mock,
        disable_tts=args.no_tts,
        mode_override=args.mode,
    )
    app.run(max_frames=args.frames)


if __name__ == "__main__":
    main()
