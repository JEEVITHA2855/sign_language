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
from src.recognition.basic_sign_detector import BasicSignDetector
from src.recognition.expanded_sign_detector import ExpandedSignDetector
from src.recognition.sign_model_router import SignModelRouter
from src.language.sentence_framing_engine import SentenceFramingEngine
from src.communication.common_text_layer import CommonTextLayer

# Universal Basic Signs & Dynamic Words Action Guide
BASIC_WORD_ACTIONS = {
    "HELLO": ("Open Palm", "Raise flat open hand facing camera to greet"),
    "YES": ("Thumbs Up", "Extend thumb upright with fist to agree"),
    "NO": ("Thumbs Down", "Point thumb downward with fist to disagree"),
    "I LOVE YOU": ("ILY Sign", "Thumb, index & pinky upright (curl middle 2)"),
    "PEACE": ("Peace / Two", "Extend index and middle in a V shape"),
    "OK": ("OK Sign", "Touch thumb and index tips, extend other 3"),
    "THANK YOU": ("Praying / Chest", "Join both hands or push flat palm forward"),
    "YOU": ("Point Finger", "Point index finger forward at partner"),
    "CALL ME": ("Phone / Shaka", "Extend thumb and pinky outward"),
    "STOP": ("Closed Fist", "Hold firm closed fist facing forward"),
    "HOSPITAL": ("ISL Hospital", "Cross fingers or tap upper arm"),
    "DOCTOR": ("ISL Doctor", "Tap two fingers on wrist pulse"),
    "HOW_ARE_YOU_": ("ISL How Are You", "Open hands turning upward toward partner"),
    "HELP": ("ASL Help", "Place thumbs up fist on open flat palm"),
    "HAPPY": ("ISL Happy", "Sweep flat hand upward across chest"),
    "MONDAY": ("ISL Monday", "Circular M gesture in the air"),
}

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
        requested_mode = str(mode_override or recognition_cfg.get("mode", "AUTO")).upper()
        self.mode = requested_mode if requested_mode in {"WORD", "FINGERSPELLING", "AUTO"} else "AUTO"
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
            window=recognition_cfg.get("word_stabilizer_window", 7),
            min_confidence=recognition_cfg.get("word_min_confidence", 0.82),
            cooldown_seconds=1.0,
        )
        self.basic_sign_detector = BasicSignDetector()
        self.word_buffer = WordBuffer()
        self._previous_hand = None
        self.word_prediction = "NONE"
        self.word_confidence = 0.0
        self.word_inference_ms = 0.0
        self._last_tts_time = 0.0
        self._last_sentence_tts_time = 0.0
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
            no_hand_timeout_frames=composer_cfg.get("no_hand_timeout_frames", 20),
            auto_space=composer_cfg.get("auto_space", True),
        )
        self.common_layer = CommonTextLayer(max_history=100)
        self.sign_router = SignModelRouter(language=self.language, mode=self.mode, models_dir="models")

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
        self.sentence_framer = SentenceFramingEngine(corrector=self.huggingface)

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
        return self.mode

    def _word_recognizer(self):
        return self.asl_recognizer if self.language == "ASL" else self.isl_recognizer

    def _process_word_frame(
        self,
        frame: np.ndarray,
        primary_hand: Optional[np.ndarray],
        all_hands: Optional[List[np.ndarray]] = None,
        handedness: Optional[List[str]] = None,
    ) -> Optional[str]:
        self.word_prediction = "NONE"
        self.word_confidence = 0.0

        # 1. Sign Model Router (Everyday Temporal Attention Model + Geometric Detector)
        self.sign_router.set_language(self.language)
        self.sign_router.set_mode(self.mode)

        if primary_hand is not None or (all_hands and len(all_hands) >= 2):
            seq_154 = None
            if self.video_buffer.frame_count() >= 15:
                _, landmarks = self.video_buffer.get_data()
                if landmarks is not None and len(landmarks) > 0:
                    seq_154 = np.array(landmarks, dtype=np.float32)

            route_res = self.sign_router.route_and_predict(primary_hand, all_hands, sequence_154=seq_154)
            if route_res.label not in ("NONE", "MODEL_UNAVAILABLE") and route_res.confidence >= self.word_stabilizer.min_confidence:
                self.word_prediction = route_res.label
                self.word_confidence = route_res.confidence
                committed = self.word_stabilizer.update(route_res.label, route_res.confidence)
                if committed:
                    self.word_buffer.add_word(committed)
                    self.sentence_framer.add_word(committed)
                    self.composer.add_token(committed)
                    self.common_layer.add_entry(committed, source=self.language, mode="WORD", confidence=route_res.confidence)
                    self.set_status(f"Confirmed Word: [{committed}]")
                    now = time.time()
                    if self.tts.enabled and (now - self._last_tts_time) > 1.0:
                        self._last_tts_time = now
                        self.tts.speak_token(committed, is_completed_word=True, is_sentence=False)
                    return committed
                return None
            else:
                self.word_stabilizer.update("NONE", 0.0)

        # 2. Dynamic Video Sign Recognition (ISL Transformer / ASL I3D)
        if primary_hand is None:
            self.word_stabilizer.mark_released()
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
                    if accepted and accepted not in ("NONE", "UNCERTAIN", "UNKNOWN", "MODEL_UNAVAILABLE"):
                        self.word_buffer.add_word(accepted)
                        self.sentence_framer.add_word(accepted)
                        self.composer.add_token(accepted)
                        self.common_layer.add_entry(accepted, source=self.language, mode="WORD", confidence=self.word_confidence)
                        self.set_status(f"Confirmed word: [{accepted}]")
                        now = time.time()
                        if self.tts.enabled and (now - self._last_tts_time) > 1.2:
                            self._last_tts_time = now
                            self.tts.speak_token(accepted, is_completed_word=True, is_sentence=False)
                self.video_buffer.clear()
                self._previous_hand = None
                return None
        else:
            # Add frame and landmarks into video buffer while hand is active
            isl_lms = None
            if self.language == "ISL" and self.extractor is not None:
                isl_lms = self.extractor.extract_isl_features(frame, all_hands_xyz=all_hands, handedness_labels=handedness)
            self.video_buffer.add(frame, landmarks=isl_lms)

            # Auto-predict if video buffer is full (32 frames reached)
            if self.video_buffer.frame_count() >= self.video_buffer.sequence_length:
                sequence, landmarks = self.video_buffer.get_data()
                if self.language == "ISL":
                    result = self._word_recognizer().predict(sequence, pose_landmarks=landmarks)
                else:
                    result = self._word_recognizer().predict(sequence)
                self.word_prediction = result.label
                self.word_confidence = result.confidence
                self.word_inference_ms = float(result.metadata.get("inference_ms", 0.0))
                accepted = self.word_stabilizer.update(result.label, result.confidence)
                if accepted and accepted not in ("NONE", "UNCERTAIN", "UNKNOWN", "MODEL_UNAVAILABLE"):
                    self.word_buffer.add_word(accepted)
                    self.sentence_framer.add_word(accepted)
                    self.composer.add_token(accepted)
                    self.common_layer.add_entry(accepted, source=self.language, mode="WORD", confidence=self.word_confidence)
                    self.set_status(f"Confirmed word: [{accepted}]")
                    now = time.time()
                    if self.tts.enabled and (now - self._last_tts_time) > 1.2:
                        self._last_tts_time = now
                        self.tts.speak_token(accepted, is_completed_word=True, is_sentence=False)
                self.video_buffer.clear()
        return None

    def trigger_ai_retransform(
        self,
        speak_after: bool = True,
    ) -> None:
        """Query the active AI language corrector to refine words into a fluent grammatical sentence."""
        raw_stream = (
            " ".join(self.sentence_framer.get_active_words())
            or self.word_buffer.get_sentence()
            or self.composer.get_raw_stream()
        )

        if not raw_stream or not raw_stream.strip():
            return

        self.ai_status = "AI FRAMING..."
        self.set_status("AI reconstructing sentence...")

        framed = self.sentence_framer.frame_sentence(flush=False)
        if framed:
            self.ai_sentence = framed
            self.composer.set_retransformed_sentence(framed)
            self.ai_status = "AI READY"
            self.set_status(f"Framed Sentence: '{framed}'")
            if speak_after and self.tts.enabled:
                self.tts.speak_token(framed, is_sentence=True)

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

                    word_committed = None
                    if self.mode in ("AUTO", "WORD"):
                        if self.extractor is not None and all_hands:
                            self.extractor.draw_landmarks(disp_cam, all_hands, handedness)
                        word_committed = self._process_word_frame(frame, primary_hand, all_hands, handedness)
                        if self.word_prediction != "NONE" and self.word_confidence >= 0.85:
                            best_label = self.word_prediction
                            conf = self.word_confidence

                    # Run Alphabet Fingerspelling if in FINGERSPELLING mode or in AUTO mode when no word gesture is committed
                    if not word_committed and primary_hand is not None and self.mode in ("AUTO", "FINGERSPELLING"):
                        if self.mode == "FINGERSPELLING" or (self.mode == "AUTO" and self.word_prediction == "NONE"):
                            # Draw aesthetic hand skeleton
                            if self.extractor is not None:
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
                                self.set_status(f"Confirmed Letter: [{committed}]")

                                # Speak completed word when user signs SPACE
                                if committed == "SPACE":
                                    last_word = self.composer.get_last_completed_word() or self.composer.get_current_word()
                                    if last_word:
                                        self.sentence_framer.add_word(last_word)
                                        self.word_buffer.add_word(last_word)
                                        self.set_status(f"Completed Word: [{last_word}]")
                                        now = time.time()
                                        if self.tts.enabled and (now - self._last_tts_time) > 0.8:
                                            self._last_tts_time = now
                                            self.tts.speak_token(last_word, is_completed_word=True, is_sentence=False)
                    elif primary_hand is None and not (all_hands and len(all_hands) >= 2):
                        # Hand not detected in this frame
                        self.stabilizer.mark_no_hand()
                        self.word_stabilizer.mark_released()
                        boundary_word = self.composer.mark_no_hand()
                        if boundary_word:
                            self.sentence_framer.add_word(boundary_word)
                            self.word_buffer.add_word(boundary_word)
                            self.set_status(f"Completed Word: [{boundary_word}]")
                            now = time.time()
                            if self.tts.enabled and (now - self._last_tts_time) > 0.8:
                                self._last_tts_time = now
                                self.tts.speak_token(boundary_word, is_completed_word=True, is_sentence=False)

                        # Auto-sentence framing after gesture pause (must have >= 2 words to frame sentence)
                        now = time.time()
                        if len(self.sentence_framer.words) >= 2 and (now - self.sentence_framer.last_word_time) > 2.2:
                            if (now - self._last_sentence_tts_time) > 3.0:
                                self._last_sentence_tts_time = now
                                framed = self.sentence_framer.frame_sentence(flush=True)
                                if framed and len(framed.strip()) >= 5:
                                    self.ai_sentence = framed
                                    self.composer.set_retransformed_sentence(framed)
                                    self.set_status(f"Framed Sentence: '{framed}'")
                                    if self.tts.enabled:
                                        self.tts.speak_token(framed, is_sentence=True)
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
                    spoken_text = self.sentence_framer.last_framed_sentence or self.composer.get_display_sentence()
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
                    self.set_status("Mode: 100+ WORDS & GESTURES")
                elif key == ord("3"):
                    self.mode = "AUTO"
                    self.set_status("Mode: AUTO (WORDS + LETTERS)")
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
                    self.sentence_framer.clear()
                    self.word_stabilizer.reset()
                    self.video_buffer.clear()
                    self.sign_segmenter.reset()
                    self.ai_sentence = ""
                    self.set_status("Cleared all words, buffers and framed sentence")
                elif key == 8:  # Backspace
                    self.sentence_framer.backspace()
                    self.word_buffer.backspace()
                    self.composer.backspace()
                    self.set_status("Deleted last word / sign")
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
        header_title = f"{self.language} WORD & BASIC SIGNS" if self._active_pipeline() == "WORD" else f"{self.language} ALPHABET GUIDE"
        cv2.putText(
            canvas,
            header_title,
            (sx + 16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Card 1: Active Gesture & Posture Instruction
        cy = 48
        cv2.rectangle(canvas, (sx + 10, cy), (sx + sw - 10, cy + 85), (28, 28, 36), -1)
        cv2.rectangle(canvas, (sx + 10, cy), (sx + sw - 10, cy + 85), (60, 60, 75), 1)

        disp_sign = active_label if active_label not in ("NONE", "UNKNOWN", "UNCERTAIN") else "--"
        sign_col = (0, 255, 180) if confidence >= 0.65 else (0, 180, 255)
        cv2.putText(canvas, "ACTIVE DETECTED SIGN", (sx + 20, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1)
        cv2.putText(canvas, disp_sign[:14], (sx + 20, cy + 48), cv2.FONT_HERSHEY_DUPLEX, 0.85, sign_col, 2, cv2.LINE_AA)

        # Retrieve action description for active sign
        if active_label in ExpandedSignDetector.SIGN_DESCRIPTIONS:
            action_info = (active_label, ExpandedSignDetector.SIGN_DESCRIPTIONS[active_label])
        elif self._active_pipeline() in ("WORD", "AUTO"):
            action_info = BASIC_WORD_ACTIONS.get(active_label, (active_label, f"Recognized sign in {self.language} vocabulary"))
        else:
            action_info = ALPHABET_ACTIONS.get(active_label, ("Resting hand", "Show an ASL alphabet sign to the camera"))

        cv2.putText(canvas, f"Action: {action_info[0]}", (sx + 85, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 220), 1)
        cv2.putText(canvas, f"{action_info[1][:42]}", (sx + 85, cy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 180), 1)

        # Confidence Bar
        bar_x, bar_y, bar_w, bar_h = sx + 20, cy + 68, sw - 80, 8
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 50), -1)
        fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
        bar_col = (0, 240, 120) if confidence >= 0.70 else ((0, 180, 255) if confidence >= 0.40 else (0, 70, 255))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        cv2.putText(canvas, f"{confidence * 100:.0f}%", (bar_x + bar_w + 6, bar_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Section 1: Keyboard Controls & Shortcuts
        cy = 145
        cv2.putText(canvas, f"CONTROLS [{self.language} | {self._active_pipeline()}]", (sx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)
        shortcuts = [
            ("[1]", "Alphabet Mode (Fingerspelling A-Z)"),
            ("[2]", f"Word Mode (100+ Signs & {self.language})"),
            ("[3]", "AUTO Mode (Combined Letters & Words)"),
            ("[L]", f"Toggle Language (Active: {self.language})"),
            ("[G]", "Frame Sentence (Gloss-to-English AI)"),
            ("[S]", "Speak Active Sentence via TTS"),
            ("[C]", "Clear Words & Sentence Buffers"),
            ("[Q]", "Quit Application"),
        ]
        sc_y = cy + 18
        for key, desc in shortcuts:
            cv2.putText(canvas, f"{key:<6} : {desc}", (sx + 16, sc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 200), 1)
            sc_y += 16

        # Section 2: Reference Guide (Alphabet Grid vs 100+ Words Guide)
        cy = 280
        if self._active_pipeline() in ("WORD", "AUTO"):
            cv2.putText(canvas, "100+ VOCABULARY GUIDE (KEY SIGNS)", (sx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)
            sample_signs = [
                ("HELLO", "Open flat palm forward"),
                ("YES / NO", "Thumbs up / Thumbs down"),
                ("HELP / DOCTOR", "Thumbs up on palm / wrist tap"),
                ("WATER / FOOD", "W-hand at lips / hand to mouth"),
                ("PLEASE / THANK YOU", "Chest circular rub / chin to out"),
                ("WHAT / WHERE", "Open palms shrug / Index waggle"),
                ("I LOVE YOU / PEACE", "ILY sign / V-peace fingers"),
                ("HOSPITAL / CALL ME", "Cross tap / Phone shaka"),
            ]
            by_pos = cy + 18
            for sname, sdesc in sample_signs:
                is_active = any(part.strip() == active_label for part in sname.split("/"))
                tcol = (0, 255, 180) if is_active else (200, 200, 210)
                prefix = ">" if is_active else "*"
                cv2.putText(canvas, f"{prefix} {sname:<18} : {sdesc[:22]}", (sx + 16, by_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.33, tcol, 1)
                by_pos += 15
        else:
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

        ref_sign = active_label if (active_label in ALPHABET_ACTIONS or active_label in ExpandedSignDetector.SIGN_DESCRIPTIONS) else "A"
        if ref_sign in ExpandedSignDetector.SIGN_DESCRIPTIONS:
            ref_action = (ref_sign, ExpandedSignDetector.SIGN_DESCRIPTIONS[ref_sign])
        else:
            ref_action = ALPHABET_ACTIONS.get(ref_sign, ("--", "--"))
        cv2.putText(canvas, f"Sign [{ref_sign}]: {ref_action[0]}", (sx + 20, cy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1)
        cv2.putText(canvas, f"{ref_action[1][:48]}", (sx + 20, cy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 190), 1)

        # Section 4: AI Status & TTS Indicator Card
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
        tts_text = "TTS: [SPEAKING...]" if tts_active else f"TTS: [{self.tts.mode.upper()} - SPEECH READY]"
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
            f"Vocabulary: 100+ Real Signs & ASL Finger-spelling",
        ]
        for line in debug_lines:
            cy += 12
            cv2.putText(canvas, line, (sx + 16, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (185, 185, 195), 1)

    def _render_bottom_ribbon(self, canvas: np.ndarray) -> None:
        """Render the bottom ribbon displaying Recognized Words, Framed Sentence, and Vocalization/TTS status."""
        rx = 0
        ry = self.cam_h
        rw = self.total_w

        # Horizontal divider
        cv2.line(canvas, (0, ry), (rw, ry), (50, 50, 60), 2)
        cv2.rectangle(canvas, (0, ry), (rw, self.total_h), (14, 14, 18), -1)

        # Line 1: Recognized Word Stream / Buffer
        active_words = self.sentence_framer.get_active_words() if hasattr(self, "sentence_framer") else []
        if active_words:
            raw_stream = " -> ".join(active_words)
        else:
            raw_stream = self.word_buffer.get_sentence() if self._active_pipeline() in ("WORD", "AUTO") else self.composer.get_raw_stream()

        if not raw_stream:
            raw_stream = "[Perform any of 100+ signs or ASL alphabet to begin]"

        cv2.putText(canvas, "RECOGNIZED WORDS:", (16, ry + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 160), 1)
        disp_raw = raw_stream if len(raw_stream) < 55 else "..." + raw_stream[-52:]
        cv2.putText(canvas, disp_raw, (175, ry + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        # Line 2: Framed Natural Sentence (from SentenceFramingEngine)
        framed_sent = self.sentence_framer.last_framed_sentence if (hasattr(self, "sentence_framer") and self.sentence_framer.last_framed_sentence) else self.composer.get_display_sentence()
        if not framed_sent:
            framed_sent = "[AI framed sentence & speech will synthesize here automatically]"
        cv2.putText(canvas, "FRAMED SENTENCE:", (16, ry + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 200), 1)
        disp_framed = framed_sent if len(framed_sent) < 50 else framed_sent[:47] + "..."
        cv2.putText(canvas, disp_framed, (195, ry + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        # Line 3: Translation or Live System Status + TTS indicator
        tts_notice = "[SPEECH: PLAYING]" if getattr(self.tts, "is_speaking", False) else "[SPEECH: READY]"
        if self.enable_trans and framed_sent and not framed_sent.startswith("["):
            trans_text = translate_sentence(framed_sent, self.target_lang)
            cv2.putText(
                canvas,
                f"[{self.target_lang.upper()} TRANSLATION]: {trans_text}   {tts_notice}",
                (16, ry + 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                f"STATUS: {self._status_message} | Words: {len(active_words)} buffered | {tts_notice}",
                (16, ry + 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (180, 185, 200),
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
