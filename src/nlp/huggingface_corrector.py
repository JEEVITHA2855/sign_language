from __future__ import annotations

import os
import threading
from typing import Callable, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


class HuggingFaceCorrector:
    """Transforms raw fingerspelling or sign words into fluent, grammatically correct sentences.

    Supports:
    1. Local Neural Hugging Face Model (google/flan-t5-small) running 100% locally on CPU (0 credits).
    2. Cloud InferenceClient (optional fallback when valid HF_TOKEN is provided).
    3. Intelligent offline heuristic fallback (instant rule-based cleaner).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        local_model_name: str = "google/flan-t5-small",
        enabled: bool = True,
        use_local_model: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("HF_TOKEN", "")
        self.cloud_model_name = model_name or os.environ.get("HF_MODEL", "Qwen/Qwen3-8B")
        self.local_model_name = os.environ.get("HF_LOCAL_MODEL", local_model_name)
        self.enabled = enabled
        self.use_local_model = use_local_model

        # Cloud client (if explicitly configured with key)
        self.client = None
        if self.enabled and self.api_key:
            try:
                self.client = InferenceClient(api_key=self.api_key, provider="auto")
            except Exception:
                self.client = None

        # Local neural transformer pipeline
        self.local_pipeline = None
        self._local_loading = False
        self._local_lock = threading.Lock()

        if self.use_local_model:
            self._start_background_local_load()

    def _start_background_local_load(self) -> None:
        """Load the local Hugging Face model asynchronously so the app starts immediately."""
        import threading

        def _loader():
            with self._local_lock:
                if self.local_pipeline is not None or self._local_loading:
                    return
                self._local_loading = True
            try:
                import warnings
                warnings.filterwarnings("ignore")
                from transformers import pipeline

                pipe = pipeline(
                    "text2text-generation",
                    model=self.local_model_name,
                    device="cpu",
                )
                with self._local_lock:
                    self.local_pipeline = pipe
                    self._local_loading = False
                print(f"[HuggingFaceCorrector] Local neural model loaded: {self.local_model_name}")
            except Exception as exc:
                with self._local_lock:
                    self._local_loading = False
                print(f"[HuggingFaceCorrector] Notice: Local model init ({exc}). Using offline heuristic.")

        thread = threading.Thread(target=_loader, daemon=True)
        thread.start()

    def retransform(
        self,
        raw_text: str,
    ) -> str:
        if not raw_text or not raw_text.strip():
            return ""

        cleaned = raw_text.strip()

        # 1. Primary: Local trained Hugging Face Seq2Seq model (google/flan-t5-small)
        if self.local_pipeline is not None:
            try:
                # Normalize token markers like SPACE
                words_clean = " ".join(cleaned.replace("SPACE", " ").split())
                prompt = f"Rewrite into a complete sentence: {words_clean}"
                res = self.local_pipeline(prompt, max_new_tokens=48)
                if res and len(res) > 0 and "generated_text" in res[0]:
                    generated = res[0]["generated_text"].strip()
                    if generated and len(generated) >= 2:
                        if not generated.endswith((".", "?", "!")):
                            generated += "."
                        return generated[0].upper() + generated[1:]
            except Exception:
                pass

        # 2. Secondary: Cloud Hugging Face API (if available and credits active)
        if self.client is not None:
            system_prompt = (
                "You are a language correction module for a sign-language application. "
                "Your task is ONLY to clean the recognized sign transcription into a natural, "
                "grammatically correct sentence. Return ONLY the corrected sentence."
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.cloud_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"/no_think\n\nCorrect this sign-language transcription:\n\n{cleaned}\n\nReturn ONLY the corrected sentence.",
                        },
                    ],
                    max_tokens=60,
                    temperature=0.1,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            except Exception:
                pass

        # 3. Tertiary: High-accuracy offline heuristic fallback
        try:
            from src.nlp.gemini_corrector import local_heuristic_retransform

            heuristic_res = local_heuristic_retransform(cleaned)
            if heuristic_res:
                return heuristic_res
        except Exception:
            pass

        return cleaned

    def retransform_async(
        self,
        raw_text: str,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        import threading

        def worker():
            result = self.retransform(raw_text)
            source = "FLAN-T5 (LOCAL HF)" if self.local_pipeline is not None else "HUGGING FACE"
            if callback:
                callback(result, source)

        thread = threading.Thread(
            target=worker,
            daemon=True,
        )
        thread.start()