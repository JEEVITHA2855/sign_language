from __future__ import annotations

import os
from typing import Callable, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


class HuggingFaceCorrector:
    """
    Uses a Hugging Face chat model to transform raw
    fingerspelling output into a cleaner sentence.

    IMPORTANT:
    This class does NOT perform sign recognition.
    It only processes the text already produced by TextComposer.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        enabled: bool = True,
    ) -> None:

        self.api_key = (
            api_key
            or os.environ.get("HF_TOKEN", "")
        )

        self.model_name = (
            model_name
            or os.environ.get("HF_MODEL", "Qwen/Qwen3-8B")
        )

        self.enabled = enabled and bool(self.api_key)

        self.client = None

        if self.enabled:
            self.client = InferenceClient(
                api_key=self.api_key,
                provider="auto",
            )

    def retransform(
        self,
        raw_text: str,
    ) -> str:

        if not raw_text or not raw_text.strip():
            return ""

        if not self.enabled or self.client is None:
            return raw_text

        system_prompt = """
You are a language correction module for a
sign-language fingerspelling application.

Your task is ONLY to clean the recognized transcription.

Allowed:
- spelling correction
- capitalization
- basic grammar correction
- punctuation

Strict rules:
1. Do not invent information.
2. Do not add words that are not reasonably implied
   by an obvious spelling mistake.
3. Do not change the meaning.
4. Preserve the original sentence structure as much
   as possible.
5. Return ONLY the corrected sentence.
6. Do not explain your answer.
7. Do not output reasoning.
"""

        try:

            response = self.client.chat.completions.create(
                model=self.model_name,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            "/no_think\n\n"
                            "Correct this sign-language "
                            "transcription:\n\n"
                            f"{raw_text}\n\n"
                            "Return ONLY the corrected sentence."
                        ),
                    },
                ],

                max_tokens=100,
                temperature=0.1,

                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            )

            content = response.choices[0].message.content

            if content:
                return content.strip()

            return raw_text

        except Exception as exc:

            print(
                f"[HuggingFace Warning] "
                f"Language correction failed: {exc}"
            )

            # Critical:
            # Application must continue working even if
            # Hugging Face is unavailable.
            return raw_text

    def retransform_async(
        self,
        raw_text: str,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:

        import threading

        def worker():

            result = self.retransform(raw_text)

            if callback:
                callback(result, "HUGGING FACE")

        thread = threading.Thread(
            target=worker,
            daemon=True,
        )

        thread.start()