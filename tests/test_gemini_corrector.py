import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nlp.gemini_corrector import GeminiCorrector


def test_legacy_gemini_api_is_supported(monkeypatch):
    calls = {}

    class FakeCompletion:
        def __init__(self, text):
            self.text = text

    class FakeGenAI:
        @staticmethod
        def configure(api_key=None):
            calls["api_key"] = api_key

        @staticmethod
        def generate_text(*, model, prompt, **kwargs):
            calls["model"] = model
            calls["prompt"] = prompt
            return FakeCompletion("Hello there!")

    monkeypatch.setitem(__import__("sys").modules, "google.generativeai", FakeGenAI)

    corr = GeminiCorrector(api_key="demo-key", enabled=True)
    result = corr.retransform("h e l l o")

    assert corr.client is not None
    assert calls["api_key"] == "demo-key"
    assert result == "Hello there!"
