from src.recognition.base_recognizer import BaseSignRecognizer
from src.recognition.recognition_result import RecognitionResult
from src.recognition.asl_recognizer import ASLRecognizer
from src.recognition.isl_recognizer import ISLRecognizer


def test_recognition_result_has_expected_fields():
    result = RecognitionResult(
        label="WATER",
        confidence=0.91,
        language="ASL",
        mode="WORD",
        timestamp=1.0,
        source="test",
    )
    assert result.label == "WATER"
    assert result.confidence == 0.91


def test_recognizers_report_unavailable_when_no_model_loaded():
    asl = ASLRecognizer(model_name="missing-model")
    isl = ISLRecognizer(model_name="missing-model")
    asl_result = asl.predict([])
    isl_result = isl.predict([])
    assert asl_result.label == "MODEL_UNAVAILABLE"
    assert isl_result.label == "MODEL_UNAVAILABLE"
    assert isinstance(asl_result, RecognitionResult)
    assert isinstance(isl_result, RecognitionResult)


def test_base_sign_recognizer_exposes_interface():
    class DummyRecognizer(BaseSignRecognizer):
        def predict(self, frames):
            return RecognitionResult("TEST", 1.0, "ASL", "WORD", 0.0, "dummy")

    rec = DummyRecognizer()
    out = rec.predict([])
    assert out.label == "TEST"
