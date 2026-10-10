from src.recognition.sign_segmenter import SignSegmenter


def test_sign_segmenter_detects_sign_boundary():
    segmenter = SignSegmenter(
        min_sign_frames=2,
        max_sign_frames=10,
        movement_threshold=0.01,
        pause_frames=1,
    )

    assert segmenter.update(0.02, 0.9, 1.0) == "SIGN_STARTED"
    assert segmenter.update(0.02, 0.9, 2.0) == "SIGNING"
    assert segmenter.update(0.005, 0.9, 3.0) == "SIGN_ENDED"
    assert segmenter.update(0.005, 0.9, 4.0) == "IDLE"
