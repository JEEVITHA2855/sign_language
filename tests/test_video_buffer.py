import numpy as np

from src.recognition.video_buffer import VideoBuffer


def test_video_buffer_tracks_sequence_and_readiness():
    buf = VideoBuffer(max_frames=4, target_fps=10, min_frames=2)
    for i in range(3):
        buf.add(np.zeros((10, 10, 3), dtype=np.uint8), timestamp=i)
        if i < 1:
            assert buf.is_ready() is False
    assert buf.frame_count() == 3
    seq = buf.get_sequence()
    assert len(seq) == 3
    assert buf.duration() >= 0.0


def test_video_buffer_resets_and_clears():
    buf = VideoBuffer(max_frames=4, target_fps=10, min_frames=1)
    buf.add(np.zeros((10, 10, 3), dtype=np.uint8), timestamp=1.0)
    buf.clear()
    assert buf.frame_count() == 0
    assert buf.is_ready() is False
