from clip_model import Clip, ClipOperations
from srt_utils import SrtSegment


def test_add_segment_indices() -> None:
    segments = [
        SrtSegment(1, 0.0, 4.0, "one two three"),
        SrtSegment(2, 4.5, 9.0, "four five six"),
        SrtSegment(3, 10.0, 16.0, "seven eight nine"),
        SrtSegment(4, 16.5, 25.0, "ten eleven twelve"),
    ]
    clips = [
        Clip(start_time=4.0, end_time=16.0, text="Test Clip"),
    ]
    updated = ClipOperations._add_segment_indices(clips, segments)
    assert updated[0].segment_start_index == 1
    assert updated[0].segment_end_index == 2
