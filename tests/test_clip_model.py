from clip_model import ClipsAIWrapper
from srt_utils import SrtSegment


def test_fallback_clips() -> None:
    segments = [
        SrtSegment(1, 0.0, 4.0, "one two three"),
        SrtSegment(2, 4.5, 9.0, "four five six"),
        SrtSegment(3, 10.0, 16.0, "seven eight nine"),
        SrtSegment(4, 16.5, 25.0, "ten eleven twelve"),
    ]
    wrapper = ClipsAIWrapper(use_clipsai=False)
    clips = wrapper.find_clips(segments, max_clips=2)
    assert clips
    assert clips[0].end_time > clips[0].start_time
