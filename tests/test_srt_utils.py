    from srt_utils import parse_srt_text


    def test_parse_srt_text() -> None:
        srt = (
            "1
"
            "00:00:01,000 --> 00:00:02,000
"
            "Hello world

"
            "2
"
            "00:00:03,000 --> 00:00:05,500
"
            "Second line

"
        )
        segments = parse_srt_text(srt)
        assert len(segments) == 2
        assert segments[0].start == 1.0
        assert segments[1].end == 5.5
