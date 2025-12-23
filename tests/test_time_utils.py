from time_utils import format_srt_timestamp, format_timestamp, parse_timestamp


def test_parse_timestamp() -> None:
    assert abs(parse_timestamp("00:01:02,500") - 62.5) < 1e-6
    assert abs(parse_timestamp("00:01:02.250") - 62.25) < 1e-6


def test_format_timestamp() -> None:
    assert format_timestamp(62.5) == "00:01:02"
    assert format_srt_timestamp(62.5) == "00:01:02,500"
