    from __future__ import annotations

    from dataclasses import dataclass
    from typing import List

    from time_utils import parse_timestamp, format_srt_timestamp


    @dataclass
    class SrtSegment:
        index: int
        start: float
        end: float
        text: str


    def parse_srt(path: str) -> List[SrtSegment]:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        return parse_srt_text(content)


    def parse_srt_text(text: str) -> List[SrtSegment]:
        lines = [line.rstrip("
") for line in text.splitlines()]
        segments: List[SrtSegment] = []
        index = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            seg_index = None
            if line.isdigit():
                seg_index = int(line)
                i += 1
            if i >= len(lines):
                break
            time_line = lines[i].strip()
            if "-->" not in time_line:
                i += 1
                continue
            start_str, end_str = [part.strip() for part in time_line.split("-->", 1)]
            try:
                start = parse_timestamp(start_str)
                end = parse_timestamp(end_str)
            except ValueError:
                i += 1
                continue
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text_value = " ".join(text_lines).strip()
            index += 1
            segments.append(SrtSegment(seg_index or index, start, end, text_value))
        return segments


    def segments_to_srt_text(segments: List[SrtSegment]) -> str:
        blocks = []
        for idx, seg in enumerate(segments, start=1):
            blocks.append(str(seg.index if seg.index else idx))
            blocks.append(
                f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}"
            )
            if seg.text:
                blocks.append(seg.text)
            blocks.append("")
        return "
".join(blocks).strip() + "
"
