from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable, List, Optional

from srt_utils import SrtSegment


@dataclass
class Clip:
    start_time: float
    end_time: float
    text: str = ""
    score: Optional[float] = None
    segment_start_index: int = 0  # Index of first segment in clip
    segment_end_index: int = 0    # Index of last segment in clip


class ClipsAIWrapper:
    def __init__(self, use_clipsai: bool = True) -> None:
        self._clip_finder_cls = None
        self._media_editor_cls = None
        self._use_clipsai = use_clipsai
        self._clipsai_loaded = False

    def _ensure_clipsai(self) -> None:
        if not self._use_clipsai or self._clipsai_loaded:
            return
        try:
            from clipsai import ClipFinder, MediaEditor
        except Exception:
            self._clip_finder_cls = None
            self._media_editor_cls = None
            self._use_clipsai = False
            self._clipsai_loaded = True
            return
        self._clip_finder_cls = ClipFinder
        self._media_editor_cls = MediaEditor
        self._clipsai_loaded = True

    def find_clips(self, segments: Iterable[SrtSegment], max_clips: int = 6) -> List[Clip]:
        segments = list(segments)
        if not segments:
            return []
        self._ensure_clipsai()
        if self._clip_finder_cls:
            payload = self._segments_to_payload(segments)
            clip_finder = self._clip_finder_cls()
            raw_clips = self._try_clip_finder(clip_finder, payload)
            if raw_clips:
                clips = self._normalize_clips(raw_clips, max_clips)
                return self._add_segment_indices(clips, segments)
        clips = self._fallback_clips(segments, max_clips)
        return self._add_segment_indices(clips, segments)

    def trim_clip(
        self,
        source_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
    ) -> None:
        editor = self._get_media_editor(source_path)
        if editor:
            if self._try_editor_trim(editor, source_path, start_time, end_time, output_path):
                return
        self._ffmpeg_trim(source_path, start_time, end_time, output_path)

    def trim_all_clips(self, source_path: Path, clips: List[Clip], output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, clip in enumerate(clips, start=1):
            output_path = output_dir / f"clip_{idx:02d}.mp4"
            self.trim_clip(source_path, clip.start_time, clip.end_time, output_path)

    def _get_media_editor(self, source_path: Path):
        self._ensure_clipsai()
        if not self._media_editor_cls:
            return None
        try:
            return self._media_editor_cls(source_path)
        except Exception:
            try:
                return self._media_editor_cls()
            except Exception:
                return None

    def _try_editor_trim(self, editor, source_path, start_time, end_time, output_path) -> bool:
        try:
            editor.trim(source_path, start_time, end_time, str(output_path))
            return True
        except TypeError:
            pass
        except Exception:
            return False
        try:
            editor.trim(start_time, end_time, str(output_path))
            return True
        except Exception:
            return False

    @staticmethod
    def _ffmpeg_trim(source_path: Path, start_time: float, end_time: float, output_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)

    @staticmethod
    def _segments_to_payload(segments: List[SrtSegment]) -> dict:
        return {
            "text": " ".join(segment.text for segment in segments).strip(),
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments
            ],
        }

    @staticmethod
    def _try_clip_finder(clip_finder, payload):
        try:
            return clip_finder.find_clips(payload)
        except Exception:
            pass
        try:
            return clip_finder.find_clips(payload.get("segments"))
        except Exception:
            return None

    def _normalize_clips(self, raw_clips, max_clips: int) -> List[Clip]:
        if isinstance(raw_clips, dict) and "clips" in raw_clips:
            raw_clips = raw_clips["clips"]
        clips: List[Clip] = []
        for raw in raw_clips or []:
            start, end, text = self._extract_clip_fields(raw)
            if start is None or end is None:
                continue
            if end <= start:
                continue
            clips.append(Clip(float(start), float(end), text or ""))
        clips.sort(key=lambda clip: clip.start_time)
        return clips[:max_clips] if max_clips else clips

    @staticmethod
    def _extract_clip_fields(raw):
        if isinstance(raw, dict):
            start = raw.get("start_time", raw.get("start"))
            end = raw.get("end_time", raw.get("end"))
            text = raw.get("text", "")
            return start, end, text
        start = getattr(raw, "start_time", None)
        if start is None:
            start = getattr(raw, "start", None)
        end = getattr(raw, "end_time", None)
        if end is None:
            end = getattr(raw, "end", None)
        text = getattr(raw, "text", "")
        return start, end, text

    def _fallback_clips(self, segments: List[SrtSegment], max_clips: int) -> List[Clip]:
        if not segments:
            return []
        target_duration = 25.0
        max_duration = 45.0
        max_gap = 2.0
        min_duration = 8.0
        chunks = []
        current = {
            "start": segments[0].start,
            "end": segments[0].end,
            "text": segments[0].text,
        }
        for seg in segments[1:]:
            gap = seg.start - current["end"]
            current_duration = current["end"] - current["start"]
            projected_duration = seg.end - current["start"]
            if gap > max_gap or current_duration >= target_duration or projected_duration > max_duration:
                chunks.append(current)
                current = {"start": seg.start, "end": seg.end, "text": seg.text}
            else:
                current["end"] = seg.end
                current["text"] = f"{current['text']} {seg.text}".strip()
        if current:
            chunks.append(current)

        scored = []
        for chunk in chunks:
            duration = chunk["end"] - chunk["start"]
            if duration < min_duration:
                continue
            score = len(chunk["text"].split())
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [chunk for _, chunk in scored[:max_clips]]
        selected.sort(key=lambda item: item["start"])
        return [
            Clip(float(chunk["start"]), float(chunk["end"]), chunk["text"].strip())
            for chunk in selected
        ]

    @staticmethod
    def _add_segment_indices(clips: List[Clip], segments: List[SrtSegment]) -> List[Clip]:
        """Add segment start/end indices to clips based on timing."""
        for clip in clips:
            # Find first segment that starts at or after clip start time
            start_idx = 0
            for i, seg in enumerate(segments):
                if seg.start >= clip.start_time:
                    start_idx = i
                    break

            # Find last segment that ends at or before clip end time
            end_idx = len(segments) - 1
            for i in range(len(segments) - 1, -1, -1):
                if segments[i].end <= clip.end_time:
                    end_idx = i
                    break

            clip.segment_start_index = start_idx
            clip.segment_end_index = end_idx

        return clips
