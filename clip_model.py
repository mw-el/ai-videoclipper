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
        pass

    def trim_clip(
        self,
        source_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
    ) -> None:
        self._smartcut_trim(source_path, start_time, end_time, output_path)

    def trim_all_clips(self, source_path: Path, clips: List[Clip], output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, clip in enumerate(clips, start=1):
            output_path = output_dir / f"clip_{idx:02d}.mp4"
            self.trim_clip(source_path, clip.start_time, clip.end_time, output_path)

    @staticmethod
    def _smartcut_trim(source_path: Path, start_time: float, end_time: float, output_path: Path) -> None:
        import logging
        logger = logging.getLogger("ai_videoclipper")

        # Check if smartcut is available
        try:
            result = subprocess.run(
                ["smartcut", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            error_msg = (
                "smartcut not found. Please install it:\n"
                "pip install smartcut\n"
                "or visit: https://github.com/wjs018/smartcut"
            )
            logger.error(f"[SMARTCUT] {error_msg}")
            raise RuntimeError(error_msg)

        # Run smartcut for frame-accurate cutting
        logger.info(f"[SMARTCUT] Cutting {source_path.name}: {start_time}s - {end_time}s")
        cmd = [
            "smartcut",
            str(source_path),
            str(output_path),
            "--keep",
            f"{start_time},{end_time}"
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[SMARTCUT] Successfully created {output_path.name}")
        except subprocess.CalledProcessError as e:
            error_msg = f"smartcut failed: {e.stderr}"
            logger.error(f"[SMARTCUT] {error_msg}")
            raise RuntimeError(error_msg)

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
