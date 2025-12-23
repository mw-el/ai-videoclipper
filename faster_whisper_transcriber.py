from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from srt_utils import SrtSegment, parse_srt

logger = logging.getLogger("ai_videoclipper")


def log_and_print(msg: str, level: str = "INFO") -> None:
    """Log and print to stderr for visibility."""
    if level == "INFO":
        logger.info(msg)
    elif level == "ERROR":
        logger.error(msg)
    elif level == "DEBUG":
        logger.debug(msg)
    elif level == "WARNING":
        logger.warning(msg)
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)


class TranscriptionError(RuntimeError):
    pass


@dataclass
class TranscriptionResult:
    srt_path: Optional[Path]  # Path to generated SRT file (if saved)
    segments: List[SrtSegment]
    text: str


class FasterWhisperTranscriber:
    def __init__(
        self,
        conda_sh: str = "~/miniconda3/etc/profile.d/conda.sh",
        conda_env: str = "fasterwhisper",
    ) -> None:
        self.conda_sh = Path(conda_sh).expanduser()
        self.conda_env = conda_env
        log_and_print("Using faster-whisper via subprocess")

    def transcribe(
        self,
        file_path: str,
        language: str = "de",
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe audio/video file using faster-whisper via subprocess.

        Args:
            file_path: Path to audio or video file
            language: Language code (e.g., "de" for German)
            task: "transcribe" or "translate"

        Returns:
            TranscriptionResult with segments and full text
        """
        import time

        source_path = Path(file_path)
        log_and_print(f"Starting transcription of: {source_path}")

        if not source_path.exists():
            log_and_print(f"File not found: {source_path}", "ERROR")
            raise TranscriptionError(f"File not found: {source_path}")

        log_and_print(f"Transcribing with language: {language}")
        log_and_print(f"Task: {task}")

        # Create temp directory for output
        temp_dir = Path(tempfile.gettempdir()) / "faster_whisper_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_srt = temp_dir / f"transcription_{int(time.time())}.srt"

        log_and_print(f"Temp SRT will be saved to: {temp_srt}")

        # Create Python script to run in fasterwhisper environment
        script = f"""
import json
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float32")
segments, info = model.transcribe(r"{source_path}", language="{language}", task="{task}")
segments = list(segments)

# Save as SRT
with open(r"{temp_srt}", "w", encoding="utf-8") as f:
    for seg in segments:
        start = f"{{int(seg.start // 3600):02d}}:{{int((seg.start % 3600) // 60):02d}}:{{int(seg.start % 60):02d}},{{int((seg.start % 1) * 1000):03d}}"
        end = f"{{int(seg.end // 3600):02d}}:{{int((seg.end % 3600) // 60):02d}}:{{int(seg.end % 60):02d}},{{int((seg.end % 1) * 1000):03d}}"
        f.write(f"{{segments.index(seg) + 1}}\\n{{start}} --> {{end}}\\n{{seg.text.strip()}}\\n\\n")

print("OK")
"""

        cmd = f"source {shlex.quote(str(self.conda_sh))} && conda activate {shlex.quote(self.conda_env)} && python -c {shlex.quote(script)}"

        log_and_print("Executing faster-whisper subprocess...")
        log_and_print(f"Using environment: {self.conda_env}")
        start_time = time.time()

        try:
            log_and_print("Calling faster-whisper process...")
            result = subprocess.run(
                ["bash", "-lc", cmd],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            log_and_print("faster-whisper process completed")

        except subprocess.TimeoutExpired:
            log_and_print("ERROR: Transcription timed out after 1 hour", "ERROR")
            raise TranscriptionError("Transcription timed out")
        except Exception as e:
            log_and_print(f"ERROR: Failed to run transcription: {type(e).__name__}: {e}", "ERROR")
            raise TranscriptionError(f"Failed to run transcription: {e}")

        elapsed = time.time() - start_time
        log_and_print(f"Transcription completed in {elapsed:.1f} seconds (return code: {result.returncode})")
        log_and_print(f"Output stderr length: {len(result.stderr)} chars")

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            log_and_print(f"Transcription failed with code {result.returncode}", "ERROR")
            log_and_print(f"stderr (first 1000 chars): {stderr[:1000]}", "ERROR")
            raise TranscriptionError(f"Transcription failed: {stderr}")

        log_and_print("Transcription successful, parsing SRT...")

        if not temp_srt.exists():
            log_and_print(f"ERROR: SRT file not created: {temp_srt}", "ERROR")
            raise TranscriptionError("SRT file was not created")

        log_and_print(f"SRT file size: {temp_srt.stat().st_size} bytes")

        # Parse the SRT file
        segments = parse_srt(str(temp_srt))
        text = " ".join(seg.text for seg in segments).strip()

        log_and_print(f"Parsed {len(segments)} segments")
        log_and_print(f"Total text length: {len(text)} characters")

        return TranscriptionResult(srt_path=temp_srt, segments=segments, text=text)

    def save_srt(self, segments: List[SrtSegment], output_path: Path) -> Path:
        """Save segments to an SRT file."""
        log_and_print(f"Saving SRT to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                # Format: index, timecode, text, blank line
                start_str = self._format_timestamp(seg.start)
                end_str = self._format_timestamp(seg.end)
                f.write(f"{seg.index}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{seg.text}\n")
                f.write("\n")

        log_and_print(f"✓ SRT saved: {output_path} ({len(segments)} segments)")
        return output_path

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
