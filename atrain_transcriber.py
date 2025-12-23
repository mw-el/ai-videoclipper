from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, List, Optional

from srt_utils import SrtSegment, parse_srt

logger = logging.getLogger("ai_videoclipper")

# Also print to stderr for debugging
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
    srt_path: Path
    segments: List[SrtSegment]
    text: str


class ATrainTranscriber:
    def __init__(
        self,
        conda_sh: str = "~/miniconda3/etc/profile.d/conda.sh",
        conda_env: str = "atrain",
        output_dir: str = "~/Documents/aTrain/transcriptions",
        runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    ) -> None:
        self.conda_sh = Path(conda_sh).expanduser()
        self.conda_env = conda_env
        self.output_dir = Path(output_dir).expanduser()
        self.runner = runner or subprocess.run

    def transcribe(
        self,
        file_path: str,
        model: str = "large-v3-turbo",
        language: str = "de",
        device: str = "GPU",
        compute_type: str = "float32",
        speaker_detection: bool = False,
        num_speakers: str = "auto",
        test_mode: bool = False,
    ) -> TranscriptionResult:
        source_path = Path(file_path)
        log_and_print(f"Starting transcription of: {source_path}")

        if not source_path.exists():
            log_and_print(f"File not found: {source_path}", "ERROR")
            raise TranscriptionError(f"File not found: {source_path}")

        if not self.conda_sh.exists():
            log_and_print(f"Conda activation script not found: {self.conda_sh}", "ERROR")
            raise TranscriptionError(
                f"Conda activation script not found: {self.conda_sh}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_and_print(f"Output directory: {self.output_dir}")

        # aTrain_core doesn't support spaces or special chars in file paths
        # Copy to temp file if path contains problematic characters
        working_path = source_path
        temp_file = None

        # Check if filename has spaces or non-ASCII characters
        has_spaces = ' ' in str(source_path)
        has_special_chars = any(ord(c) > 127 for c in str(source_path))

        if has_spaces or has_special_chars:
            log_and_print(f"File path has spaces or special chars, creating temp copy...")
            temp_dir = Path(tempfile.gettempdir()) / "ai_videoclipper"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Create safe filename: keep only ASCII alphanumerics, dots, hyphens, underscores
            ext = source_path.suffix
            base_name = source_path.stem
            # Replace problematic characters with underscores
            safe_base = ''.join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
            # Remove consecutive underscores
            while '__' in safe_base:
                safe_base = safe_base.replace('__', '_')
            safe_base = safe_base.strip('_')
            safe_name = safe_base + ext

            temp_file = temp_dir / safe_name
            log_and_print(f"Original: {source_path.name}")
            log_and_print(f"Safe name: {safe_name}")
            log_and_print(f"Copying to temp file: {temp_file}")
            shutil.copy2(source_path, temp_file)
            working_path = temp_file
            log_and_print(f"Using working path: {working_path}")

        cmd = self._build_command(
            working_path,
            model=model,
            language=language,
            device=device,
            compute_type=compute_type,
            speaker_detection=speaker_detection,
            num_speakers=num_speakers,
            test_mode=test_mode,
        )
        log_and_print(f"aTrain command: {cmd}", "DEBUG")

        log_and_print("Executing aTrain_core transcribe...")
        log_and_print(f"Working in conda env: {self.conda_env}")
        start_ts = time.time()

        try:
            log_and_print(f"Subprocess call with: bash -lc [command]")
            log_and_print(f"Command length: {len(cmd)} characters")

            # Note: if self.runner is mocked in tests, it may not support timeout
            # Try with timeout, fall back without if unsupported
            try:
                log_and_print("Calling subprocess.run with 3600s timeout...")
                result = self.runner(["bash", "-lc", cmd], capture_output=True, text=True, timeout=3600)
                log_and_print("subprocess.run returned")
            except TypeError as te:
                # Runner doesn't support timeout (e.g., in tests)
                log_and_print(f"Timeout not supported, calling without timeout: {te}")
                result = self.runner(["bash", "-lc", cmd], capture_output=True, text=True)
                log_and_print("subprocess.run returned (no timeout)")
        except subprocess.TimeoutExpired as te:
            log_and_print("ERROR: aTrain_core timed out after 1 hour", "ERROR")
            raise TranscriptionError("aTrain_core transcription timed out after 1 hour")
        except Exception as e:
            log_and_print(f"ERROR: Failed to run aTrain_core: {type(e).__name__}: {e}", "ERROR")
            import traceback
            log_and_print(f"Traceback: {traceback.format_exc()}", "ERROR")
            raise TranscriptionError(f"Failed to run aTrain_core: {e}")
        finally:
            # Clean up temp file if it was created
            if temp_file:
                log_and_print(f"Checking for temp file cleanup: {temp_file}")
                if temp_file.exists():
                    try:
                        log_and_print(f"Deleting temp file...")
                        temp_file.unlink()
                        log_and_print(f"Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        log_and_print(f"Warning: Could not delete temp file {temp_file}: {e}", "WARNING")
                else:
                    log_and_print(f"Temp file already deleted: {temp_file}")

        elapsed = time.time() - start_ts
        log_and_print(f"aTrain_core completed in {elapsed:.1f} seconds (return code: {result.returncode})")
        log_and_print(f"Output stdout length: {len(result.stdout)} chars")
        log_and_print(f"Output stderr length: {len(result.stderr)} chars")

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            log_and_print(f"aTrain_core failed with code {result.returncode}", "ERROR")
            log_and_print(f"stderr (first 500 chars): {stderr[:500]}", "ERROR")
            log_and_print(f"stdout (first 500 chars): {result.stdout[:500]}", "ERROR")
            raise TranscriptionError(
                f"aTrain_core failed with code {result.returncode}: {stderr}"
            )

        log_and_print("aTrain_core completed successfully, searching for SRT file...")
        log_and_print(f"Start timestamp for search: {start_ts}")
        srt_path = self.find_latest_srt(start_ts, source_path)
        log_and_print(f"Found SRT file: {srt_path}")

        log_and_print(f"Parsing SRT file...")
        segments = parse_srt(str(srt_path))
        text = " ".join(segment.text for segment in segments).strip()
        log_and_print(f"Parsed {len(segments)} segments from SRT")
        log_and_print(f"Total text length: {len(text)} characters")

        return TranscriptionResult(srt_path=srt_path, segments=segments, text=text)

    def find_latest_srt(self, start_ts: float, source_path: Path) -> Path:
        log_and_print(f"find_latest_srt called with start_ts={start_ts}")
        log_and_print(f"Output dir: {self.output_dir}")

        if not self.output_dir.exists():
            log_and_print(f"Output directory not found: {self.output_dir}", "ERROR")
            raise TranscriptionError(f"Output directory not found: {self.output_dir}")

        base = source_path.stem.lower()
        log_and_print(f"Looking for SRT files in: {self.output_dir}")
        log_and_print(f"Base name filter: {base}")

        # Wait a moment for aTrain to fully write files
        log_and_print(f"Waiting 1 second for files to be written...")
        time.sleep(1)
        log_and_print(f"Resume search after wait")

        log_and_print(f"Current time: {time.time()}")
        log_and_print(f"Searching for files modified after: {start_ts - 2}")

        candidates = []
        all_srt_files = list(self.output_dir.rglob("*.srt"))
        log_and_print(f"Total SRT files found in tree: {len(all_srt_files)}")

        for path in all_srt_files:
            log_and_print(f"Checking: {path}")
            try:
                mtime = path.stat().st_mtime
                log_and_print(f"  mtime: {mtime} (is >= {start_ts - 2}? {mtime >= start_ts - 2})")
            except OSError as e:
                log_and_print(f"Could not stat {path}: {e}", "WARNING")
                continue
            if mtime >= start_ts - 2:
                candidates.append((mtime, path))
                log_and_print(f"✓ Found SRT candidate: {path.parent.name}/{path.name}")

        if not candidates:
            log_and_print("No SRT output found from aTrain_core", "ERROR")
            log_and_print(f"Checked directory: {self.output_dir}", "ERROR")
            log_and_print(f"Start time: {start_ts}, current time: {time.time()}", "ERROR")
            # List what SRT files exist (for debugging)
            all_srts = list(self.output_dir.rglob("*.srt"))
            if all_srts:
                log_and_print(f"Existing SRT files in directory: {len(all_srts)}", "DEBUG")
                for srt in sorted(all_srts)[-3:]:
                    log_and_print(f"  - {srt.parent.name}/{srt.name} (mtime={srt.stat().st_mtime})", "DEBUG")
            raise TranscriptionError("No SRT output found from aTrain_core")

        preferred = [item for item in candidates if base in item[1].name.lower()]
        if preferred:
            log_and_print(f"Filtered to preferred candidates: {len(preferred)}", "DEBUG")
            candidates = preferred

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[0][1]
        log_and_print(f"Selected SRT file: {selected}")
        return selected

    def _build_command(
        self,
        source_path: Path,
        model: str,
        language: str,
        device: str,
        compute_type: str,
        speaker_detection: bool,
        num_speakers: str,
        test_mode: bool,
    ) -> str:
        resolved_language = self._normalize_language(language)
        safe_source = shlex.quote(str(source_path))
        cmd = (
            f"source {shlex.quote(str(self.conda_sh))} && "
            f"conda activate {shlex.quote(self.conda_env)} && "
            f"aTrain_core transcribe {safe_source} --model {model} "
            f"--language {resolved_language} --device {device} --compute_type {compute_type}"
        )
        if speaker_detection:
            cmd += " --speaker_detection"
            if str(num_speakers) != "auto":
                cmd += f" --num_speakers {num_speakers}"
        if test_mode:
            cmd += " --test"
        return cmd

    @staticmethod
    def _normalize_language(language: str) -> str:
        language = (language or "").strip().lower()
        if language in {"de-de", "de-ch"}:
            return "de"
        if language in {"auto", "auto-detect"}:
            return "auto"
        return language or "auto"
