from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import time
from typing import Callable, List, Optional

from srt_utils import SrtSegment, parse_srt


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
        if not source_path.exists():
            raise TranscriptionError(f"File not found: {source_path}")
        if not self.conda_sh.exists():
            raise TranscriptionError(
                f"Conda activation script not found: {self.conda_sh}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(
            source_path,
            model=model,
            language=language,
            device=device,
            compute_type=compute_type,
            speaker_detection=speaker_detection,
            num_speakers=num_speakers,
            test_mode=test_mode,
        )
        start_ts = time.time()
        result = self.runner(["bash", "-lc", cmd], capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise TranscriptionError(
                f"aTrain_core failed with code {result.returncode}: {stderr}"
            )

        srt_path = self.find_latest_srt(start_ts, source_path)
        segments = parse_srt(str(srt_path))
        text = " ".join(segment.text for segment in segments).strip()
        return TranscriptionResult(srt_path=srt_path, segments=segments, text=text)

    def find_latest_srt(self, start_ts: float, source_path: Path) -> Path:
        if not self.output_dir.exists():
            raise TranscriptionError(f"Output directory not found: {self.output_dir}")
        base = source_path.stem.lower()
        candidates = []
        for path in self.output_dir.rglob("*.srt"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= start_ts - 2:
                candidates.append((mtime, path))
        if not candidates:
            raise TranscriptionError("No SRT output found from aTrain_core")

        preferred = [item for item in candidates if base in item[1].name.lower()]
        if preferred:
            candidates = preferred
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

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
