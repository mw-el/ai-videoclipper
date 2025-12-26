from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

if os.environ.get("AI_VIDECLIPPER_FORCE_CPU", "1") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import whisperx


def _patch_torch_load() -> None:
    try:
        from typing import Any
        from collections import defaultdict, OrderedDict
        from omegaconf import DictConfig  # type: ignore
        from omegaconf.base import ContainerMetadata, Metadata  # type: ignore
        from omegaconf.listconfig import ListConfig  # type: ignore
        from omegaconf.nodes import AnyNode  # type: ignore
        from pyannote.audio.core.model import Introspection  # type: ignore
        from pyannote.audio.core.task import Specifications, Problem, Resolution  # type: ignore
        from torch.torch_version import TorchVersion  # type: ignore
        torch.serialization.add_safe_globals([
            ListConfig,
            DictConfig,
            ContainerMetadata,
            Any,
            list,
            dict,
            tuple,
            set,
            defaultdict,
            OrderedDict,
            int,
            float,
            str,
            bool,
            AnyNode,
            Metadata,
            TorchVersion,
            Introspection,
            Specifications,
            Problem,
            Resolution,
        ])
    except Exception:
        pass

    original_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_compat  # type: ignore[assignment]


def _load_segments(payload_path: Path) -> dict:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if "segments" not in payload:
        raise ValueError("segments payload missing")
    return payload


def _align_segments(
    audio_path: Path,
    segments: List[dict],
    language: str | None,
    model_name: str,
) -> dict:
    audio = whisperx.load_audio(str(audio_path))

    def run_alignment(device: str, compute_type: str) -> dict:
        if language:
            detected_language = language
        else:
            model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
            result = model.transcribe(audio)
            detected_language = result.get("language")
            if not detected_language:
                raise RuntimeError("Failed to detect language for alignment")

        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device,
        )
        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        return {
            "aligned": aligned,
            "language": detected_language,
        }

    use_cuda = torch.cuda.is_available() and torch.backends.cudnn.is_available()
    device = "cuda" if use_cuda else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    try:
        result = run_alignment(device, compute_type)
    except RuntimeError as exc:
        message = str(exc).lower()
        if device == "cuda" and ("cudnn" in message or "cannot load symbol" in message):
            result = run_alignment("cpu", "int8")
        else:
            raise

    aligned = result["aligned"]
    detected_language = result["language"]
    words = []
    for seg in aligned.get("segments", []):
        words.extend(seg.get("words", []))
    if not words:
        raise RuntimeError("No aligned words returned")
    first_word = min(words, key=lambda w: w["start"])
    last_word = max(words, key=lambda w: w["end"])
    return {
        "first_word_start": float(first_word["start"]),
        "last_word_end": float(last_word["end"]),
        "word_count": len(words),
        "language": detected_language,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperX alignment helper")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", default=None)
    parser.add_argument("--model", default="large-v2")
    args = parser.parse_args()
    _patch_torch_load()

    payload = _load_segments(args.segments)
    segments = payload.get("segments", [])
    if not segments:
        raise RuntimeError("No segments provided for alignment")

    alignment = _align_segments(args.audio, segments, args.language, args.model)
    args.output.write_text(json.dumps(alignment, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
