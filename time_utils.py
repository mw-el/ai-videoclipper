from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def format_timestamp_ms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def parse_timestamp(value: str) -> float:
    if not value:
        raise ValueError("Empty timestamp")
    value = value.strip()
    if "," in value:
        main, ms = value.split(",", 1)
    elif "." in value:
        main, ms = value.split(".", 1)
    else:
        main, ms = value, "0"
    parts = main.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {value}")
    hours, minutes, secs = [int(part) for part in parts]
    ms = int(ms.ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + secs + (ms / 1000.0)
