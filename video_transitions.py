"""
Video transition utilities for combining clips with Dip to Black effect.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
import logging

logger = logging.getLogger("ai_videoclipper")


def create_clip_with_dip_to_black(
    source_video: Path,
    hook_start: float,
    hook_end: float,
    main_start: float,
    main_end: float,
    output_path: Path,
    freeze_duration: float = 0.5,
    black_duration: float = 0.3,
) -> None:
    """
    Creates final clip with Hook + Dip to Black transition + Main Clip.

    Structure:
    [Hook Video] → [Freeze last frame 0.5s + Fade to Black] →
    [Black 0.3s] →
    [Fade from Black + Freeze first frame 0.5s] → [Main Video]

    Args:
        source_video: Source video file
        hook_start: Hook start time in seconds
        hook_end: Hook end time in seconds
        main_start: Main clip start time in seconds
        main_end: Main clip end time in seconds
        output_path: Output file path
        freeze_duration: Duration to freeze frames during fade (default 0.5s)
        black_duration: Duration of black screen between clips (default 0.3s)
    """
    logger.info(
        f"[TRANSITION] Creating clip with dip-to-black: "
        f"Hook {hook_start:.1f}-{hook_end:.1f}s, "
        f"Main {main_start:.1f}-{main_end:.1f}s"
    )

    # Calculate durations
    hook_duration = hook_end - hook_start
    main_duration = main_end - main_start
    fade_duration = freeze_duration  # Fade happens during freeze

    # FFmpeg filter complex for:
    # 1. Extract hook segment
    # 2. Freeze last frame + fade to black
    # 3. Black screen
    # 4. Fade from black + freeze first frame
    # 5. Extract main segment
    # 6. Concatenate everything

    filter_complex = (
        # Extract hook segment
        f"[0:v]trim=start={hook_start}:end={hook_end},setpts=PTS-STARTPTS[hook_v];"
        f"[0:a]atrim=start={hook_start}:end={hook_end},asetpts=PTS-STARTPTS[hook_a];"

        # Freeze last frame of hook for fade duration
        f"[hook_v]tpad=stop_mode=clone:stop_duration={freeze_duration}[hook_freeze];"

        # Fade to black on frozen frame
        f"[hook_freeze]fade=t=out:st={hook_duration}:d={fade_duration}:color=black[hook_fade];"

        # Pad audio with silence for freeze duration
        f"[hook_a]apad=pad_dur={freeze_duration}[hook_a_pad];"

        # Create black screen
        f"color=c=black:s=1920x1080:d={black_duration}[black_v];"
        f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={black_duration}[black_a];"

        # Extract main segment
        f"[0:v]trim=start={main_start}:end={main_end},setpts=PTS-STARTPTS[main_v];"
        f"[0:a]atrim=start={main_start}:end={main_end},asetpts=PTS-STARTPTS[main_a];"

        # Freeze first frame of main for fade duration
        f"[main_v]tpad=start_mode=clone:start_duration={freeze_duration}[main_freeze];"

        # Fade from black on frozen frame
        f"[main_freeze]fade=t=in:st=0:d={fade_duration}:color=black[main_fade];"

        # Pad audio with silence at start for freeze duration
        f"anullsrc=channel_layout=stereo:sample_rate=48000:duration={freeze_duration}[silence];"
        f"[silence][main_a]concat=n=2:v=0:a=1[main_a_pad];"

        # Concatenate all parts: hook + black + main
        f"[hook_fade][hook_a_pad][black_v][black_a][main_fade][main_a_pad]concat=n=3:v=1:a=1[outv][outa]"
    )

    cmd = [
        "ffmpeg",
        "-i", str(source_video),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",  # Overwrite output
        str(output_path)
    ]

    logger.debug(f"[TRANSITION] FFmpeg command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        logger.info(f"[TRANSITION] ✓ Created clip with transition: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[TRANSITION] FFmpeg failed: {e.stderr}")
        raise RuntimeError(f"Video transition failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        logger.error(f"[TRANSITION] FFmpeg timeout after 10 minutes")
        raise RuntimeError("Video transition timed out")


def create_simple_clip(
    source_video: Path,
    start: float,
    end: float,
    output_path: Path,
) -> None:
    """
    Creates a simple clip without transitions (fallback when no hook).

    Args:
        source_video: Source video file
        start: Start time in seconds
        end: End time in seconds
        output_path: Output file path
    """
    logger.info(f"[CLIP] Creating simple clip: {start:.1f}-{end:.1f}s")

    cmd = [
        "ffmpeg",
        "-i", str(source_video),
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        logger.info(f"[CLIP] ✓ Created simple clip: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[CLIP] FFmpeg failed: {e.stderr}")
        raise RuntimeError(f"Clip creation failed: {e.stderr}")
