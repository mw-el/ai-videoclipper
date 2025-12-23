# AI VideoClipper

Local PyQt6 desktop app for creating short clips from a long video. Transcription uses aTrainCore
(atrain env) and clip selection uses ClipsAI.

## Requirements
- Ubuntu 24.04 or similar
- Miniconda
- aTrainCore installed in the atrain conda environment
- FFmpeg

## Setup
```bash
./install.sh
```

This creates the ai-videoclipper conda environment and a GitHub repo.

## Run
```bash
./run.sh
```

## Usage
1. Select a video file.
2. The app runs aTrain_core transcription and loads the newest SRT from
   `~/Documents/aTrain/transcriptions`.
3. ClipFinder proposes 5 to 6 clips.
4. Use the preview player and SRT viewer to adjust start/end markers.
5. Export one clip or all clips to `output/clips`.

## Output
Default output directory: `output/clips/`

## Screenshots
Add screenshots here if needed.

## Notes
- Transcription uses the external aTrainCore CLI from `_AA_aTrainCore`.
- If `aTrain_core` is missing, the app will show an error.
