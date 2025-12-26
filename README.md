# AI VideoClipper

Local PyQt6 desktop app for creating short clips from a long video. Transcription uses aTrainCore
(atrain env).

## Requirements
- Ubuntu 24.04 or similar
- Miniconda
- aTrainCore installed in the atrain conda environment
- FFmpeg

## Environments
- `ai-videoclipper`: main app + UI, scene detection, export.
- `atrain`: aTrainCore CLI for transcription (external project).
- `ai-videoclipper-whisperx`: WhisperX alignment (kept separate to avoid AV/Torch conflicts).

Main app pip dependencies are listed in `requirements.txt` (PyQt6, faster-whisper, smartcut, numpy).

## Setup
```bash
./install.sh
```

This creates the ai-videoclipper conda environment and a GitHub repo.
Pip dependencies are installed from `requirements.txt`.

## Run
```bash
./run.sh
```

## Usage
1. Select a video file.
2. The app runs aTrain_core transcription and loads the newest SRT from
   `~/Documents/aTrain/transcriptions`.
3. Scene detection runs on demand (local candidate generation + Claude scoring).
4. Use the preview player and SRT viewer to adjust start/end markers.
5. Export one clip or all clips to `output/clips`.

## Output
Default output directory: `output/clips/`

## Screenshots
Add screenshots here if needed.

## Notes

- Transcription uses the external aTrainCore CLI from `_AA_aTrainCore`.
- If `aTrain_core` is missing, the app will show an error.
- Optional: `mediapipe` + `opencv-python` enable face expressivity features.
- Optional: `whisperx` enables precision alignment for final cuts (run via `ai-videoclipper-whisperx`).
- Face landmarker model is auto-downloaded to `assets/mediapipe/face_landmarker.task`.
- On startup, missing required/optional Python modules are logged under `[DEPS]`.

## CI/CD Status

- **GitHub Actions CI is currently disabled** (`.github/workflows/ci.yml.disabled`)
- Tests need to be fixed before re-enabling CI
- To re-enable: rename `ci.yml.disabled` back to `ci.yml`
