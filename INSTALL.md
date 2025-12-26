# Install Guide

## System Requirements
- Ubuntu 24.04 or similar
- Miniconda
- FFmpeg (`ffmpeg`)
- libmagic (`libmagic1`)
- libsndfile (`libsndfile1`)
- Git + GitHub CLI (`git`, `gh`)

## Conda Environments
This project uses three separate environments:

1) `ai-videoclipper` (main app)
   - Runs the PyQt6 UI, scene detection, and export.
   - Created by `install.sh`.
   - Pip dependencies tracked in `requirements.txt`.

2) `atrain` (transcription)
   - External aTrainCore environment, provided by `_AA_aTrainCore`.
   - Required for transcription only.

3) `ai-videoclipper-whisperx` (alignment)
   - Runs WhisperX alignment in isolation to avoid AV/Torch conflicts.
   - Manual setup below.

## Install Main App
```bash
./install.sh
```
`install.sh` creates the conda env and installs pip deps from `requirements.txt`.

Optional face analysis (run inside `ai-videoclipper`):
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-videoclipper
pip install mediapipe opencv-python
```

## Install WhisperX (separate env)
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n ai-videoclipper-whisperx python=3.10
conda activate ai-videoclipper-whisperx
pip install whisperx
```

Alignment runs via:
```bash
conda run -n ai-videoclipper-whisperx python whisperx_align.py --help
```

## Verify aTrainCore
Make sure `aTrain_core` is available in the `atrain` environment:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate atrain
command -v aTrain_core
```

## Run
```bash
./run.sh
```

On startup, the app logs missing required/optional modules under `[DEPS]`.
