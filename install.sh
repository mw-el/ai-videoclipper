#!/bin/bash
set -e

echo "AI VideoClipper Setup (Ubuntu 24.04)"

sudo apt update
sudo apt install -y wget git ffmpeg libmagic1 libsndfile1 gh

if [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "Conda activation script not found at ~/miniconda3/etc/profile.d/conda.sh"
  echo "Install Miniconda first and re-run this script."
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml || conda env update -f environment.yml
conda activate ai-videoclipper
pip install -r requirements.txt

if ! bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate atrain && command -v aTrain_core >/dev/null"; then
  echo "WARNING: aTrain_core not found. Install aTrainCore in the atrain env."
fi

echo ""
echo "Optional WhisperX (separate env to avoid AV/Torch conflicts):"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda create -y -n ai-videoclipper-whisperx python=3.10"
echo "  conda activate ai-videoclipper-whisperx"
echo "  pip install whisperx"
echo "  conda run -n ai-videoclipper-whisperx python whisperx_align.py --help"

if [ ! -d .git ]; then
  git init
  git checkout -b main
fi

git add .
git commit -m "Initial: AI VideoClipper App" || true

if ! git remote | grep -q "^origin$"; then
  gh repo create ai-videoclipper --public --source=. --remote=origin --push
else
  git push -u origin main
fi

echo "Repo ready: https://github.com/$(git config user.name)/ai-videoclipper"
echo "Run: ./run.sh"
