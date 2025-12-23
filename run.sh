#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-videoclipper
python clips_editor.py
