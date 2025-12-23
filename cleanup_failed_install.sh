#!/bin/bash
set -euo pipefail

echo "Cleaning failed AI VideoClipper installs..."

# Remove failed env if it exists
if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "ai-videoclipper"; then
    echo "Removing conda env: ai-videoclipper"
    conda env remove -n ai-videoclipper -y
  fi

  echo "Cleaning conda caches..."
  conda clean -a -y
fi

# Remove partial downloads in conda pkgs cache
if [ -d "$HOME/miniconda3/pkgs" ]; then
  echo "Removing partial conda packages..."
  rm -f "$HOME/miniconda3/pkgs"/*.conda.partial || true
  rm -f "$HOME/miniconda3/pkgs"/*.tar.bz2.part || true
  rm -f "$HOME/miniconda3/pkgs"/*.tmp || true
fi

# Remove any leftover env directory if present
if [ -d "$HOME/miniconda3/envs/ai-videoclipper" ]; then
  echo "Removing env directory: ~/miniconda3/envs/ai-videoclipper"
  rm -rf "$HOME/miniconda3/envs/ai-videoclipper"
fi

echo "Cleanup complete. You can rerun ./install.sh"
