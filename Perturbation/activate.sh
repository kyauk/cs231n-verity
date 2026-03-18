#!/usr/bin/env bash
# Source this file to activate the venv with all required env vars:
#   source activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/venv/bin/activate"

# ImageMagick (required by wand / LIBERO rendering)
export MAGICK_HOME=/opt/homebrew/opt/imagemagick
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/imagemagick/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}

echo "Activated venv with ImageMagick env vars set."
