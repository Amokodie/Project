#!/usr/bin/env bash
# Build output/engine_rul_video.mp4 from video/engine_rul_video.md via deck2video.
# Run this from the PROJECT ROOT in Git Bash, after `source deck2video/setup.sh`.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d "deck2video" ]; then
  echo "ERROR: deck2video/ not found. Clone it first:"
  echo "  git clone https://github.com/pjdoland/deck2video"
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: no virtualenv active. Run:"
  echo "  source deck2video/setup.sh"
  exit 1
fi

mkdir -p output

python -m deck2video \
  video/engine_rul_video.md \
  --output output/engine_rul_video.mp4

echo
echo "Done. Video written to: output/engine_rul_video.mp4"
