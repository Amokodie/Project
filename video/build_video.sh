#!/usr/bin/env bash
# Build output/engine_rul_video.mp4 from video/engine_rul_video.md via deck2video.
# Run this from the PROJECT ROOT in Git Bash.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECK_DIR="$PROJECT_ROOT/deck2video"
VENV_ACTIVATE="$DECK_DIR/.venv/Scripts/activate"   # Windows layout
[ -f "$VENV_ACTIVATE" ] || VENV_ACTIVATE="$DECK_DIR/.venv/bin/activate"  # Unix fallback

if [ ! -d "$DECK_DIR" ]; then
  echo "ERROR: deck2video/ not found. Clone it first:"
  echo "  git clone https://github.com/pjdoland/deck2video"
  exit 1
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "ERROR: deck2video venv not found. Install with:"
  echo "  cd deck2video && source setup.sh"
  exit 1
fi

# Activate if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

# Ensure ffmpeg is reachable (winget added it to PATH but only for new shells)
FFMPEG_WINGET="$LOCALAPPDATA/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin"
if ! command -v ffmpeg >/dev/null 2>&1 && [ -d "$FFMPEG_WINGET" ]; then
  export PATH="$FFMPEG_WINGET:$PATH"
fi

mkdir -p "$PROJECT_ROOT/output"

# deck2video must be run from its own repo root so `python -m deck2video`
# resolves the package (pyproject.toml doesn't install it as a distribution).
cd "$DECK_DIR"

python -m deck2video \
  "$PROJECT_ROOT/video/engine_rul_video.md" \
  --output "$PROJECT_ROOT/output/engine_rul_video.mp4"

echo
echo "Done. Video written to: $PROJECT_ROOT/output/engine_rul_video.mp4"
