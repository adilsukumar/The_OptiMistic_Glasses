#!/usr/bin/env bash
set -euo pipefail

echo "=== Optimistic Smart Glasses: Raspberry Pi Setup ==="

sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-tk \
  portaudio19-dev libasound2-dev espeak-ng \
  xdotool wmctrl chromium

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium || true

mkdir -p models/speech_models
if [ ! -d "models/speech_models/vosk-model-small-en-us-0.15" ]; then
  echo "Downloading Vosk small English model..."
  curl -L -o vosk.zip "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
  unzip -q vosk.zip -d models/speech_models
  rm -f vosk.zip
fi

echo "Setup complete! Run ./scripts/start_optimistic.sh to start."

