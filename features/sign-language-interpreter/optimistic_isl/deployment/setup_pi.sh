#!/usr/bin/env bash
set -euo pipefail

# Optimistic ISL Smart Glasses — Raspberry Pi 4 / Pi 5 Setup Script (64-bit Bookworm/Bullseye)

echo "=== Optimistic ISL Raspberry Pi Setup ==="

sudo apt-get update
sudo apt-get install -y \
    python3-venv \
    python3-pip \
    python3-opencv \
    libatlas-base-dev \
    libopenblas-dev \
    v4l-utils \
    curl

# Optional Picamera2 system package for official Raspberry Pi Camera Module
if apt-cache show python3-picamera2 >/dev/null 2>&1; then
    sudo apt-get install -y python3-picamera2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv_pi"

# --system-site-packages allows accessing apt-installed picamera2 cleanly
python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements_pi.txt"

# Ensure MediaPipe HandLandmarker asset is present locally for offline execution
MP_TASK="${SCRIPT_DIR}/hand_landmarker.task"
if [ ! -f "${MP_TASK}" ]; then
    echo "Downloading MediaPipe hand_landmarker.task for offline Pi execution..."
    curl -fsSL -o "${MP_TASK}" \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

echo ""
echo "Setup complete! Run headless smart-glasses inference with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python3 ${SCRIPT_DIR}/camera_inference.py --headless --ws-port 8765"
echo "Add --picamera2 if using the CSI Raspberry Pi Camera Module."
