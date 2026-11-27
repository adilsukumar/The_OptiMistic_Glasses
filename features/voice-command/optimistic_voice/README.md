# Optimistic – Voice-Controlled Smart Glasses Assistant

End-to-end offline-capable voice assistant for **Optimistic Smart Glasses**, built for **Raspberry Pi** (Mode 1: local Pi desktop control) and **Windows Desktop** (Mode 2: local Windows or authenticated Pi-to-Windows companion control).

## Features

- **Mandatory Startup Sequence**: Speaks `"Welcome back Optimistic innovators"` followed by `"I am ready. How can I help you?"` before activating the microphone listener, while displaying the greeting on the Optimistic HUD.
- **Offline Speech-to-Text**: Supports **Vosk** (`vosk-model-small-en-us-0.15`, lightweight for Raspberry Pi) and **faster-whisper**, with adaptive Voice Activity Detection (VAD), DC-offset/noise gating, and duplicate command filtering.
- **Offline Text-to-Speech**: Sequential non-overlapping speech queue supporting **Piper TTS**, **pyttsx3**, and **eSpeak NG**.
- **Natural Language Command Understanding**:
  - **Websites**: Open Google, YouTube, Gmail, Google Maps, ChatGPT, Wikipedia, Instagram, Facebook, or explicit URLs.
  - **Exact Text Typing**: `"Type hello world"`, `"Type YouTube.com"` — preserves exact spelling without silent correction.
  - **Applications**: Open Notepad/text editor, Calculator, File Manager, Browser, and Terminal (with explicit confirmation).
  - **YouTube Browser Automation (Playwright)**: Open YouTube, search for videos, select 1st/2nd/3rd video, play, pause, resume, go back, scroll up/down, next page, volume up/down, mute/unmute.
  - **Desktop Control**: Minimize/maximize/close window, switch application, show desktop, copy/paste/select all, Enter/Escape/Backspace, and confirmation-gated sensitive actions.
- **Listening Modes**: Continuous listening (default) or Wake-Word mode (`--wake-word`, `"Hey Optimistic"`).
- **Optimistic Smart-Glasses HUD**: Minimal high-contrast Tkinter display (`display/display_manager.py`) with callback hooks for hardware display integration.

## Quick Start (Windows Local)

```powershell
cd optimistic_voice
.\scripts\setup_windows.ps1
..\.venv\Scripts\python.exe main.py
```

For interactive text-only testing (without speaking into a mic):
```powershell
..\.venv\Scripts\python.exe main.py --text-only
```

## Quick Start (Raspberry Pi)

```bash
cd optimistic_voice
chmod +x scripts/setup_pi.sh scripts/start_optimistic.sh
./scripts/setup_pi.sh
./scripts/start_optimistic.sh
```

## Mode 2: Raspberry Pi Controlling a Windows Desktop

1. **On Windows** (set a shared secret token and start the companion server):
   ```powershell
   $env:OPTIMISTIC_AUTH_TOKEN="your-strong-shared-secret"
   ..\.venv\Scripts\python.exe integration\windows_client.py
   ```
2. **On Raspberry Pi** (point to the Windows PC's local IP):
   ```bash
   export OPTIMISTIC_MODE="remote_windows"
   export OPTIMISTIC_REMOTE_HOST="192.168.1.50"
   export OPTIMISTIC_REMOTE_PORT="8765"
   export OPTIMISTIC_AUTH_TOKEN="your-strong-shared-secret"
   ./scripts/start_optimistic.sh
   ```

All Pi-to-Windows commands are signed with HMAC-SHA256 + timestamp freshness verification and restricted to whitelisted `IntentType` actions (no arbitrary shell execution).

## Running Tests

```powershell
.\.venv\Scripts\python.exe -m pytest optimistic_voice/tests -v
```

