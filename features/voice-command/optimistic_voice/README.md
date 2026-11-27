# Optimistic – Voice-Controlled Smart Glasses Assistant

End-to-end offline-capable voice assistant for **Optimistic Smart Glasses**, built for **Raspberry Pi** (Mode 1: local Pi desktop control) and **Windows Desktop** (Mode 2: local Windows or authenticated Pi-to-Windows companion control).

## Features

- **Mandatory Startup Sequence**: Speaks `"Welcome back Optimistic innovators"` followed by `"I am ready. How can I help you?"` before activating the microphone listener, while displaying the greeting on the Optimistic HUD.
- **Conversational Intelligence (Multi-Provider LLM)**: Answers any general knowledge question or conversation using **xAI Grok**, **Google Gemini**, **Groq LPU**, **Ollama** (100% offline), or **OpenAI**, with intelligent speech sanitization for natural voice answers.
- **Instant Speech Interruption (Barge-In)**: Saying `"OptiMistic stop"`, `"Stop talking"`, or `"Be quiet"` purges audio queues immediately (<50ms) using low-level SAPI/hardware abort hooks.
- **Bluetooth & Multi-Sink Audio**: Automatic detection and priority routing for Bluetooth headsets and glasses audio output.
- **Offline Speech-to-Text**: Supports **Vosk** (`vosk-model-small-en-us-0.15`, lightweight for Raspberry Pi) and **faster-whisper**, with adaptive Voice Activity Detection (VAD), DC-offset/noise gating, and duplicate command filtering.
- **Offline Text-to-Speech**: Sequential non-overlapping speech queue supporting **Windows SAPI / Piper TTS**, **pyttsx3**, and **eSpeak NG**.
- **Natural Language Command Understanding**:
  - **Websites**: Open Google, YouTube, Gmail, Google Maps, ChatGPT, Wikipedia, Instagram, Facebook, or explicit URLs.
  - **Exact Text Typing**: `"Type hello world"`, `"Type YouTube.com"` — preserves exact spelling without silent correction into Notepad or any active window.
  - **Applications**: Open Notepad/text editor, Calculator, File Manager, Browser, and Terminal (with explicit confirmation).
  - **YouTube Automation (Multi-Tier Playwright + OS Browser Control)**:
    - Search for videos (`"search youtube for chill lofi music"` or `"type lofi in youtube search bar"`)
    - Select videos by index (`"select the first video"`)
    - Control playback (`"OptiMistic pause the video"`, `"play the video"`, `"resume video"`) across Playwright and external desktop browsers (Chrome, Edge, Brave).
    - Window management (`"OptiMistic close this window"` specifically targets and closes the YouTube window/tab).
  - **Desktop Control**: Minimize/maximize/close window, switch application, show desktop, copy/paste/select all, Enter/Escape/Backspace, and confirmation-gated sensitive actions.
- **Listening Modes**: Continuous listening (default) or Wake-Word mode (`--wake-word`, `"Hey Optimistic"`).
- **Optimistic Smart-Glasses HUD & Web Dashboard**: Minimal high-contrast Tkinter display (`display/display_manager.py`) plus an interactive Web HUD testing dashboard (`web_app.py`).

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
