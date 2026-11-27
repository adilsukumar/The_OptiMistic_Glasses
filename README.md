# The OptiMistic Glasses

Smart glasses assistive AI platform powered by Raspberry Pi 3.

## Features

| Feature | Directory | Description |
| :--- | :--- | :--- |
| **Sign Language Interpreter** | [`features/sign-language-interpreter/optimistic_isl`](features/sign-language-interpreter/optimistic_isl) | Real-time, offline Indian Sign Language (ISL) recognition using MediaPipe 21-point hand landmarks and a quantized TensorFlow Lite temporal model, streaming structured JSON over a local WebSocket interface to the glasses HUD. |
| **Voice Command** | [`features/voice-command/optimistic_voice`](features/voice-command/optimistic_voice) | End-to-end offline-capable voice-controlled assistant for Raspberry Pi and Windows companion desktops using Vosk STT, Piper/pyttsx3/eSpeak TTS, Playwright YouTube/browser automation, desktop control, and a minimal smart-glasses HUD. |
