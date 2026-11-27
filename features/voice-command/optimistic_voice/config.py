"""Configuration for Optimistic Voice-Controlled Smart Glasses Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "speech_models"


def load_env_file() -> None:
    """Load key-value pairs from .env file into os.environ if not already set."""
    search_paths = [
        BASE_DIR / ".env",
        BASE_DIR.parent / ".env",
        BASE_DIR.parent.parent / ".env",
    ]
    for p in search_paths:
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass
            break


load_env_file()

STARTUP_GREETING = "Welcome back Optimistic innovators"
READY_PROMPT = "I am ready. How can I help you?"
WAKE_RESPONSE = "Yes, I'm listening."


@dataclass
class OptimisticConfig:
    """Runtime configuration for Optimistic voice assistant."""

    # Operating mode: "local" (control current OS) or "remote_windows" (Pi -> Windows client)
    execution_mode: str = os.getenv("OPTIMISTIC_MODE", "local")

    # Startup
    startup_greeting: str = STARTUP_GREETING
    ready_prompt: str = READY_PROMPT

    # Listening & Wake Word
    wake_word_enabled: bool = os.getenv("OPTIMISTIC_WAKE_WORD", "true").lower() == "true"
    wake_phrases: tuple[str, ...] = ("optimistic", "hey optimistic", "ok optimistic", "hi optimistic", "hello optimistic")
    wake_response: str = WAKE_RESPONSE
    wake_session_timeout_s: float = 15.0
    muted: bool = False

    # Audio & VAD
    sample_rate: int = 16000
    channels: int = 1
    chunk_samples: int = 1600  # 100 ms frames at 16 kHz
    input_device: int | str | None = None
    vad_energy_threshold: float = float(os.getenv("OPTIMISTIC_VAD_THRESHOLD", "0.0030"))
    vad_speech_pad_chunks: int = int(os.getenv("OPTIMISTIC_VAD_PAD", "6"))  # 600 ms pre-roll
    vad_silence_chunks: int = int(os.getenv("OPTIMISTIC_VAD_SILENCE", "7"))  # 700 ms silence ends utterance (fast & snappy)
    max_utterance_s: float = float(os.getenv("OPTIMISTIC_MAX_UTTERANCE", "15.0"))
    noise_reduction_enabled: bool = os.getenv("OPTIMISTIC_NOISE_REDUCTION", "true").lower() == "true"
    live_partial_feedback: bool = os.getenv("OPTIMISTIC_LIVE_FEEDBACK", "true").lower() == "true"

    # Speech-to-Text
    stt_engine: str = os.getenv("OPTIMISTIC_STT_ENGINE", "auto")  # auto | faster_whisper | vosk | mock
    stt_min_confidence: float = float(os.getenv("OPTIMISTIC_STT_CONFIDENCE", "0.45"))
    vosk_model_path: Path = Path(
        os.getenv("OPTIMISTIC_VOSK_MODEL", str(MODELS_DIR / "vosk-model-small-en-us-0.15"))
    )
    whisper_model_size: str = os.getenv("OPTIMISTIC_WHISPER_MODEL", "tiny.en")
    whisper_device: str = os.getenv("OPTIMISTIC_WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("OPTIMISTIC_WHISPER_COMPUTE", "int8")
    duplicate_command_cooldown_s: float = 2.0

    # Text-to-Speech
    tts_engine: str = os.getenv("OPTIMISTIC_TTS_ENGINE", "auto")  # auto | piper | pyttsx3 | espeak | mock
    piper_model_path: Path = Path(
        os.getenv("OPTIMISTIC_PIPER_MODEL", str(MODELS_DIR / "en_US-lessac-medium.onnx"))
    )
    tts_rate: int = int(os.getenv("OPTIMISTIC_TTS_RATE", "175"))
    tts_volume: float = float(os.getenv("OPTIMISTIC_TTS_VOLUME", "0.95"))
    tts_output_device: str | int | None = os.getenv("OPTIMISTIC_TTS_OUTPUT", "auto")  # auto | bluetooth | laptop | index

    # Display
    display_mode: str = os.getenv("OPTIMISTIC_DISPLAY_MODE", "auto")  # auto | tkinter | console | headless
    display_width: int = 480
    display_height: int = 240

    # Browser & Automation
    browser_headless: bool = os.getenv("OPTIMISTIC_BROWSER_HEADLESS", "false").lower() == "true"
    browser_channel: str | None = os.getenv("OPTIMISTIC_BROWSER_CHANNEL", None)  # e.g., "chrome", "msedge"
    typing_interval_s: float = 0.015

    # LLM / Conversational AI
    llm_enabled: bool = os.getenv("OPTIMISTIC_LLM_ENABLED", "true").lower() == "true"
    llm_provider: str = os.getenv("OPTIMISTIC_LLM_PROVIDER", "auto")  # auto | grok | gemini | groq | ollama | mock
    llm_model: str = os.getenv("OPTIMISTIC_LLM_MODEL", "")  # empty uses provider default
    xai_api_key: str = os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", ""))
    grok_model: str = os.getenv("GROK_MODEL", "grok-2-mini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    # Pi <-> Windows Companion Integration
    remote_host: str = os.getenv("OPTIMISTIC_REMOTE_HOST", "127.0.0.1")
    remote_port: int = int(os.getenv("OPTIMISTIC_REMOTE_PORT", "8765"))
    auth_token: str = os.getenv("OPTIMISTIC_AUTH_TOKEN", "")
    pi_server_host: str = os.getenv("OPTIMISTIC_PI_HOST", "0.0.0.0")
    pi_server_port: int = int(os.getenv("OPTIMISTIC_PI_PORT", "8766"))

    # Known websites mapping
    known_websites: dict[str, str] = field(
        default_factory=lambda: {
            "google": "https://www.google.com",
            "goggle": "https://www.google.com",
            "googol": "https://www.google.com",
            "goolge": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "you tube": "https://www.youtube.com",
            "u tube": "https://www.youtube.com",
            "utube": "https://www.youtube.com",
            "ytube": "https://www.youtube.com",
            "yutube": "https://www.youtube.com",
            "gmail": "https://mail.google.com",
            "g mail": "https://mail.google.com",
            "gee mail": "https://mail.google.com",
            "google mail": "https://mail.google.com",
            "google maps": "https://maps.google.com",
            "maps": "https://maps.google.com",
            "chatgpt": "https://chatgpt.com",
            "chat gpt": "https://chatgpt.com",
            "wikipedia": "https://www.wikipedia.org",
            "wiki": "https://www.wikipedia.org",
            "instagram": "https://www.instagram.com",
            "insta": "https://www.instagram.com",
            "facebook": "https://www.facebook.com",
            "face book": "https://www.facebook.com",
            "fb": "https://www.facebook.com",
            "github": "https://github.com",
            "git hub": "https://github.com",
            "reddit": "https://www.reddit.com",
            "linkedin": "https://www.linkedin.com",
            "linked in": "https://www.linkedin.com",
            "twitter": "https://x.com",
            "x": "https://x.com",
            "netflix": "https://www.netflix.com",
            "net flix": "https://www.netflix.com",
            "amazon": "https://www.amazon.com",
            "stackoverflow": "https://stackoverflow.com",
            "stack overflow": "https://stackoverflow.com",
        }
    )
