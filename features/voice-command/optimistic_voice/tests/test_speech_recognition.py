"""Tests for VAD, WakeWordDetector, and SpeechRecognizer (including real Vosk model)."""

from __future__ import annotations

import numpy as np

from optimistic_voice.audio.speech_recognition import SpeechRecognizer
from optimistic_voice.audio.voice_activity import VoiceActivityDetector
from optimistic_voice.audio.wake_word import WakeWordDetector
from optimistic_voice.config import OptimisticConfig


def test_vad_detects_speech_then_silence() -> None:
    vad = VoiceActivityDetector(
        sample_rate=16000,
        energy_threshold=0.02,
        speech_pad_chunks=1,
        silence_chunks=2,
        chunk_samples=1600,
    )
    silence = np.zeros(1600, dtype=np.float32)
    speech = 0.2 * np.sin(np.linspace(0, 40, 1600, dtype=np.float32))

    assert vad.process_chunk(silence) is None
    assert vad.process_chunk(speech) is None
    assert vad.process_chunk(speech) is None
    assert vad.process_chunk(silence) is None
    utterance = vad.process_chunk(silence)
    assert utterance is not None
    assert len(utterance) >= 1600 * 3


def test_wake_word_continuous_vs_gated() -> None:
    continuous = WakeWordDetector(enabled=False)
    should_run, cleaned, triggered = continuous.process_transcript("open google")
    assert should_run is True
    assert cleaned == "open google"
    assert triggered is False

    gated = WakeWordDetector(enabled=True, wake_phrases=("hey optimistic",))
    should_run, _, triggered = gated.process_transcript("open google")
    assert should_run is False
    assert triggered is False

    should_run, cleaned, triggered = gated.process_transcript("Hey Optimistic")
    assert should_run is False
    assert triggered is True

    should_run, cleaned, triggered = gated.process_transcript("Hey Optimistic open youtube")
    assert should_run is True
    assert cleaned == "open youtube"
    assert triggered is True


def test_vosk_model_loads_and_runs_offline() -> None:
    cfg = OptimisticConfig()
    assert cfg.vosk_model_path.exists(), f"Missing Vosk model at {cfg.vosk_model_path}"
    stt = SpeechRecognizer(
        engine="vosk",
        sample_rate=16000,
        vosk_model_path=cfg.vosk_model_path,
    )
    assert stt.active_engine == "vosk"
    # Verify transcribing silent/tone audio returns string cleanly without crashing
    audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(audio)
    assert isinstance(result, str)
