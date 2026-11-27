"""Tests for TextToSpeechEngine queue, ordering, and configuration."""

from __future__ import annotations

from optimistic_voice.audio.text_to_speech import TextToSpeechEngine


def test_tts_startup_greeting_order_and_queue() -> None:
    tts = TextToSpeechEngine(engine="mock", rate=180, volume=0.9)
    tts.speak("Welcome back Optimistic innovators", block=True)
    tts.speak("I am ready. How can I help you?", block=True)
    tts.speak("Opening Google.", block=True)

    assert tts.spoken_history == [
        "Welcome back Optimistic innovators",
        "I am ready. How can I help you?",
        "Opening Google.",
    ]
    tts.shutdown()


def test_tts_rate_and_volume_settings() -> None:
    tts = TextToSpeechEngine(engine="mock", rate=160, volume=0.75)
    tts.set_rate(200)
    tts.set_volume(0.5)
    assert tts.rate == 200
    assert tts.volume == 0.5
    tts.shutdown()


def test_tts_stop_clears_queue() -> None:
    tts = TextToSpeechEngine(engine="mock", rate=180, volume=0.9)
    tts.speak("First phrase", block=False)
    tts.speak("Second phrase that will be cancelled", block=False)
    tts.stop()
    assert tts._queue.empty()
    tts.shutdown()
