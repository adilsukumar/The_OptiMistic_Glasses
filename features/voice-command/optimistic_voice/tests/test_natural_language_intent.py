"""Tests for natural phrasing, conversational command understanding, and streaming feedback."""

from __future__ import annotations

import numpy as np
import pytest

from optimistic_voice.assistant.command_parser import CommandParser
from optimistic_voice.assistant.intent_classifier import IntentType
from optimistic_voice.audio.speech_recognition import SpeechRecognizer
from optimistic_voice.audio.voice_activity import VoiceActivityDetector
from optimistic_voice.config import OptimisticConfig


def test_natural_conversational_website_commands() -> None:
    parser = CommandParser()

    examples = [
        ("Open Google", "google"),
        ("Can you open Google for me?", "google"),
        ("Go to Google", "google"),
        ("Launch my browser and search Google", "google"),
        ("I want to check something on Google", "google"),
        ("Could you please open YouTube com", "youtube"),
        ("Take me to Wikipedia please", "wikipedia"),
        ("Let's check Gmail", "gmail"),
        ("I need to check Reddit", "reddit"),
    ]

    for utterance, expected_site in examples:
        cmd = parser.parse(utterance)
        assert cmd.intent == IntentType.OPEN_WEBSITE, f"Failed on {utterance!r}: got {cmd.intent}"
        assert cmd.params["name"] == expected_site
        assert cmd.confidence >= 0.85


def test_natural_application_commands() -> None:
    parser = CommandParser()

    examples = [
        ("open note pad", "notepad"),
        ("open note", "notepad"),
        ("open pad", "notepad"),
        ("open the note", "notepad"),
        ("open the pad", "notepad"),
        ("open not bad", "notepad"),
        ("launch pad", "notepad"),
        ("open notes", "notepad"),
        ("I want to take some notes in notepad", "notepad"),
        ("Can you open the calculator for me please", "calculator"),
        ("Launch calc", "calculator"),
        ("Bring up file explorer", "file_manager"),
        ("Can you open my files", "file_manager"),
        ("Open Chrome browser", "browser"),
    ]

    for utterance, expected_app in examples:
        cmd = parser.parse(utterance)
        assert cmd.intent == IntentType.OPEN_APPLICATION, f"Failed on {utterance!r}: got {cmd.intent}"
        assert cmd.params["app"] == expected_app
        assert cmd.confidence >= 0.85


def test_natural_youtube_search_and_playback() -> None:
    parser = CommandParser()

    cmd1 = parser.parse("Search YouTube for chill lofi music")
    assert cmd1.intent == IntentType.YOUTUBE_SEARCH
    assert "chill lofi music" in cmd1.params["query"]

    # Direct "search for <command>" matches YouTube search
    cmd_sf = parser.parse("search for interstellar soundtrack")
    assert cmd_sf.intent == IntentType.YOUTUBE_SEARCH
    assert cmd_sf.params["query"] == "interstellar soundtrack"

    # Typing into search bar
    cmd_type_bar = parser.parse("type in youtube search bar lofi beats")
    assert cmd_type_bar.intent == IntentType.YOUTUBE_SEARCH
    assert cmd_type_bar.params["query"] == "lofi beats"

    cmd_type_bar2 = parser.parse("type python crash course in youtube search bar")
    assert cmd_type_bar2.intent == IntentType.YOUTUBE_SEARCH
    assert cmd_type_bar2.params["query"] == "python crash course"

    # "open youtube and search for X"
    cmd_open_search = parser.parse("open youtube and search for coding tutorial")
    assert cmd_open_search.intent == IntentType.YOUTUBE_SEARCH
    assert cmd_open_search.params["query"] == "coding tutorial"

    cmd2 = parser.parse("Can you play some jazz on youtube for me")
    assert cmd2.intent == IntentType.YOUTUBE_SEARCH
    assert "jazz" in cmd2.params["query"]

    cmd3 = parser.parse("pause the video please")
    assert cmd3.intent == IntentType.YOUTUBE_PAUSE

    cmd4 = parser.parse("select video number 2")
    assert cmd4.intent == IntentType.YOUTUBE_SELECT_VIDEO
    assert cmd4.params["index"] == 2


def test_natural_desktop_and_typing_commands() -> None:
    parser = CommandParser()

    cmd1 = parser.parse("Please minimize this window")
    assert cmd1.intent == IntentType.DESKTOP_ACTION
    assert cmd1.params["action"] == "minimize_window"

    cmd2 = parser.parse("Close the current window for me")
    assert cmd2.intent == IntentType.DESKTOP_ACTION
    assert cmd2.params["action"] == "close_window"

    cmd3 = parser.parse("Can you type hello world for me")
    assert cmd3.intent == IntentType.TYPE_TEXT
    assert cmd3.params["text"] == "hello world"

    cmd_np1 = parser.parse("Type on notepad hello world")
    assert cmd_np1.intent == IntentType.TYPE_TEXT
    assert cmd_np1.params["text"] == "hello world"
    assert cmd_np1.params["target_app"] == "notepad"

    cmd_np2 = parser.parse("Open notepad and type this is my note")
    assert cmd_np2.intent == IntentType.TYPE_TEXT
    assert cmd_np2.params["text"] == "this is my note"
    assert cmd_np2.params["target_app"] == "notepad"

    cmd_np3 = parser.parse("OptiMistic type testing note on pad")
    assert cmd_np3.intent == IntentType.TYPE_TEXT
    assert cmd_np3.params["text"] == "testing note"
    assert cmd_np3.params["target_app"] == "notepad"


def test_vad_allows_natural_speech_pauses() -> None:
    # 1.2s silence timeout = 12 chunks of 100ms
    vad = VoiceActivityDetector(
        sample_rate=16000,
        energy_threshold=0.01,
        speech_pad_chunks=4,
        silence_chunks=10,
        chunk_samples=1600,
    )
    silence = np.zeros(1600, dtype=np.float32)
    tone = 0.2 * np.sin(np.linspace(0, 314, 1600, dtype=np.float32))

    # Pre-speech silence
    assert vad.process_chunk(silence) is None

    # First word ("Can you open...")
    for _ in range(3):
        assert vad.process_chunk(tone) is None
    assert vad.is_speaking is True

    # Natural brief pause between words (5 chunks = 500ms pause)
    # Must NOT cut off the utterance!
    for _ in range(5):
        assert vad.process_chunk(silence) is None
        assert vad.is_speaking is True, "VAD cut off speech prematurely during a natural 500ms pause!"

    # Second word ("...Google for me")
    for _ in range(3):
        assert vad.process_chunk(tone) is None

    # Final sentence-ending silence (10 chunks) -> triggers completion
    final_utt = None
    for _ in range(10):
        res = vad.process_chunk(silence)
        if res is not None:
            final_utt = res
            break

    assert final_utt is not None, "VAD failed to return utterance after final silence timeout"
    assert len(final_utt) > 1600 * 6


def test_streaming_speech_recognition_interface() -> None:
    cfg = OptimisticConfig()
    stt = SpeechRecognizer(engine="vosk", vosk_model_path=cfg.vosk_model_path)
    stt.start_stream()

    silence = np.zeros(1600, dtype=np.int16)
    partial = stt.feed_stream_chunk(silence)
    assert isinstance(partial, str)

    final = stt.finish_stream(silence)
    assert isinstance(final, str)


def test_optimistic_wake_prefix_and_filtering() -> None:
    from optimistic_voice.audio.wake_word import WakeWordDetector

    detector = WakeWordDetector(enabled=True)

    # 1. Sentences starting with "OptiMistic" and variants should trigger and extract clean commands
    valid_wake_commands = [
        ("OptiMistic open google", "open google"),
        ("OptiMistic, can you open notepad for me?", "can you open notepad for me?"),
        ("opti mystic search youtube for jazz", "search youtube for jazz"),
        ("opti mistic open calculator", "open calculator"),
        ("opt mystic close window", "close window"),
        ("optic mystic open calculator", "open calculator"),
        ("optemistic open google", "open google"),
        ("optimist open google", "open google"),
        ("optimists search for cats", "search for cats"),
        ("up to mystic open google", "open google"),
        ("up the mystic search for lofi", "search for lofi"),
        ("app to mystic open google", "open google"),
        ("optima stick open google", "open google"),
        ("optimus search for movie", "search for movie"),
        ("Hey OptiMistic minimize window", "minimize window"),
        ("OK OptiMistic close this window", "close this window"),
        ("hello optimistic what's on my screen", "what's on my screen"),
    ]

    for utterance, expected_clean in valid_wake_commands:
        should_run, cleaned, triggered = detector.process_transcript(utterance)
        assert should_run is True, f"Failed to trigger on valid prefix: {utterance}"
        assert triggered is True
        assert cleaned.strip(" ,.!?\"'").lower() == expected_clean.strip(" ,.!?\"'").lower()

    # 2. Sentences WITHOUT "OptiMistic" prefix must be IGNORED
    detector.sleep()  # ensure not awake
    ignored_utterances = [
        "open google",
        "can you open notepad for me?",
        "search youtube for chill music",
        "turn on the lights",
        "hello how are you",
        "what time is it",
    ]

    for utterance in ignored_utterances:
        should_run, cleaned, triggered = detector.process_transcript(utterance)
        assert should_run is False, f"Erroneously triggered on non-wake utterance: {utterance}"
        assert triggered is False

    # 3. Standalone "OptiMistic" should trigger wake session and prompt for command
    should_run, cleaned, triggered = detector.process_transcript("OptiMistic")
    assert should_run is False
    assert triggered is True
    assert detector.is_awake is True

    # Immediate follow-up within active window should now be accepted even without repeating prefix
    should_run, cleaned, triggered = detector.process_transcript("open google")
    assert should_run is True
    assert cleaned == "open google"
    assert detector.is_awake is False  # returns to sleep after command


def test_command_parser_handles_optimistic_prefix_directly() -> None:
    parser = CommandParser()

    cmd = parser.parse("OptiMistic open google")
    assert cmd.intent == IntentType.OPEN_WEBSITE
    assert cmd.params["name"] == "google"

    cmd2 = parser.parse("OptiMistic, can you open notepad for me?")
    assert cmd2.intent == IntentType.OPEN_APPLICATION
    assert cmd2.params["app"] == "notepad"

    cmd3 = parser.parse("opti mystic search youtube for lofi beats")
    assert cmd3.intent == IntentType.YOUTUBE_SEARCH
    assert "lofi beats" in cmd3.params["query"]

    cmd4 = parser.parse("up to mystic search for python")
    assert cmd4.intent == IntentType.YOUTUBE_SEARCH
    assert cmd4.params["query"] == "python"

    cmd5 = parser.parse("optima stick open google")
    assert cmd5.intent == IntentType.OPEN_WEBSITE
    assert cmd5.params["name"] == "google"

    cmd6 = parser.parse("optimist open notepad")
    assert cmd6.intent == IntentType.OPEN_APPLICATION
    assert cmd6.params["app"] == "notepad"


def test_notepad_false_positives_prevented() -> None:
    parser = CommandParser(enable_llm=True)

    # Conversational phrases containing "note", "pad", "editor", "not bad" must NEVER open Notepad
    false_positives = [
        "what is a note",
        "it is not bad at all",
        "who is the editor of this book",
        "where did I leave my mouse pad",
        "write a funny joke",
        "tell me a story about space",
    ]

    for phrase in false_positives:
        cmd = parser.parse(phrase)
        assert cmd.intent != IntentType.OPEN_APPLICATION, f"False positive triggered on: {phrase}"

    # Explicit requests to open Notepad MUST open Notepad
    valid_requests = [
        "Open Notepad",
        "Launch notepad",
        "OptiMistic, open notepad please",
        "start notepad app",
    ]

    for req in valid_requests:
        cmd = parser.parse(req)
        assert cmd.intent == IntentType.OPEN_APPLICATION
        assert cmd.params["app"] == "notepad"
