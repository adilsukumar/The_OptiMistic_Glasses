"""Integration and acceptance tests for OptimisticAssistant and Pi-Windows auth."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optimistic_voice.assistant.command_router import CommandRouter
from optimistic_voice.assistant.intent_classifier import IntentType
from optimistic_voice.audio.text_to_speech import TextToSpeechEngine
from optimistic_voice.automation import ExecutionResult
from optimistic_voice.config import OptimisticConfig, READY_PROMPT, STARTUP_GREETING
from optimistic_voice.display.display_manager import DisplayManager
from optimistic_voice.integration.communication import sign_message, verify_message
from optimistic_voice.main import OptimisticAssistant


def _make_mock_assistant() -> tuple[OptimisticAssistant, MagicMock]:
    cfg = OptimisticConfig(tts_engine="mock", display_mode="headless", wake_word_enabled=False)
    tts = TextToSpeechEngine(engine="mock")
    display = DisplayManager(mode="headless")
    router = MagicMock(spec=CommandRouter)
    router.execute.return_value = ExecutionResult(success=True, message="ok")
    assistant = OptimisticAssistant(config=cfg, router=router, tts=tts, display=display)
    return assistant, router


def test_mandatory_startup_greeting_order() -> None:
    assistant, _ = _make_mock_assistant()
    assistant.startup()
    assert assistant.tts.spoken_history[:2] == [STARTUP_GREETING, READY_PROMPT]
    assert assistant.display.history[0][0] == STARTUP_GREETING
    assistant.shutdown()


def test_acceptance_sequence_end_to_end() -> None:
    assistant, router = _make_mock_assistant()
    assistant.startup()

    sequence = [
        ("Open Google.", IntentType.OPEN_WEBSITE, "Opening Google."),
        ("Open YouTube.", IntentType.OPEN_WEBSITE, "Opening YouTube."),
        ("Search for artificial intelligence.", IntentType.YOUTUBE_SEARCH, "Searching for artificial intelligence."),
        ("Select the first video.", IntentType.YOUTUBE_SELECT_VIDEO, "Selecting the first video."),
        ("Pause video.", IntentType.YOUTUBE_PAUSE, "Pausing the video."),
        ("Open Notepad.", IntentType.OPEN_APPLICATION, "Opening Notepad."),
        ("Type hello world.", IntentType.TYPE_TEXT, "Typing hello world."),
        ("Close current window.", IntentType.DESKTOP_ACTION, "Closing the current window."),
    ]

    for idx, (utterance, expected_intent, expected_speech) in enumerate(sequence):
        res = assistant.process_utterance(utterance)
        assert res is not None and res.success is True
        called_cmd = router.execute.call_args_list[idx].args[0]
        assert called_cmd.intent == expected_intent
        assert assistant.tts.spoken_history[-1] == expected_speech

    assistant.shutdown()


def test_confirmation_flow_for_sensitive_action() -> None:
    assistant, router = _make_mock_assistant()
    res1 = assistant.process_utterance("Shut down computer")
    assert res1 is not None and res1.message == "Awaiting confirmation"
    assert router.execute.call_count == 0

    res2 = assistant.process_utterance("No")
    assert res2 is not None and res2.message == "Action cancelled."
    assert router.execute.call_count == 0
    assistant.shutdown()


def test_hmac_authentication_roundtrip() -> None:
    secret = "super-secret-optimistic-token"
    msg = sign_message(secret, {"intent": "open_website", "params": {"url": "https://www.google.com"}})
    decoded = verify_message(secret, msg)
    assert decoded["intent"] == "open_website"

    with pytest.raises(PermissionError):
        verify_message("wrong-secret", msg)

