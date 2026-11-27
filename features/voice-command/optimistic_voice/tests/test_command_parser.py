"""Tests for natural language command parsing across all required categories."""

from __future__ import annotations

from optimistic_voice.assistant.command_parser import CommandParser
from optimistic_voice.assistant.intent_classifier import IntentType


def test_open_known_and_unknown_websites() -> None:
    parser = CommandParser()

    cmd = parser.parse("Open Google.")
    assert cmd.intent == IntentType.OPEN_WEBSITE
    assert cmd.params["url"] == "https://www.google.com"

    cmd_yt = parser.parse("Open YouTube.")
    assert cmd_yt.intent == IntentType.OPEN_WEBSITE
    assert cmd_yt.params["url"] == "https://www.youtube.com"

    cmd_gmail = parser.parse("Open Gmail")
    assert cmd_gmail.intent == IntentType.OPEN_WEBSITE
    assert cmd_gmail.params["url"] == "https://mail.google.com"

    cmd_unknown = parser.parse("Open someunknownwebsite website")
    assert cmd_unknown.intent == IntentType.OPEN_WEBSITE_UNKNOWN


def test_exact_text_typing_preserves_spelling() -> None:
    parser = CommandParser()

    cmd1 = parser.parse("Type YouTube.com.")
    assert cmd1.intent == IntentType.TYPE_TEXT
    assert cmd1.params["text"] == "YouTube.com"

    cmd2 = parser.parse("Type hello world.")
    assert cmd2.intent == IntentType.TYPE_TEXT
    assert cmd2.params["text"] == "hello world"

    # Intentional misspelling must NOT be corrected
    cmd3 = parser.parse("Type helllo wrld 123!")
    assert cmd3.intent == IntentType.TYPE_TEXT
    assert cmd3.params["text"] == "helllo wrld 123!"


def test_open_applications_and_terminal_confirmation() -> None:
    parser = CommandParser()

    notepad = parser.parse("Open Notepad.")
    assert notepad.intent == IntentType.OPEN_APPLICATION
    assert notepad.params["app"] == "notepad"
    assert notepad.requires_confirmation is False

    terminal = parser.parse("Open Terminal")
    assert terminal.intent == IntentType.OPEN_APPLICATION
    assert terminal.params["app"] == "terminal"
    assert terminal.requires_confirmation is True


def test_youtube_commands() -> None:
    parser = CommandParser()

    s = parser.parse("Search for artificial intelligence.")
    assert s.intent == IntentType.YOUTUBE_SEARCH
    assert s.params["query"] == "artificial intelligence"

    v1 = parser.parse("Select the first video.")
    assert v1.intent == IntentType.YOUTUBE_SELECT_VIDEO
    assert v1.params["index"] == 1

    v2 = parser.parse("Select the second video.")
    assert v2.intent == IntentType.YOUTUBE_SELECT_VIDEO
    assert v2.params["index"] == 2

    ambiguous = parser.parse("Select the video")
    assert ambiguous.intent == IntentType.AMBIGUOUS

    assert parser.parse("Play video").intent == IntentType.YOUTUBE_PLAY
    assert parser.parse("Pause video.").intent == IntentType.YOUTUBE_PAUSE
    assert parser.parse("Resume video").intent == IntentType.YOUTUBE_RESUME
    assert parser.parse("Go back").intent == IntentType.BROWSER_BACK
    assert parser.parse("Scroll down").params["direction"] == "down"
    assert parser.parse("Scroll up").params["direction"] == "up"


def test_desktop_and_sensitive_commands() -> None:
    parser = CommandParser()

    assert parser.parse("Close current window.").params["action"] == "close_window"
    assert parser.parse("Minimize window").params["action"] == "minimize_window"
    assert parser.parse("Maximize window").params["action"] == "maximize_window"
    assert parser.parse("Copy selected text").params["action"] == "copy"
    assert parser.parse("Paste copied text").params["action"] == "paste"
    assert parser.parse("Select all").params["action"] == "select_all"

    shutdown = parser.parse("Shut down the computer")
    assert shutdown.intent == IntentType.SENSITIVE_ACTION
    assert shutdown.requires_confirmation is True

