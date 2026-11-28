from __future__ import annotations

import logging
import subprocess
import threading
from collections import deque

logger = logging.getLogger(__name__)


def format_transcript(phrases):
    return " ".join(phrase if phrase.endswith((".", "!", "?")) else phrase + "." for phrase in phrases)


class WindowsSpeaker:
    """Play confirmed English phrases in order without blocking camera inference."""

    def __init__(self):
        self._pending = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def speak(self, text):
        if text:
            with self._condition:
                if not self._closing:
                    self._pending.append(text)
                    self._condition.notify()

    def clear(self):
        with self._condition:
            self._pending.clear()

    def close(self):
        with self._condition:
            self._closing = True
            self._pending.clear()
            self._condition.notify()

    def _run(self):
        voice = None
        com = None
        english = None
        try:
            try:
                import pythoncom
                import win32com.client

                pythoncom.CoInitialize()
                com = pythoncom
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                english = voice.GetVoices("Language=409")
                if english.Count:
                    voice.Voice = english.Item(0)
            except Exception:
                logger.debug("SAPI unavailable; falling back to System.Speech", exc_info=True)

            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closing or self._pending)
                    if self._closing:
                        return
                    text = self._pending[0] if len(self._pending) == 1 else format_transcript(self._pending)
                    self._pending.clear()
                try:
                    if voice is not None:
                        voice.Speak(text, 0)
                        continue
                except Exception:
                    voice = None
                try:
                    subprocess.run(
                        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                         "[Console]::InputEncoding = [Text.Encoding]::UTF8; "
                         "Add-Type -AssemblyName System.Speech; "
                         "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                         "$voice.Speak([Console]::In.ReadToEnd())"],
                        input=text.encode("utf-8"), capture_output=True, check=True, timeout=60,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except (OSError, subprocess.SubprocessError):
                    logger.exception("Could not play translated speech")
        finally:
            voice = None
            english = None
            if com is not None:
                com.CoUninitialize()
