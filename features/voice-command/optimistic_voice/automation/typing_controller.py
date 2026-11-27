"""Exact text typing controller without silent spell-checking or alteration."""

from __future__ import annotations

import platform
import time

from optimistic_voice.automation import ExecutionResult


class TypingController:
    """Types exact user-requested text into the currently focused input field."""

    def __init__(self, interval_s: float = 0.015) -> None:
        self.interval_s = interval_s

    def type_text(self, text: str) -> ExecutionResult:
        if text == "":
            return ExecutionResult(success=False, message="No text provided to type.")
        try:
            import pyautogui

            # If text is pure printable ASCII, type directly to avoid touching clipboard;
            # otherwise use clipboard paste so unicode characters are preserved intact.
            if all(32 <= ord(ch) <= 126 for ch in text):
                pyautogui.write(text, interval=self.interval_s)
            else:
                import pyperclip

                pyperclip.copy(text)
                time.sleep(0.05)
                mod = "command" if platform.system() == "Darwin" else "ctrl"
                pyautogui.hotkey(mod, "v")
            return ExecutionResult(success=True, message=f"Typed: {text}", data={"text": text})
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Typing failed: {exc}")
