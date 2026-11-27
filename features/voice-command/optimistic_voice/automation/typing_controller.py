"""Exact text typing controller without silent spell-checking or alteration."""

from __future__ import annotations

import platform
import time

from optimistic_voice.automation import ExecutionResult


class TypingController:
    """Types exact user-requested text into Notepad or currently focused input field."""

    def __init__(self, interval_s: float = 0.015) -> None:
        self.interval_s = interval_s

    def ensure_notepad_ready(self) -> None:
        """Ensure Notepad is open and focused for typing."""
        if platform.system() == "Windows":
            try:
                import ctypes

                user32 = ctypes.windll.user32
                found_hwnd = [0]
                wnd_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

                def _enum_proc(hwnd: int, extra: int) -> bool:
                    if user32.IsWindowVisible(hwnd):
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buff = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buff, length + 1)
                            if "notepad" in buff.value.lower():
                                found_hwnd[0] = hwnd
                                return False
                    return True

                user32.EnumWindows(wnd_proc(_enum_proc), 0)
                if found_hwnd[0]:
                    user32.ShowWindow(found_hwnd[0], 9)  # SW_RESTORE
                    user32.SetForegroundWindow(found_hwnd[0])
                    time.sleep(0.15)
                    return
            except Exception:
                pass

            try:
                import subprocess

                subprocess.Popen(["notepad.exe"])
                time.sleep(0.4)
            except Exception:
                pass
        elif platform.system() == "Linux":
            try:
                import shutil
                import subprocess

                for ed in ("gedit", "mousepad", "leafpad", "xed"):
                    if shutil.which(ed):
                        subprocess.Popen([ed])
                        time.sleep(0.35)
                        break
            except Exception:
                pass

    def type_text(self, text: str, target_app: str = "notepad") -> ExecutionResult:
        if text == "":
            return ExecutionResult(success=False, message="No text provided to type.")
        try:
            if target_app == "notepad":
                self.ensure_notepad_ready()

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
