"""Desktop window, keyboard shortcut, scroll, and volume automation."""

from __future__ import annotations

import platform

from optimistic_voice.automation import ExecutionResult


class DesktopController:
    """Controls desktop windows, clipboard hotkeys, scrolling, and system volume."""

    def __init__(self) -> None:
        self.os_name = platform.system()

    def execute_action(self, action: str) -> ExecutionResult:
        try:
            import pyautogui

            pyautogui.FAILSAFE = False
            mod = "command" if self.os_name == "Darwin" else "ctrl"

            if action == "minimize_window":
                if self.os_name == "Windows":
                    pyautogui.hotkey("win", "down")
                else:
                    pyautogui.hotkey("alt", "f9")
            elif action == "maximize_window":
                if self.os_name == "Windows":
                    pyautogui.hotkey("win", "up")
                else:
                    pyautogui.hotkey("alt", "f10")
            elif action == "close_window":
                if self.os_name == "Darwin":
                    pyautogui.hotkey("command", "w")
                else:
                    pyautogui.hotkey("alt", "f4")
            elif action == "switch_application":
                pyautogui.hotkey("alt", "tab")
            elif action == "show_desktop":
                if self.os_name == "Windows":
                    pyautogui.hotkey("win", "d")
                else:
                    pyautogui.hotkey("ctrl", "alt", "d")
            elif action == "press_enter":
                pyautogui.press("enter")
            elif action == "press_escape":
                pyautogui.press("escape")
            elif action == "press_backspace":
                pyautogui.press("backspace")
            elif action == "copy":
                pyautogui.hotkey(mod, "c")
            elif action == "paste":
                pyautogui.hotkey(mod, "v")
            elif action == "select_all":
                pyautogui.hotkey(mod, "a")
            else:
                return ExecutionResult(success=False, message=f"Unsupported desktop action: {action}")

            return ExecutionResult(success=True, message=f"Executed {action}")
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Desktop action '{action}' failed: {exc}")

    def scroll(self, direction: str = "down", clicks: int = 500) -> ExecutionResult:
        try:
            import pyautogui

            amount = -abs(clicks) if direction.lower() == "down" else abs(clicks)
            pyautogui.scroll(amount)
            return ExecutionResult(success=True, message=f"Scrolled {direction}")
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Scroll failed: {exc}")

    def adjust_volume(self, action: str) -> ExecutionResult:
        try:
            import pyautogui

            key_map = {
                "up": "volumeup",
                "down": "volumedown",
                "mute": "volumemute",
                "unmute": "volumemute",
            }
            key = key_map.get(action)
            if not key:
                return ExecutionResult(success=False, message=f"Unknown volume action: {action}")
            repeats = 3 if action in ("up", "down") else 1
            for _ in range(repeats):
                pyautogui.press(key)
            return ExecutionResult(success=True, message=f"Volume {action} executed")
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Volume control failed: {exc}")
