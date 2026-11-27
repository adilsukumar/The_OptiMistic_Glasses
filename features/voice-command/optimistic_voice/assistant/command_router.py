"""Routes parsed commands to local automation or the Windows companion client."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand
from optimistic_voice.assistant.llm_client import LLMClient
from optimistic_voice.automation import ExecutionResult
from optimistic_voice.automation.application_launcher import ApplicationLauncher
from optimistic_voice.automation.browser_control import BrowserController
from optimistic_voice.automation.desktop_control import DesktopController
from optimistic_voice.automation.typing_controller import TypingController
from optimistic_voice.automation.youtube_control import YouTubeController
from optimistic_voice.config import OptimisticConfig


class CommandRouter:
    """Executes structured commands locally or via an authenticated remote client."""

    def __init__(
        self,
        config: OptimisticConfig,
        browser: BrowserController | None = None,
        youtube: YouTubeController | None = None,
        desktop: DesktopController | None = None,
        launcher: ApplicationLauncher | None = None,
        typing: TypingController | None = None,
        llm: LLMClient | None = None,
        remote_client: Any = None,
    ) -> None:
        self.config = config
        self.browser = browser or BrowserController(
            headless=config.browser_headless,
            channel=config.browser_channel,
        )
        self.youtube = youtube or YouTubeController(self.browser)
        self.desktop = desktop or DesktopController()
        self.launcher = launcher or ApplicationLauncher()
        self.typing = typing or TypingController(interval_s=config.typing_interval_s)
        self.llm = llm or LLMClient(config)
        self.remote_client = remote_client

    def execute(self, cmd: ParsedCommand) -> ExecutionResult:
        if self.config.execution_mode == "remote_windows" and self.remote_client is not None:
            return self.remote_client.send_command(cmd)
        return self.execute_local(cmd)

    def execute_local(self, cmd: ParsedCommand) -> ExecutionResult:
        intent = cmd.intent
        p = cmd.params

        if intent == IntentType.OPEN_WEBSITE:
            return self.browser.open_url(p["url"])

        if intent == IntentType.LLM_QUERY:
            query = p.get("query", cmd.raw_text)
            return self.llm.ask(query)

        if intent == IntentType.TYPE_TEXT:
            return self.typing.type_text(p.get("text", ""), target_app=p.get("target_app", "notepad"))

        if intent == IntentType.OPEN_APPLICATION:
            return self.launcher.launch(p.get("app", ""))

        if intent == IntentType.YOUTUBE_SEARCH:
            return self.youtube.search(p.get("query", ""))

        if intent == IntentType.YOUTUBE_SELECT_VIDEO:
            return self.youtube.select_video(int(p.get("index", 1)))

        if intent == IntentType.YOUTUBE_PLAY:
            return self.youtube.play()

        if intent == IntentType.YOUTUBE_PAUSE:
            return self.youtube.pause()

        if intent == IntentType.YOUTUBE_RESUME:
            return self.youtube.resume()

        if intent == IntentType.YOUTUBE_NEXT_PAGE:
            return self.youtube.next_page()

        if intent == IntentType.YOUTUBE_BACK_TO_RESULTS:
            return self.youtube.back_to_results()

        if intent == IntentType.BROWSER_BACK:
            return self.browser.go_back()

        if intent == IntentType.SCROLL:
            direction = p.get("direction", "down")
            # Scroll inside browser if it's active, otherwise desktop scroll
            if self.browser.get_current_url():
                return self.browser.scroll(direction)
            return self.desktop.scroll(direction)

        if intent == IntentType.VOLUME:
            return self.desktop.adjust_volume(p.get("action", "up"))

        if intent == IntentType.DESKTOP_ACTION:
            action = p.get("action", "")
            if action == "close_window":
                closed_any = False
                # 1. Close Playwright browser page if open
                try:
                    if self.browser is not None and getattr(self.browser, "has_open_page", lambda: False)():
                        self.browser.close_current_page()
                        closed_any = True
                except Exception:
                    pass

                # 2. Close any YouTube window open in external browser (Chrome/Edge/Brave)
                if self._close_youtube_os_window():
                    closed_any = True

                if closed_any:
                    return ExecutionResult(success=True, message="Closed the YouTube window.")

                # 3. Standard fallback: close current active window
                return self.desktop.execute_action("close_window")

            return self.desktop.execute_action(action)

        if intent == IntentType.STOP_ASSISTANT:
            return ExecutionResult(success=True, message="Stopped.")

        if intent == IntentType.SENSITIVE_ACTION:
            return self._execute_sensitive_action(p.get("action", ""))

        return ExecutionResult(success=False, message=f"Unsupported intent: {intent.value}")

    @staticmethod
    def _close_youtube_os_window() -> bool:
        """Find and close any active Windows OS window playing YouTube."""
        import platform
        if platform.system() != "Windows":
            return False
        try:
            import ctypes
            import time
            import pyautogui

            user32 = ctypes.windll.user32
            found: list[tuple[int, str]] = []

            def enum_cb(hwnd: int, lparam: int) -> bool:
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        title = buff.value
                        if "youtube" in title.lower():
                            found.append((hwnd, title))
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
            if found:
                hwnd, _ = found[0]
                user32.ShowWindow(hwnd, 9)
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.10)
                # In browser, Ctrl+W closes the YouTube tab cleanly
                pyautogui.hotkey("ctrl", "w")
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _execute_sensitive_action(action: str) -> ExecutionResult:
        os_name = platform.system()
        if action == "shutdown":
            cmd = ["shutdown", "/s", "/t", "5"] if os_name == "Windows" else ["sudo", "shutdown", "-h", "now"]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ExecutionResult(success=True, message="Shutting down the computer.")
        if action == "restart":
            cmd = ["shutdown", "/r", "/t", "5"] if os_name == "Windows" else ["sudo", "reboot"]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ExecutionResult(success=True, message="Restarting the computer.")
        return ExecutionResult(
            success=False,
            message="For safety, file deletion via voice requires selecting the file manually.",
        )

    def close(self) -> None:
        self.browser.close()
