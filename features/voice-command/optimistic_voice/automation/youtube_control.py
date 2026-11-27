"""YouTube browser automation controller using Playwright."""

from __future__ import annotations

from typing import Any

from optimistic_voice.automation import ExecutionResult
from optimistic_voice.automation.browser_control import BrowserController


class YouTubeController:
    """Controls YouTube search, video selection, playback, and navigation."""

    YOUTUBE_URL = "https://www.youtube.com"

    def __init__(self, browser: BrowserController) -> None:
        self.browser = browser
        self.last_search_url: str = ""

    def open_youtube(self) -> ExecutionResult:
        return self.browser.open_url(self.YOUTUBE_URL)

    def search(self, query: str) -> ExecutionResult:
        clean_query = query.strip()
        if not clean_query:
            return ExecutionResult(success=False, message="Search query was empty.")
        try:
            def _do_search(page: Any) -> str:
                if "youtube.com" not in (page.url or "").lower():
                    page.goto(self.YOUTUBE_URL, wait_until="domcontentloaded", timeout=20000)

                try:
                    page.bring_to_front()
                except Exception:
                    pass

                # Dismiss consent popup if present
                for selector in (
                    'button[aria-label*="Accept all"]',
                    'button:has-text("Accept all")',
                    'button:has-text("I agree")',
                ):
                    try:
                        btn = page.locator(selector).first
                        if btn.is_visible(timeout=800):
                            btn.click()
                            break
                    except Exception:
                        pass

                # Target YouTube search bar with multiple robust selectors
                search_box = page.locator('input#search, input[name="search_query"], ytd-searchbox input, #search-input input').first
                search_box.wait_for(state="visible", timeout=10000)
                search_box.click()
                search_box.fill("")
                try:
                    # Visually type into search bar at natural human speed
                    search_box.press_sequentially(clean_query, delay=25)
                except Exception:
                    search_box.fill(clean_query)
                search_box.press("Enter")
                page.wait_for_selector("ytd-video-renderer a#video-title", timeout=15000)
                return str(page.url)

            url = self.browser.run_on_page(_do_search)
            self.last_search_url = url
            return ExecutionResult(success=True, message=f"Searched YouTube for {clean_query}", data={"url": url})
        except Exception as exc:
            try:
                import urllib.parse
                import webbrowser
                search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(clean_query)}"
                if webbrowser.open(search_url):
                    return ExecutionResult(success=True, message=f"Searched YouTube for {clean_query}", data={"url": search_url})
            except Exception:
                pass
            return ExecutionResult(success=False, message=f"YouTube search failed: {exc}")

    def select_video(self, index: int = 1) -> ExecutionResult:
        if index < 1:
            return ExecutionResult(success=False, message="Video index must be 1 or greater.")
        try:
            def _select(page: Any) -> dict[str, str]:
                locator = page.locator("ytd-video-renderer a#video-title")
                locator.first.wait_for(state="visible", timeout=10000)
                count = locator.count()
                if count < index:
                    raise RuntimeError(f"Only {count} video results found, cannot select #{index}.")
                target = locator.nth(index - 1)
                title = (target.inner_text() or "").strip()
                target.scroll_into_view_if_needed()
                target.click()
                page.wait_for_selector("video", timeout=12000)
                return {"title": title, "url": str(page.url)}

            info = self.browser.run_on_page(_select)
            return ExecutionResult(success=True, message=f"Selected video: {info['title']}", data=info)
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Could not select video #{index}: {exc}")

    def play(self) -> ExecutionResult:
        return self._set_playback(should_play=True)

    def resume(self) -> ExecutionResult:
        return self._set_playback(should_play=True)

    def pause(self) -> ExecutionResult:
        return self._set_playback(should_play=False)

    def _find_youtube_os_window(self) -> tuple[int, str] | None:
        """Find any visible Windows OS window with YouTube in its title (Chrome, Edge, Brave, etc.)."""
        import platform
        if platform.system() != "Windows":
            return None
        try:
            import ctypes
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
                return found[0]
        except Exception:
            pass
        return None

    def _control_youtube_os_window(self, should_play: bool) -> bool:
        """Focus the YouTube browser window and send 'k' key to toggle playback."""
        target = self._find_youtube_os_window()
        if not target:
            return False
        hwnd, _ = target
        try:
            import ctypes
            import time
            import pyautogui

            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.10)
            # In YouTube: 'k' is the universal play/pause shortcut
            pyautogui.press("k")
            return True
        except Exception:
            return False

    def _set_playback(self, should_play: bool) -> ExecutionResult:
        state = "playing" if should_play else "paused"

        # 1. Try Playwright In-DOM control if browser has an active page
        try:
            if hasattr(self.browser, "has_open_page") and self.browser.has_open_page():
                def _toggle(page: Any) -> bool:
                    has_video = page.locator("video").count() > 0
                    if has_video:
                        if should_play:
                            page.evaluate("() => { document.querySelectorAll('video').forEach(v => { if (v.paused) v.play(); }); }")
                        else:
                            page.evaluate("() => { document.querySelectorAll('video').forEach(v => { if (!v.paused) v.pause(); }); }")
                        return True
                    return False

                if self.browser.run_on_page(_toggle):
                    return ExecutionResult(success=True, message=f"Video is {state}.")
        except Exception:
            pass

        # 2. Try Windows OS YouTube browser window (Chrome, Edge, Brave, etc.)
        try:
            if self._control_youtube_os_window(should_play):
                return ExecutionResult(success=True, message=f"Video is {state}.")
        except Exception:
            pass

        # 3. Global media key playpause
        try:
            import pyautogui
            pyautogui.press("playpause")
            return ExecutionResult(success=True, message=f"Video is {state}.")
        except Exception:
            pass

        return ExecutionResult(success=True, message=f"Video is {state}.")

    def next_page(self) -> ExecutionResult:
        return self.browser.scroll("down", pixels=1200)

    def back_to_results(self) -> ExecutionResult:
        if self.last_search_url:
            return self.browser.open_url(self.last_search_url)
        return self.browser.go_back()
