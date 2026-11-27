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

                search_box = page.locator('input[name="search_query"], #search-input input').first
                search_box.wait_for(state="visible", timeout=10000)
                search_box.click()
                search_box.fill(clean_query)
                search_box.press("Enter")
                page.wait_for_selector("ytd-video-renderer a#video-title", timeout=15000)
                return str(page.url)

            url = self.browser.run_on_page(_do_search)
            self.last_search_url = url
            return ExecutionResult(success=True, message=f"Searched YouTube for {clean_query}", data={"url": url})
        except Exception as exc:
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

    def _set_playback(self, should_play: bool) -> ExecutionResult:
        try:
            def _toggle(page: Any) -> bool:
                page.wait_for_selector("video", timeout=6000)
                if should_play:
                    page.evaluate("() => { const v = document.querySelector('video'); if (v && v.paused) v.play(); }")
                else:
                    page.evaluate("() => { const v = document.querySelector('video'); if (v && !v.paused) v.pause(); }")
                paused = bool(page.evaluate("() => { const v = document.querySelector('video'); return v ? v.paused : true; }"))
                return (not paused) if should_play else paused

            ok = self.browser.run_on_page(_toggle)
            state = "playing" if should_play else "paused"
            if ok:
                return ExecutionResult(success=True, message=f"Video is {state}.")
            return ExecutionResult(success=False, message=f"Could not verify video is {state}.")
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Video playback control failed: {exc}")

    def next_page(self) -> ExecutionResult:
        return self.browser.scroll("down", pixels=1200)

    def back_to_results(self) -> ExecutionResult:
        if self.last_search_url:
            return self.browser.open_url(self.last_search_url)
        return self.browser.go_back()
