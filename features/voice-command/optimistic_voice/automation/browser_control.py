"""Thread-safe Playwright browser automation controller."""

from __future__ import annotations

import concurrent.futures
import webbrowser
from typing import Any, Callable, TypeVar

from optimistic_voice.automation import ExecutionResult

T = TypeVar("T")


class BrowserController:
    """Manages a persistent Playwright browser session on a dedicated worker thread."""

    def __init__(self, headless: bool = False, channel: str | None = None) -> None:
        self.headless = headless
        self.channel = channel
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="OptimisticBrowser")
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    def _run(self, fn: Callable[[], T]) -> T:
        return self._executor.submit(fn).result()

    def _ensure_page_sync(self) -> Any:
        if self._page is not None and not self._page.is_closed():
            return self._page

        from playwright.sync_api import sync_playwright

        if self._playwright is None:
            self._playwright = sync_playwright().start()

        if self._browser is None:
            launch_kwargs: dict[str, Any] = {
                "headless": self.headless,
                "args": ["--disable-blink-features=AutomationControlled"],
            }
            if self.channel:
                launch_kwargs["channel"] = self.channel
            try:
                self._browser = self._playwright.chromium.launch(**launch_kwargs)
            except Exception:
                # Retry without channel if specified channel isn't installed
                launch_kwargs.pop("channel", None)
                self._browser = self._playwright.chromium.launch(**launch_kwargs)

        if self._context is None:
            self._context = self._browser.new_context(viewport={"width": 1280, "height": 800})

        self._page = self._context.new_page()
        return self._page

    def get_page(self) -> Any:
        return self._run(self._ensure_page_sync)

    def run_on_page(self, fn: Callable[[Any], T]) -> T:
        """Execute a callable receiving the active Playwright Page on the browser thread."""
        return self._run(lambda: fn(self._ensure_page_sync()))

    def open_url(self, url: str) -> ExecutionResult:
        try:
            def _open(page: Any) -> dict[str, str]:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                try:
                    page.bring_to_front()
                except Exception:
                    pass
                try:
                    title = page.title()
                except Exception:
                    title = ""
                return {"url": page.url, "title": title}

            info = self.run_on_page(_open)
            return ExecutionResult(success=True, message=f"Opened {info['url']}", data=info)
        except Exception as exc:
            # Fallback to OS default browser if Playwright cannot launch
            try:
                if webbrowser.open(url):
                    return ExecutionResult(success=True, message=f"Opened {url} in default browser")
            except Exception:
                pass
            return ExecutionResult(success=False, message=f"Failed to open {url}: {exc}")

    def go_back(self) -> ExecutionResult:
        try:
            def _back(page: Any) -> str:
                page.go_back(wait_until="domcontentloaded", timeout=10000)
                return page.url

            url = self.run_on_page(_back)
            return ExecutionResult(success=True, message=f"Navigated back to {url}")
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Could not go back: {exc}")

    def scroll(self, direction: str = "down", pixels: int = 600) -> ExecutionResult:
        try:
            delta = abs(pixels) if direction.lower() == "down" else -abs(pixels)
            self.run_on_page(lambda page: page.evaluate(f"window.scrollBy({{top: {delta}, behavior: 'smooth'}});"))
            return ExecutionResult(success=True, message=f"Scrolled {direction}")
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Scroll failed: {exc}")

    def get_current_url(self) -> str:
        try:
            return self.run_on_page(lambda page: str(page.url))
        except Exception:
            return ""

    def has_open_page(self) -> bool:
        return self._page is not None and not self._page.is_closed()

    def close_current_page(self) -> ExecutionResult:
        """Close the currently active browser page/window."""
        try:
            def _close_page() -> str:
                url = ""
                if self._page is not None and not self._page.is_closed():
                    url = str(self._page.url)
                    self._page.close()
                    self._page = None
                return url

            url = self._run(_close_page)
            return ExecutionResult(success=True, message=f"Closed browser window: {url}")
        except Exception as exc:
            return ExecutionResult(success=False, message=f"Could not close browser window: {exc}")

    def close(self) -> None:
        def _shutdown() -> None:
            for obj in (self._context, self._browser, self._playwright):
                if obj is not None:
                    try:
                        obj.close() if hasattr(obj, "close") else obj.stop()
                    except Exception:
                        pass
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None

        try:
            self._run(_shutdown)
        except Exception:
            pass
        self._executor.shutdown(wait=False)
