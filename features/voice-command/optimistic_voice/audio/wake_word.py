"""Wake-word detection for 'Hey Optimistic' and configurable listening modes."""

from __future__ import annotations

import re
import time


class WakeWordDetector:
    """Detects configurable wake phrases and manages active listening windows."""

    def __init__(
        self,
        enabled: bool = False,
        wake_phrases: tuple[str, ...] = ("hey optimistic", "ok optimistic", "okay optimistic", "optimistic"),
        session_timeout_s: float = 15.0,
    ) -> None:
        self.enabled = enabled
        self.wake_phrases = tuple(p.lower().strip() for p in wake_phrases)
        self.session_timeout_s = session_timeout_s
        self._awake_until: float = 0.0

    @property
    def is_awake(self) -> bool:
        if not self.enabled:
            return True
        return time.monotonic() < self._awake_until

    def keep_awake(self) -> None:
        self._awake_until = time.monotonic() + self.session_timeout_s

    def sleep(self) -> None:
        self._awake_until = 0.0

    def process_transcript(self, text: str) -> tuple[bool, str, bool]:
        """Evaluate transcript against wake-word mode.

        Returns:
            (should_process_command, cleaned_command_text, wake_triggered_now)
        """
        raw = text.strip()
        if not raw:
            return False, "", False

        if not self.enabled:
            return True, raw, False

        lowered = raw.lower()
        for phrase in self.wake_phrases:
            pattern = rf"\b{re.escape(phrase)}\b[,:\s]*"
            match = re.search(pattern, lowered)
            if match:
                remainder = (raw[: match.start()] + raw[match.end() :]).strip(" ,.!?")
                self.keep_awake()
                if remainder:
                    return True, remainder, True
                return False, "", True

        if self.is_awake:
            self.keep_awake()
            return True, raw, False

        return False, "", False
