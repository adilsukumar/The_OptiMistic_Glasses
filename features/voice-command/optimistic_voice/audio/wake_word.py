"""Wake-word detection for 'Hey Optimistic' and configurable listening modes."""

from __future__ import annotations

import difflib
import re
import time

PREFIX_REGEX = re.compile(
    r"^(?:(?:hey|hi|hello|ok|okay|yo|dear)\s+)?"
    r"(?:optimi[sz]t(?:ic(?:s|al)?)?|opti[- ']?m[yi]stic(?:s)?|opt[- ']?m[yi]stic(?:s)?|"
    r"optic[- ']?m[yi]stic(?:s)?|opty?mistic|optemistic|optima[- ]?stick|"
    r"up[- ]?(?:to|the)?[- ]?m[yi]stic|app[- ]?(?:to|the)?[- ]?m[yi]stic|"
    r"apt[- ]?(?:to|the)?[- ]?m[yi]stic|off[- ]?(?:the)?[- ]?m[yi]stic|"
    r"auto[- ]?m[yi]stic|atomistic|optimus)\b[,:\s]*",
    re.IGNORECASE,
)


class WakeWordDetector:
    """Detects configurable wake prefix 'OptiMistic' and manages active listening windows."""

    def __init__(
        self,
        enabled: bool = True,
        wake_phrases: tuple[str, ...] = ("optimistic", "hey optimistic", "ok optimistic", "hi optimistic", "hello optimistic"),
        session_timeout_s: float = 12.0,
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

    @staticmethod
    def _match_fuzzy_prefix(raw: str) -> tuple[bool, str]:
        """Fuzzy match on leading 1-3 words sounding like 'OptiMistic'."""
        cleaned = raw.strip()
        if not cleaned:
            return False, ""
        lead_m = re.match(r"^(?:hey|hi|hello|ok|okay|yo|dear)\s+", cleaned, re.IGNORECASE)
        sub = cleaned[lead_m.end():].strip() if lead_m else cleaned
        words = sub.split()
        targets = ("optimistic", "optimist", "optimystic", "optimistics")
        for n in (1, 2, 3):
            if len(words) >= n:
                candidate = "".join(w.lower().strip(" ,.!?\"'") for w in words[:n])
                for target in targets:
                    if difflib.SequenceMatcher(None, target, candidate).ratio() >= 0.72:
                        remainder = " ".join(words[n:]).strip(" ,.!?\"'")
                        return True, remainder
        return False, ""

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

        # 1. Regex check for all known phonetic pronunciations of "OptiMistic"
        match = PREFIX_REGEX.match(raw)
        if match:
            remainder = raw[match.end() :].strip(" ,.!?\"'")
            if remainder:
                self.sleep()
                return True, remainder, True
            self.keep_awake()
            return False, "", True

        # 2. Secondary fuzzy similarity check on the leading 1-3 words
        fuzzy_matched, remainder = self._match_fuzzy_prefix(raw)
        if fuzzy_matched:
            if remainder:
                self.sleep()
                return True, remainder, True
            self.keep_awake()
            return False, "", True

        # Also support any custom wake phrases if specified and anchored to start
        lowered = raw.lower()
        for phrase in self.wake_phrases:
            pattern = rf"^(?:hey\s+|ok\s+|hi\s+)?{re.escape(phrase)}\b[,:\s]*"
            m = re.match(pattern, lowered)
            if m:
                remainder = raw[m.end() :].strip(" ,.!?\"'")
                self.keep_awake()
                if remainder:
                    return True, remainder, True
                return False, "", True

        # 3. If already awake from a standalone "OptiMistic" wake call, accept the immediate follow-up
        if self.is_awake:
            self.sleep()
            return True, raw, False

        return False, "", False
