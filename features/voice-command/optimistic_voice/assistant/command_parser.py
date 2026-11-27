"""Natural language command parser for Optimistic voice commands."""

from __future__ import annotations

import re
from typing import Mapping

from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand
from optimistic_voice.config import OptimisticConfig

ORDINAL_MAP: dict[str, int] = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "1": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "2": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "3": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "4": 4,
    "fifth": 5,
    "5th": 5,
    "five": 5,
    "5": 5,
}

APP_ALIASES: dict[str, str] = {
    "notepad": "notepad",
    "text editor": "notepad",
    "editor": "notepad",
    "calculator": "calculator",
    "calc": "calculator",
    "file manager": "file_manager",
    "files": "file_manager",
    "file explorer": "file_manager",
    "explorer": "file_manager",
    "terminal": "terminal",
    "command prompt": "terminal",
    "cmd": "terminal",
    "powershell": "terminal",
    "browser": "browser",
    "web browser": "browser",
    "chrome": "browser",
    "google chrome": "browser",
    "edge": "browser",
    "firefox": "browser",
}


class CommandParser:
    """Parses natural spoken English into structured ParsedCommand objects."""

    def __init__(self, known_websites: Mapping[str, str] | None = None) -> None:
        self.known_websites = {
            k.lower(): v for k, v in (known_websites or OptimisticConfig().known_websites).items()
        }

    @staticmethod
    def _clean(text: str) -> str:
        cleaned = " ".join(text.strip().split())
        return cleaned.rstrip("?!.") if cleaned.endswith(("?", "!")) else cleaned

    def parse(self, text: str) -> ParsedCommand:
        raw = text.strip()
        if not raw:
            return ParsedCommand(intent=IntentType.UNKNOWN, raw_text=raw, confidence=0.0)

        lowered = " ".join(raw.lower().split()).rstrip(".,!?")

        # 1. Confirmation responses
        if lowered in {"yes", "yeah", "yep", "confirm", "do it", "proceed", "sure", "ok", "okay"}:
            return ParsedCommand(intent=IntentType.CONFIRM_YES, raw_text=raw)
        if lowered in {"no", "nope", "cancel", "stop", "abort", "never mind", "nevermind", "don't"}:
            return ParsedCommand(intent=IntentType.CONFIRM_NO, raw_text=raw)

        # 2. Assistant mute / unmute
        if lowered in {"mute listening", "pause listening", "mute microphone", "stop listening"}:
            return ParsedCommand(intent=IntentType.MUTE_ASSISTANT, raw_text=raw)
        if lowered in {"unmute listening", "resume listening", "unmute microphone", "start listening"}:
            return ParsedCommand(intent=IntentType.UNMUTE_ASSISTANT, raw_text=raw)

        # 3. Type text (preserve exact spelling/content)
        type_match = re.match(r"^(?:please\s+)?(?:type|write|enter\s+text)\s+(.+)$", raw.strip(), re.IGNORECASE)
        if type_match:
            typed_text = type_match.group(1).strip()
            if typed_text.endswith(".") and not typed_text.endswith(".."):
                typed_text = typed_text[:-1]
            return ParsedCommand(
                intent=IntentType.TYPE_TEXT,
                raw_text=raw,
                params={"text": typed_text},
            )

        # 4. Sensitive / destructive actions (always require explicit confirmation)
        if re.search(r"\b(shut\s*down|power\s*off|turn\s*off\s+computer)\b", lowered):
            return ParsedCommand(
                intent=IntentType.SENSITIVE_ACTION,
                raw_text=raw,
                params={"action": "shutdown"},
                requires_confirmation=True,
                confirmation_prompt="Are you sure you want to shut down the computer?",
            )
        if re.search(r"\b(restart|reboot)\b", lowered):
            return ParsedCommand(
                intent=IntentType.SENSITIVE_ACTION,
                raw_text=raw,
                params={"action": "restart"},
                requires_confirmation=True,
                confirmation_prompt="Are you sure you want to restart the computer?",
            )
        if re.search(r"\bdelete\s+(?:the\s+)?(?:file|folder|document)\b", lowered):
            return ParsedCommand(
                intent=IntentType.SENSITIVE_ACTION,
                raw_text=raw,
                params={"action": "delete_file"},
                requires_confirmation=True,
                confirmation_prompt="Are you sure you want to delete that file?",
            )

        # 5. YouTube search
        yt_search = re.match(
            r"^(?:please\s+)?search(?:\s+youtube)?\s+for\s+(.+)$",
            lowered,
        )
        if yt_search:
            query = yt_search.group(1).strip()
            return ParsedCommand(
                intent=IntentType.YOUTUBE_SEARCH,
                raw_text=raw,
                params={"query": query},
            )

        # 6. YouTube playback & navigation controls (check before ambiguous selection)
        if lowered in {"play video", "play the video", "start video"}:
            return ParsedCommand(intent=IntentType.YOUTUBE_PLAY, raw_text=raw)
        if lowered in {"pause video", "pause the video", "pause"}:
            return ParsedCommand(intent=IntentType.YOUTUBE_PAUSE, raw_text=raw)
        if lowered in {"resume video", "resume the video", "resume", "continue video", "unpause"}:
            return ParsedCommand(intent=IntentType.YOUTUBE_RESUME, raw_text=raw)

        # 7. YouTube video selection
        if re.match(r"^(?:select|open|click)\s+(?:the\s+)?(?:video|result)$", lowered):
            return ParsedCommand(
                intent=IntentType.AMBIGUOUS,
                raw_text=raw,
                clarification_prompt="Which video would you like me to select?",
            )

        select_match = re.match(
            r"^(?:please\s+)?(?:select|open|play|click)\s+(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|\d+)\s+(?:video|result)$",
            lowered,
        )
        if select_match:
            idx = ORDINAL_MAP.get(select_match.group(1), 1)
            return ParsedCommand(
                intent=IntentType.YOUTUBE_SELECT_VIDEO,
                raw_text=raw,
                params={"index": idx},
            )
        if lowered in {"go to next page", "go to the next page", "next page"}:
            return ParsedCommand(intent=IntentType.YOUTUBE_NEXT_PAGE, raw_text=raw)
        if lowered in {"go back to search results", "back to search results", "back to results"}:
            return ParsedCommand(intent=IntentType.YOUTUBE_BACK_TO_RESULTS, raw_text=raw)
        if lowered in {"go back", "back", "previous page"}:
            return ParsedCommand(intent=IntentType.BROWSER_BACK, raw_text=raw)

        # 8. Scroll & volume
        if lowered in {"scroll down", "page down"}:
            return ParsedCommand(intent=IntentType.SCROLL, raw_text=raw, params={"direction": "down"})
        if lowered in {"scroll up", "page up"}:
            return ParsedCommand(intent=IntentType.SCROLL, raw_text=raw, params={"direction": "up"})
        if lowered in {"increase volume", "volume up", "turn up volume", "louder"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "up"})
        if lowered in {"decrease volume", "volume down", "turn down volume", "quieter"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "down"})
        if lowered in {"mute", "mute audio", "mute volume"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "mute"})
        if lowered in {"unmute", "unmute audio", "unmute volume"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "unmute"})

        # 9. Desktop control commands
        desktop_patterns: list[tuple[str, str]] = [
            (r"^(?:minimize|minimise)\s+(?:the\s+)?(?:current\s+)?window$", "minimize_window"),
            (r"^(?:maximize|maximise)\s+(?:the\s+)?(?:current\s+)?window$", "maximize_window"),
            (r"^close\s+(?:the\s+)?(?:current\s+)?window$", "close_window"),
            (r"^(?:switch|next)\s+(?:application|app|window)$", "switch_application"),
            (r"^(?:open|show)\s+(?:the\s+)?desktop$", "show_desktop"),
            (r"^press\s+enter$", "press_enter"),
            (r"^press\s+escape$", "press_escape"),
            (r"^press\s+backspace$", "press_backspace"),
            (r"^copy(?:\s+selected\s+text)?$", "copy"),
            (r"^paste(?:\s+copied\s+text)?$", "paste"),
            (r"^select\s+all$", "select_all"),
        ]
        for pattern, action in desktop_patterns:
            if re.match(pattern, lowered):
                return ParsedCommand(
                    intent=IntentType.DESKTOP_ACTION,
                    raw_text=raw,
                    params={"action": action},
                )

        # 10. Open website or application
        open_match = re.match(r"^(?:please\s+)?(?:open|launch|start|go\s+to)\s+(?:the\s+)?(.+)$", lowered)
        if open_match:
            target = open_match.group(1).strip()

            # Check known websites first
            if target in self.known_websites:
                return ParsedCommand(
                    intent=IntentType.OPEN_WEBSITE,
                    raw_text=raw,
                    params={"name": target, "url": self.known_websites[target]},
                )

            # Explicit URL like example.com
            if re.match(r"^[a-z0-9-]+(\.[a-z0-9-]+)+(/.*)?$", target):
                url = target if target.startswith("http") else f"https://{target}"
                return ParsedCommand(
                    intent=IntentType.OPEN_WEBSITE,
                    raw_text=raw,
                    params={"name": target, "url": url},
                )

            # Check applications
            if target in APP_ALIASES:
                app_id = APP_ALIASES[target]
                requires_conf = app_id == "terminal"
                return ParsedCommand(
                    intent=IntentType.OPEN_APPLICATION,
                    raw_text=raw,
                    params={"app": app_id, "display_name": target},
                    requires_confirmation=requires_conf,
                    confirmation_prompt="Opening Terminal can run system commands. Are you sure you want to open Terminal?"
                    if requires_conf
                    else "",
                )

            # If user explicitly asked for a website that isn't recognized
            if "website" in target or "site" in target or target.startswith("www."):
                return ParsedCommand(
                    intent=IntentType.OPEN_WEBSITE_UNKNOWN,
                    raw_text=raw,
                    params={"target": target},
                    clarification_prompt=f"I don't recognize the website '{target}'. Could you please clarify the website name or URL?",
                )

            # Otherwise treat as generic installed application lookup
            return ParsedCommand(
                intent=IntentType.OPEN_APPLICATION,
                raw_text=raw,
                params={"app": target, "display_name": target},
            )

        return ParsedCommand(intent=IntentType.UNKNOWN, raw_text=raw, confidence=0.0)
