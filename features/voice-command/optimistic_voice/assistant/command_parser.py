"""Natural language intent interpretation and command parser for Optimistic voice commands."""

from __future__ import annotations

import difflib
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
    # Notepad (Matched only when preceded by open/launch/start verbs)
    "notepad": "notepad",
    "note pad": "notepad",
    "note-pad": "notepad",
    "notes pad": "notepad",
    "notes": "notepad",
    "note": "notepad",
    "pad": "notepad",
    "the note": "notepad",
    "the pad": "notepad",
    "my note": "notepad",
    "my notes": "notepad",
    "text editor": "notepad",
    "editor": "notepad",
    "text pad": "notepad",
    "writing pad": "notepad",
    "notebook": "notepad",
    "memo": "notepad",
    "note app": "notepad",
    "notepad app": "notepad",
    "note bad": "notepad",
    "not bad": "notepad",
    "node pad": "notepad",
    "no pad": "notepad",
    "know pad": "notepad",
    "not pad": "notepad",
    "load pad": "notepad",
    "road pad": "notepad",
    "note pat": "notepad",
    "note back": "notepad",
    "note pack": "notepad",

    # Calculator
    "calculator": "calculator",
    "calculater": "calculator",
    "culculator": "calculator",
    "kalkulator": "calculator",
    "calculat": "calculator",
    "calc": "calculator",
    "calci": "calculator",
    "cal c": "calculator",
    "cal cu": "calculator",
    "cal": "calculator",

    # File Manager
    "file manager": "file_manager",
    "file explorer": "file_manager",
    "files": "file_manager",
    "my files": "file_manager",
    "my documents": "file_manager",
    "documents": "file_manager",
    "explorer": "file_manager",
    "folders": "file_manager",
    "my folders": "file_manager",

    # Terminal
    "terminal": "terminal",
    "command prompt": "terminal",
    "cmd": "terminal",
    "powershell": "terminal",
    "power shell": "terminal",
    "console": "terminal",
    "shell": "terminal",

    # Browser
    "browser": "browser",
    "web browser": "browser",
    "internet browser": "browser",
    "chrome": "browser",
    "google chrome": "browser",
    "edge": "browser",
    "microsoft edge": "browser",
    "firefox": "browser",
}

# Conversational prefixes and request wrappers that should be stripped for clean intent matching
POLITE_PREFIX_PATTERN = re.compile(
    r"^(?:please\s+|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|would\s+you\s+(?:mind\s+)?|"
    r"i\s+want\s+to\s+|i\s+want\s+you\s+to\s+|i'd\s+like\s+to\s+|i\s+would\s+like\s+to\s+|"
    r"i\s+need\s+to\s+|i\s+need\s+you\s+to\s+|let's\s+|can\s+we\s+|help\s+me\s+|go\s+ahead\s+and\s+)+",
    re.IGNORECASE,
)

# Trailing courteous phrases
POLITE_SUFFIX_PATTERN = re.compile(
    r"(?:,\s*|\s+)(?:for\s+me|please|kindly|thanks|thank\s+you)$",
    re.IGNORECASE,
)

# Fillers and hesitations
FILLER_PATTERN = re.compile(r"\b(um|uh|er|ah|like|you\s+know)\b", re.IGNORECASE)

# Wake prefix pattern to strip if passed into parser
WAKE_PREFIX_PATTERN = re.compile(
    r"^(?:(?:hey|hi|hello|ok|okay|yo|dear)\s+)?"
    r"(?:optimi[sz]t(?:ic(?:s|al)?)?|opti[- ']?m[yi]stic(?:s)?|opt[- ']?m[yi]stic(?:s)?|"
    r"optic[- ']?m[yi]stic(?:s)?|opty?mistic|optemistic|optima[- ]?stick|"
    r"up[- ]?(?:to|the)?[- ]?m[yi]stic|app[- ]?(?:to|the)?[- ]?m[yi]stic|"
    r"apt[- ]?(?:to|the)?[- ]?m[yi]stic|off[- ]?(?:the)?[- ]?m[yi]stic|"
    r"auto[- ]?m[yi]stic|atomistic|optimus)\b[,:\s]*",
    re.IGNORECASE,
)

# Question and conversational intent patterns routed to LLM
QUESTION_OR_CONVERSATIONAL_REGEX = re.compile(
    r"^(?:"
    r"what|who|why|where|when|which|how|whose|whom|"
    r"is|are|was|were|do|does|did|can|could|would|should|will|have|has|had|"
    r"tell\s+me|explain|describe|define|calculate|summarize|translate|give\s+me|write|compose|help\s+me|"
    r"hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|howdy|greetings|"
    r"how\s+are\s+you|who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|"
    r"tell\s+me\s+a\s+joke|make\s+me\s+laugh|say\s+something\s+funny|joke|"
    r"thank\s+you|thanks|thanks\s+a\s+lot|bye|goodbye"
    r")\b",
    re.IGNORECASE,
)


class CommandParser:
    """Robust Natural Language Intent Interpreter for Optimistic voice commands."""

    def __init__(
        self,
        known_websites: Mapping[str, str] | None = None,
        enable_llm: bool = True,
    ) -> None:
        self.known_websites = {
            k.lower(): v for k, v in (known_websites or OptimisticConfig().known_websites).items()
        }
        self.enable_llm = enable_llm

    @staticmethod
    def _clean(text: str) -> str:
        cleaned = " ".join(text.strip().split())
        return cleaned.rstrip("?!.") if cleaned.endswith(("?", "!")) else cleaned

    def _strip_conversational_wrappers(self, text: str) -> str:
        s = " ".join(text.split()).strip(".,!?\"'")
        m = WAKE_PREFIX_PATTERN.match(s)
        if m:
            s = s[m.end():].strip()
        else:
            # Fuzzy wake prefix strip if leading words resemble "optimistic"
            lead_m = re.match(r"^(?:hey|hi|hello|ok|okay|yo|dear)\s+", s, re.IGNORECASE)
            sub_s = s[lead_m.end():].strip() if lead_m else s
            words = sub_s.split()
            targets = ("optimistic", "optimist", "optimystic", "optimistics")
            for n in (1, 2, 3):
                if len(words) >= n:
                    cand = "".join(w.lower().strip(" ,.!?\"'") for w in words[:n])
                    if any(difflib.SequenceMatcher(None, t, cand).ratio() >= 0.72 for t in targets):
                        s = " ".join(words[n:]).strip()
                        break
        s = FILLER_PATTERN.sub(" ", s)
        s = POLITE_PREFIX_PATTERN.sub("", s)
        s = POLITE_SUFFIX_PATTERN.sub("", s)
        s = re.sub(r"[,;]+", " ", s)
        return " ".join(s.split()).strip()

    def parse(self, text: str) -> ParsedCommand:
        raw = text.strip()
        if not raw:
            return ParsedCommand(intent=IntentType.UNKNOWN, raw_text=raw, confidence=0.0)

        lowered = " ".join(raw.lower().split()).rstrip(".,!?\"'")
        cleaned = self._strip_conversational_wrappers(lowered)

        # ---------------------------------------------------------
        # 1. Exact Text Typing (Must preserve verbatim text/spelling)
        # ---------------------------------------------------------
        type_match = re.match(
            r"^(?:(?:hey|hi|hello|ok|okay|yo|dear)\s+)?"
            r"(?:optimi[sz]t(?:ic(?:s|al)?)?|opti[- ']?m[yi]stic(?:s)?|opt[- ']?m[yi]stic(?:s)?|"
            r"optic[- ']?m[yi]stic(?:s)?|opty?mistic|optemistic|optima[- ]?stick|"
            r"up[- ]?(?:to|the)?[- ]?m[yi]stic|app[- ]?(?:to|the)?[- ]?m[yi]stic|"
            r"apt[- ]?(?:to|the)?[- ]?m[yi]stic|off[- ]?(?:the)?[- ]?m[yi]stic|"
            r"auto[- ]?m[yi]stic|atomistic|optimus)?\b[,:\s]*"
            r"(?:please\s+|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?)*"
            r"(?:(?:open\s+(?:the\s+)?notepad\s+and\s+)?(?:type|write|enter\s+text|type\s+out)|type)\s+(.+)$",
            raw.strip(),
            re.IGNORECASE,
        )
        if type_match:
            typed_text = type_match.group(1).strip()

            # If user asks to "write a poem / essay / joke / story / code / email", route to LLM
            if re.match(
                r"^(?:a|an|me|some)\s+(?:poem|joke|story|essay|email|letter|song|script|code|summary|paragraph|quote)\b",
                typed_text,
                re.IGNORECASE,
            ):
                if self.enable_llm:
                    return ParsedCommand(
                        intent=IntentType.LLM_QUERY,
                        raw_text=raw,
                        params={"query": f"write {typed_text}"},
                        confidence=0.95,
                    )

            # If the user asks to type into the YouTube search bar, route to YOUTUBE_SEARCH
            yt_lead = re.match(
                r"^(?:in(?:to)?\s+(?:the\s+)?youtube(?:\s+search(?:\s+bar)?)?|on\s+youtube(?:\s+search(?:\s+bar)?)?)\s+(.+)$",
                typed_text,
                re.IGNORECASE,
            )
            if yt_lead:
                q = yt_lead.group(1).strip()
                q = re.sub(r"(?:,\s*|\s+)(?:for\s+me|please)$", "", q, flags=re.IGNORECASE).strip()
                return ParsedCommand(
                    intent=IntentType.YOUTUBE_SEARCH,
                    raw_text=raw,
                    params={"query": q},
                    confidence=0.98,
                )
            yt_trail = re.match(
                r"^(.+?)\s+(?:in(?:to)?\s+(?:the\s+)?youtube(?:\s+search(?:\s+bar)?)?|on\s+youtube(?:\s+search(?:\s+bar)?)?)$",
                typed_text,
                re.IGNORECASE,
            )
            if yt_trail:
                q = yt_trail.group(1).strip()
                q = re.sub(r"(?:,\s*|\s+)(?:for\s+me|please)$", "", q, flags=re.IGNORECASE).strip()
                return ParsedCommand(
                    intent=IntentType.YOUTUBE_SEARCH,
                    raw_text=raw,
                    params={"query": q},
                    confidence=0.98,
                )

            # Strip leading/trailing notepad references if mentioned in command
            np_lead = re.match(r"^(?:(?:in|on)\s+(?:the\s+)?(?:notepad|note|pad))\s+(.+)$", typed_text, re.IGNORECASE)
            if np_lead:
                typed_text = np_lead.group(1).strip()
            np_trail = re.match(r"^(.+?)\s+(?:(?:in|on)\s+(?:the\s+)?(?:notepad|note|pad))$", typed_text, re.IGNORECASE)
            if np_trail:
                typed_text = np_trail.group(1).strip()

            # Strip trailing courtesy words from typing if explicitly stated
            typed_text = re.sub(r"(?:,\s*|\s+)(?:for\s+me|please)$", "", typed_text, flags=re.IGNORECASE).strip()
            if typed_text.endswith(".") and not typed_text.endswith(".."):
                typed_text = typed_text[:-1]
            return ParsedCommand(
                intent=IntentType.TYPE_TEXT,
                raw_text=raw,
                params={"text": typed_text, "target_app": "notepad"},
                confidence=0.98,
            )

        # ---------------------------------------------------------
        # 2. Stop & Confirmation responses
        # ---------------------------------------------------------
        if cleaned in {"stop", "cancel", "quiet", "shut up", "stop talking", "be quiet", "halt", "stop speaking", "pause talking", "stop it"}:
            return ParsedCommand(intent=IntentType.STOP_ASSISTANT, raw_text=raw, confidence=0.99)

        if cleaned in {"yes", "yeah", "yep", "confirm", "do it", "proceed", "sure", "ok", "okay", "yes please", "yes do it"}:
            return ParsedCommand(intent=IntentType.CONFIRM_YES, raw_text=raw, confidence=0.99)
        if cleaned in {"no", "nope", "abort", "never mind", "nevermind", "don't", "no cancel"}:
            return ParsedCommand(intent=IntentType.CONFIRM_NO, raw_text=raw, confidence=0.99)

        # ---------------------------------------------------------
        # 3. Assistant mute / unmute
        # ---------------------------------------------------------
        if cleaned in {"mute listening", "pause listening", "mute microphone", "stop listening", "go to sleep"}:
            return ParsedCommand(intent=IntentType.MUTE_ASSISTANT, raw_text=raw, confidence=0.95)
        if cleaned in {"unmute listening", "resume listening", "unmute microphone", "start listening", "wake up"}:
            return ParsedCommand(intent=IntentType.UNMUTE_ASSISTANT, raw_text=raw, confidence=0.95)

        # ---------------------------------------------------------
        # 4. Sensitive / destructive actions (Require explicit confirmation)
        # ---------------------------------------------------------
        if re.search(r"\b(shut\s*down|power\s*off|turn\s*off)\s*(?:the\s+)?(?:computer|pc|system)?\b", cleaned):
            return ParsedCommand(
                intent=IntentType.SENSITIVE_ACTION,
                raw_text=raw,
                params={"action": "shutdown"},
                requires_confirmation=True,
                confirmation_prompt="Are you sure you want to shut down the computer?",
                confidence=0.95,
            )
        if re.search(r"\b(restart|reboot)\s*(?:the\s+)?(?:computer|pc|system)?\b", cleaned):
            return ParsedCommand(
                intent=IntentType.SENSITIVE_ACTION,
                raw_text=raw,
                params={"action": "restart"},
                requires_confirmation=True,
                confirmation_prompt="Are you sure you want to restart the computer?",
                confidence=0.95,
            )
        if re.search(r"\bdelete\s+(?:the\s+)?(?:file|folder|document)\b", cleaned):
            return ParsedCommand(
                intent=IntentType.SENSITIVE_ACTION,
                raw_text=raw,
                params={"action": "delete_file"},
                requires_confirmation=True,
                confirmation_prompt="Are you sure you want to delete that file?",
                confidence=0.95,
            )

        # ---------------------------------------------------------
        # 5. YouTube Search & Type in YouTube Search Bar
        # ---------------------------------------------------------
        # A. Explicit typing in YouTube search bar:
        type_yt_match = re.match(
            r"^(?:type|enter|write)\s+(?:in(?:to)?\s+(?:the\s+)?youtube\s+search(?:\s+bar)?\s+)?(.+?)(?:\s+in(?:to)?\s+(?:the\s+)?youtube(?:\s+search(?:\s+bar)?)?)?$",
            cleaned,
            re.IGNORECASE,
        )
        if type_yt_match and re.search(r"\byoutube\b", cleaned):
            yt_q = type_yt_match.group(1).strip()
            yt_q = re.sub(r"\b(?:in(?:to)?\s+(?:the\s+)?youtube(?:\s+search(?:\s+bar)?)?|on\s+youtube)\b", "", yt_q, flags=re.IGNORECASE).strip()
            if yt_q:
                return ParsedCommand(
                    intent=IntentType.YOUTUBE_SEARCH,
                    raw_text=raw,
                    params={"query": yt_q},
                    confidence=0.98,
                )

        # B. "open youtube and search for X" / "go to youtube and search for X"
        open_and_search = re.match(
            r"^(?:open|go\s+to|launch)\s+youtube\s+(?:and\s+)?(?:search(?:\s+for)?|type|find|look\s+up)\s+(.+)$",
            cleaned,
            re.IGNORECASE,
        )
        if open_and_search:
            yt_q = open_and_search.group(1).strip()
            return ParsedCommand(
                intent=IntentType.YOUTUBE_SEARCH,
                raw_text=raw,
                params={"query": yt_q},
                confidence=0.98,
            )

        # C. General search patterns:
        yt_patterns = [
            r"^(?:search(?:\s+youtube)?\s+for|search\s+for)\s+(.+?)(?:\s+(?:on|in)\s+youtube)?$",
            r"^(?:search\s+on\s+youtube\s+for|search\s+on\s+youtube)\s+(.+)$",
            r"^search\s+youtube\s+(.+)$",
            r"^search\s+(.+?)\s+(?:on|in)\s+youtube$",
            r"^youtube\s+search(?:\s+for)?\s+(.+)$",
            r"^(?:look\s+up|find)\s+(.+?)\s+(?:on|in)\s+youtube$",
            r"^play\s+(.+?)\s+(?:on|in)\s+youtube$",
        ]
        for pat in yt_patterns:
            m = re.match(pat, cleaned, re.IGNORECASE)
            if m:
                yt_q = m.group(1).strip()
                yt_q = re.sub(r"^(?:videos?\s+about|some)\s+", "", yt_q, flags=re.IGNORECASE).strip()
                yt_q = re.sub(r"\b(?:on|in)\s+youtube\b", "", yt_q, flags=re.IGNORECASE).strip()
                if yt_q:
                    return ParsedCommand(
                        intent=IntentType.YOUTUBE_SEARCH,
                        raw_text=raw,
                        params={"query": yt_q},
                        confidence=0.96,
                    )

        # ---------------------------------------------------------
        # 6. YouTube playback & navigation controls
        # ---------------------------------------------------------
        # Play / Resume
        if (
            re.search(r"^(?:please\s+)?(?:resume|unpause)\s+(?:the\s+|this\s+)?(?:current\s+|playing\s+)?(?:youtube\s+)?(?:video|playback|stream|song|clip)?$", cleaned)
            or re.search(r"^(?:please\s+)?(?:play|start|continue)\s+(?:the\s+|this\s+)?(?:current\s+|playing\s+)?(?:youtube\s+)?(?:video|playback|stream|song|clip)$", cleaned)
            or cleaned in {"play video", "play the video", "start video", "start the video", "resume video", "resume the video", "resume", "continue video", "unpause", "play this video", "resume this video"}
        ):
            if "resume" in cleaned or "unpause" in cleaned:
                return ParsedCommand(intent=IntentType.YOUTUBE_RESUME, raw_text=raw, confidence=0.95)
            return ParsedCommand(intent=IntentType.YOUTUBE_PLAY, raw_text=raw, confidence=0.95)

        # Pause
        if (
            re.search(r"^(?:please\s+)?pause\s+(?:the\s+|this\s+)?(?:current\s+|playing\s+)?(?:youtube\s+)?(?:video|playback|stream|song|clip)?$", cleaned)
            or re.search(r"^(?:please\s+)?stop\s+(?:the\s+|this\s+)?(?:current\s+|playing\s+)?(?:youtube\s+)?(?:video|playback|stream|song|clip)$", cleaned)
            or cleaned in {"pause video", "pause the video", "pause this video", "pause", "stop video", "stop the video", "stop playing", "pause playback", "pause the playback"}
        ):
            return ParsedCommand(intent=IntentType.YOUTUBE_PAUSE, raw_text=raw, confidence=0.98)

        # ---------------------------------------------------------
        # 7. YouTube video selection
        # ---------------------------------------------------------
        if re.match(r"^(?:select|open|click|pick)\s+(?:the\s+)?(?:video|result)$", cleaned):
            return ParsedCommand(
                intent=IntentType.AMBIGUOUS,
                raw_text=raw,
                clarification_prompt="Which video would you like me to select?",
                confidence=0.90,
            )

        select_match = re.match(
            r"^(?:select|open|play|click|pick|choose)\s+(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|\d+)(?:\s+(?:video|result|one))?$",
            cleaned,
        )
        if not select_match:
            select_match = re.match(
                r"^(?:select|open|play|click|pick|choose)\s+(?:the\s+)?(?:video|result)\s+(?:number\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|\d+)$",
                cleaned,
            )
        if select_match:
            idx_str = select_match.group(1)
            idx = ORDINAL_MAP.get(idx_str, int(idx_str) if idx_str.isdigit() else 1)
            return ParsedCommand(
                intent=IntentType.YOUTUBE_SELECT_VIDEO,
                raw_text=raw,
                params={"index": idx},
                confidence=0.95,
            )

        if cleaned in {"go to next page", "go to the next page", "next page"}:
            return ParsedCommand(intent=IntentType.YOUTUBE_NEXT_PAGE, raw_text=raw, confidence=0.95)
        if cleaned in {"go back to search results", "back to search results", "back to results"}:
            return ParsedCommand(intent=IntentType.YOUTUBE_BACK_TO_RESULTS, raw_text=raw, confidence=0.95)
        if cleaned in {"go back", "back", "previous page"}:
            return ParsedCommand(intent=IntentType.BROWSER_BACK, raw_text=raw, confidence=0.95)

        # ---------------------------------------------------------
        # 8. Scroll & volume
        # ---------------------------------------------------------
        if cleaned in {"scroll down", "page down", "move down"}:
            return ParsedCommand(intent=IntentType.SCROLL, raw_text=raw, params={"direction": "down"}, confidence=0.95)
        if cleaned in {"scroll up", "page up", "move up"}:
            return ParsedCommand(intent=IntentType.SCROLL, raw_text=raw, params={"direction": "up"}, confidence=0.95)
        if cleaned in {"increase volume", "volume up", "turn up volume", "louder"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "up"}, confidence=0.95)
        if cleaned in {"decrease volume", "volume down", "turn down volume", "quieter"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "down"}, confidence=0.95)
        if cleaned in {"mute", "mute audio", "mute volume"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "mute"}, confidence=0.95)
        if cleaned in {"unmute", "unmute audio", "unmute volume"}:
            return ParsedCommand(intent=IntentType.VOLUME, raw_text=raw, params={"action": "unmute"}, confidence=0.95)

        # ---------------------------------------------------------
        # 9. Desktop control commands
        # ---------------------------------------------------------
        desktop_patterns: list[tuple[str, str]] = [
            (r"^(?:minimize|minimise)\s+(?:the\s+)?(?:current\s+|this\s+)?(?:youtube\s+|video\s+)?(?:window|tab)$", "minimize_window"),
            (r"^(?:minimize|minimise)\s+it$", "minimize_window"),
            (r"^(?:maximize|maximise)\s+(?:the\s+)?(?:current\s+|this\s+)?(?:youtube\s+|video\s+)?(?:window|tab)$", "maximize_window"),
            (r"^(?:maximize|maximise)\s+it$", "maximize_window"),
            (r"^close\s+(?:the\s+|this\s+)?(?:current\s+)?(?:youtube\s+|video\s+)?(?:window|winodw|tab)$", "close_window"),
            (r"^close\s+(?:the\s+|this\s+)?(?:youtube|video)\s+(?:window|winodw|tab)$", "close_window"),
            (r"^close\s+it$", "close_window"),
            (r"^(?:switch|next)\s+(?:application|app|window)$", "switch_application"),
            (r"^(?:open|show|go\s+to)\s+(?:the\s+)?desktop$", "show_desktop"),
            (r"^press\s+enter$", "press_enter"),
            (r"^press\s+escape$", "press_escape"),
            (r"^press\s+backspace$", "press_backspace"),
            (r"^copy(?:\s+selected\s+text|\s+this)?$", "copy"),
            (r"^paste(?:\s+copied\s+text|\s+this)?$", "paste"),
            (r"^select\s+all$", "select_all"),
        ]
        for pattern, action in desktop_patterns:
            if re.match(pattern, cleaned):
                return ParsedCommand(
                    intent=IntentType.DESKTOP_ACTION,
                    raw_text=raw,
                    params={"action": action},
                    confidence=0.95,
                )

        # ---------------------------------------------------------
        # 10. Natural Website and Application Intent Matching
        # ---------------------------------------------------------
        # First, check for known websites in the natural command
        # Handles: "Open Google", "Can you open Google for me?", "Go to Google",
        # "Launch my browser and search Google", "I want to check something on Google", etc.
        for site_key, site_url in sorted(self.known_websites.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = rf"\b{re.escape(site_key)}(?:\.com|\s+com|\s+dot\s+com)?\b"
            if re.search(pattern, cleaned, re.IGNORECASE):
                # If YouTube is mentioned with search verbs, route to search
                if ("youtube" in site_key or "tube" in site_key) and re.search(r"\b(search|find|look\s+up|play)\b", cleaned):
                    # extract search query if present
                    q_match = re.search(r"\b(?:search\s+for|find|play|look\s+up)\s+(.+?)(?:\s+on\s+youtube)?$", cleaned)
                    if q_match and q_match.group(1).strip() not in ("youtube", "you tube"):
                        return ParsedCommand(
                            intent=IntentType.YOUTUBE_SEARCH,
                            raw_text=raw,
                            params={"query": q_match.group(1).strip()},
                            confidence=0.95,
                        )
                canonical_name = site_key
                if "google.com" in site_url and "mail" not in site_url and "maps" not in site_url:
                    canonical_name = "google"
                elif "youtube.com" in site_url:
                    canonical_name = "youtube"
                elif "chatgpt.com" in site_url:
                    canonical_name = "chatgpt"
                elif "facebook.com" in site_url:
                    canonical_name = "facebook"
                elif "netflix.com" in site_url:
                    canonical_name = "netflix"
                elif "github.com" in site_url:
                    canonical_name = "github"
                elif "linkedin.com" in site_url:
                    canonical_name = "linkedin"
                return ParsedCommand(
                    intent=IntentType.OPEN_WEBSITE,
                    raw_text=raw,
                    params={"name": canonical_name, "url": site_url},
                    confidence=0.96,
                )

        # Check explicit URL patterns like example.com or http://...
        url_match = re.search(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?:/.*)?\b", cleaned)
        if url_match:
            domain = url_match.group(1)
            full_url = f"https://{domain}"
            return ParsedCommand(
                intent=IntentType.OPEN_WEBSITE,
                raw_text=raw,
                params={"name": domain, "url": full_url},
                confidence=0.95,
            )

        # Application Launch Matching
        # An application MUST only be launched if the user explicitly commanded:
        # - "open <app>" / "launch <app>" / "start <app>" / "bring up <app>" / "run <app>"
        # - Or the command consists solely of the app name (e.g. "Notepad", "Calculator")
        open_app_match = re.match(
            r"^(?:open|launch|start|run|bring\s+up|open\s+up|switch\s+to|take\s+(?:some\s+)?notes\s+in)\s+(?:the\s+)?(?:application\s+|app\s+)?(.+)$",
            cleaned,
            re.IGNORECASE,
        )
        app_candidate = open_app_match.group(1).strip() if open_app_match else (cleaned if cleaned in APP_ALIASES else None)

        if app_candidate:
            # If user explicitly asked for an unknown website
            if "website" in app_candidate or "site" in app_candidate or app_candidate.startswith("www."):
                return ParsedCommand(
                    intent=IntentType.OPEN_WEBSITE_UNKNOWN,
                    raw_text=raw,
                    params={"target": app_candidate},
                    clarification_prompt=f"I don't recognize the website '{app_candidate}'. Could you please clarify the website name or URL?",
                    confidence=0.85,
                )

            # Match against known application aliases
            for alias, app_id in sorted(APP_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
                if alias == app_candidate or re.search(rf"\b{re.escape(alias)}\b", app_candidate, re.IGNORECASE):
                    requires_conf = app_id == "terminal"
                    display_name = "Notepad" if app_id == "notepad" else (
                        "Calculator" if app_id == "calculator" else alias.title()
                    )
                    return ParsedCommand(
                        intent=IntentType.OPEN_APPLICATION,
                        raw_text=raw,
                        params={"app": app_id, "display_name": display_name},
                        requires_confirmation=requires_conf,
                        confirmation_prompt="Opening Terminal can run system commands. Are you sure you want to open Terminal?"
                        if requires_conf
                        else "",
                        confidence=0.96,
                    )

            # If user explicitly gave an open command but target isn't pre-configured, launch generic app
            if open_app_match:
                return ParsedCommand(
                    intent=IntentType.OPEN_APPLICATION,
                    raw_text=raw,
                    params={"app": app_candidate, "display_name": app_candidate.title()},
                    confidence=0.75,
                )

        # Conversational questions, chit-chat, or general knowledge queries routed to LLM
        if self.enable_llm:
            raw_no_wake = WAKE_PREFIX_PATTERN.sub("", raw).strip(" .,!?:")
            if (
                QUESTION_OR_CONVERSATIONAL_REGEX.search(raw_no_wake)
                or QUESTION_OR_CONVERSATIONAL_REGEX.search(cleaned)
                or len(cleaned.split()) >= 2
            ):
                query_text = raw_no_wake if raw_no_wake else cleaned
                return ParsedCommand(
                    intent=IntentType.LLM_QUERY,
                    raw_text=raw,
                    params={"query": query_text},
                    confidence=0.95,
                )

        # Fallback: Unrecognized command with low confidence
        return ParsedCommand(
            intent=IntentType.UNKNOWN,
            raw_text=raw,
            confidence=0.0,
            clarification_prompt=f"I heard '{raw}', but I didn't recognize that command. Could you please repeat?",
        )
