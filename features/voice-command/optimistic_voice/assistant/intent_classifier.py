"""Intent definitions and natural language intent classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    OPEN_WEBSITE = "open_website"
    OPEN_WEBSITE_UNKNOWN = "open_website_unknown"
    TYPE_TEXT = "type_text"
    OPEN_APPLICATION = "open_application"
    YOUTUBE_SEARCH = "youtube_search"
    YOUTUBE_SELECT_VIDEO = "youtube_select_video"
    YOUTUBE_PLAY = "youtube_play"
    YOUTUBE_PAUSE = "youtube_pause"
    YOUTUBE_RESUME = "youtube_resume"
    YOUTUBE_NEXT_PAGE = "youtube_next_page"
    YOUTUBE_BACK_TO_RESULTS = "youtube_back_to_results"
    BROWSER_BACK = "browser_back"
    SCROLL = "scroll"
    VOLUME = "volume"
    DESKTOP_ACTION = "desktop_action"
    SENSITIVE_ACTION = "sensitive_action"
    MUTE_ASSISTANT = "mute_assistant"
    UNMUTE_ASSISTANT = "unmute_assistant"
    STOP_ASSISTANT = "stop_assistant"
    LLM_QUERY = "llm_query"
    CONFIRM_YES = "confirm_yes"
    CONFIRM_NO = "confirm_no"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Structured command produced by the NLU parser."""

    intent: IntentType
    raw_text: str
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    requires_confirmation: bool = False
    confirmation_prompt: str = ""
    clarification_prompt: str = ""
