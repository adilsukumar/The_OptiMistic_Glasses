"""Formats natural spoken responses and display strings for every command."""

from __future__ import annotations

from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand

ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
}


class ResponseManager:
    """Generates natural spoken and visual responses for commands and outcomes."""

    @staticmethod
    def format_type_speech(text: str) -> str:
        # Speak "YouTube dot com" naturally while keeping typed text exact
        spoken = text.replace(".", " dot ")
        spoken = " ".join(spoken.split())
        return f"Typing {spoken}."

    def pre_execution_message(self, cmd: ParsedCommand) -> str:
        intent = cmd.intent
        p = cmd.params
        if intent == IntentType.OPEN_WEBSITE:
            name = p.get("name", "website").title()
            if name.lower() == "youtube":
                name = "YouTube"
            elif name.lower() == "chatgpt":
                name = "ChatGPT"
            elif name.lower() == "gmail":
                name = "Gmail"
            return f"Opening {name}."
        if intent == IntentType.TYPE_TEXT:
            return self.format_type_speech(p.get("text", ""))
        if intent == IntentType.OPEN_APPLICATION:
            return f"Opening {p.get('display_name', p.get('app', 'application')).title()}."
        if intent == IntentType.YOUTUBE_SEARCH:
            return f"Searching for {p.get('query', '')}."
        if intent == IntentType.YOUTUBE_SELECT_VIDEO:
            idx = int(p.get("index", 1))
            ordinal = ORDINAL_WORDS.get(idx, f"number {idx}")
            return f"Selecting the {ordinal} video."
        if intent == IntentType.YOUTUBE_PLAY:
            return "Playing the video."
        if intent == IntentType.YOUTUBE_PAUSE:
            return "Pausing the video."
        if intent == IntentType.YOUTUBE_RESUME:
            return "Resuming the video."
        if intent == IntentType.YOUTUBE_NEXT_PAGE:
            return "Going to the next page."
        if intent == IntentType.YOUTUBE_BACK_TO_RESULTS:
            return "Going back to search results."
        if intent == IntentType.BROWSER_BACK:
            return "Going back."
        if intent == IntentType.SCROLL:
            direction = p.get("direction", "down")
            return f"Scrolling {direction}."
        if intent == IntentType.VOLUME:
            action = p.get("action", "up")
            return {
                "up": "Increasing volume.",
                "down": "Decreasing volume.",
                "mute": "Muting audio.",
                "unmute": "Unmuting audio.",
            }.get(action, "Adjusting volume.")
        if intent == IntentType.DESKTOP_ACTION:
            action = p.get("action", "")
            mapping = {
                "minimize_window": "Minimizing the current window.",
                "maximize_window": "Maximizing the current window.",
                "close_window": "Closing the current window.",
                "switch_application": "Switching application.",
                "show_desktop": "Opening desktop.",
                "press_enter": "Pressing Enter.",
                "press_escape": "Pressing Escape.",
                "press_backspace": "Pressing Backspace.",
                "copy": "Copying selected text.",
                "paste": "Pasting copied text.",
                "select_all": "Selecting all.",
            }
            return mapping.get(action, "Executing desktop action.")
        return "Processing command."

    def failure_message(self, cmd: ParsedCommand, reason: str = "") -> str:
        if cmd.intent == IntentType.OPEN_WEBSITE:
            name = cmd.params.get("name", "the website")
            return f"I couldn't open {name}. Please try again."
        if reason:
            return reason
        return "Sorry, I couldn't complete that action. Please try again."

    @staticmethod
    def unknown_command_message() -> str:
        return "Sorry, I didn't understand. Could you repeat that?"
