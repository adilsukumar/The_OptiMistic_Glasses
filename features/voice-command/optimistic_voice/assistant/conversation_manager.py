"""Manages conversational state, pending confirmations, and clarifications."""

from __future__ import annotations

from dataclasses import dataclass

from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand


@dataclass
class ConversationState:
    pending_confirmation: ParsedCommand | None = None
    youtube_active: bool = False
    last_command: ParsedCommand | None = None


class ConversationManager:
    """Tracks multi-turn confirmation and context for Optimistic."""

    def __init__(self) -> None:
        self.state = ConversationState()

    def has_pending_confirmation(self) -> bool:
        return self.state.pending_confirmation is not None

    def set_pending_confirmation(self, cmd: ParsedCommand) -> None:
        self.state.pending_confirmation = cmd

    def resolve_confirmation(self, cmd: ParsedCommand) -> tuple[str, ParsedCommand | None]:
        """Resolve a pending confirmation given the user's new utterance.

        Returns:
            ("confirmed", original_cmd) | ("cancelled", None) | ("unrelated", cmd)
        """
        pending = self.state.pending_confirmation
        if pending is None:
            return "unrelated", cmd

        if cmd.intent == IntentType.CONFIRM_YES:
            self.state.pending_confirmation = None
            return "confirmed", pending

        if cmd.intent in (IntentType.CONFIRM_NO, IntentType.STOP_ASSISTANT):
            self.state.pending_confirmation = None
            return "cancelled", None

        # User said something else; keep or clear depending on clarity
        self.state.pending_confirmation = None
        return "unrelated", cmd

    def update_context(self, cmd: ParsedCommand, success: bool) -> None:
        if not success:
            return
        self.state.last_command = cmd
        if cmd.intent in (
            IntentType.YOUTUBE_SEARCH,
            IntentType.YOUTUBE_SELECT_VIDEO,
            IntentType.YOUTUBE_PLAY,
            IntentType.YOUTUBE_PAUSE,
            IntentType.YOUTUBE_RESUME,
        ):
            self.state.youtube_active = True
        elif cmd.intent == IntentType.OPEN_WEBSITE and cmd.params.get("name") == "youtube":
            self.state.youtube_active = True
