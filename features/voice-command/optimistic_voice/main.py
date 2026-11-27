"""Main entrypoint for Optimistic Voice-Controlled Smart Glasses Assistant."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimistic_voice.assistant.command_parser import CommandParser
from optimistic_voice.assistant.command_router import CommandRouter
from optimistic_voice.assistant.conversation_manager import ConversationManager
from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand
from optimistic_voice.assistant.response_manager import ResponseManager
from optimistic_voice.audio.microphone import MicrophoneStream
from optimistic_voice.audio.speech_recognition import SpeechRecognizer
from optimistic_voice.audio.text_to_speech import TextToSpeechEngine
from optimistic_voice.audio.voice_activity import VoiceActivityDetector
from optimistic_voice.audio.wake_word import WakeWordDetector
from optimistic_voice.automation import ExecutionResult
from optimistic_voice.config import OptimisticConfig
from optimistic_voice.display.display_manager import DisplayManager
from optimistic_voice.integration.communication import RemoteWindowsClient


class OptimisticAssistant:
    """End-to-end voice assistant orchestrating audio, NLU, automation, TTS, and display."""

    def __init__(
        self,
        config: OptimisticConfig | None = None,
        router: CommandRouter | None = None,
        tts: TextToSpeechEngine | None = None,
        display: DisplayManager | None = None,
        stt: SpeechRecognizer | None = None,
    ) -> None:
        self.config = config or OptimisticConfig()
        self.display = display or DisplayManager(
            mode=self.config.display_mode,
            width=self.config.display_width,
            height=self.config.display_height,
        )
        self.tts = tts or TextToSpeechEngine(
            engine=self.config.tts_engine,
            piper_model_path=self.config.piper_model_path,
            rate=self.config.tts_rate,
            volume=self.config.tts_volume,
        )
        remote_client = None
        if self.config.execution_mode == "remote_windows":
            remote_client = RemoteWindowsClient(
                host=self.config.remote_host,
                port=self.config.remote_port,
                auth_token=self.config.auth_token,
            )
        self.router = router or CommandRouter(self.config, remote_client=remote_client)
        self.parser = CommandParser(self.config.known_websites)
        self.responses = ResponseManager()
        self.conversation = ConversationManager()
        self.wake_word = WakeWordDetector(
            enabled=self.config.wake_word_enabled,
            wake_phrases=self.config.wake_phrases,
            session_timeout_s=self.config.wake_session_timeout_s,
        )
        self.stt = stt
        self.mic: MicrophoneStream | None = None
        self.vad = VoiceActivityDetector(
            sample_rate=self.config.sample_rate,
            energy_threshold=self.config.vad_energy_threshold,
            speech_pad_chunks=self.config.vad_speech_pad_chunks,
            silence_chunks=self.config.vad_silence_chunks,
            max_utterance_s=self.config.max_utterance_s,
            chunk_samples=self.config.chunk_samples,
            noise_reduction_enabled=self.config.noise_reduction_enabled,
        )
        self._running = False

    def startup(self) -> None:
        """Mandatory startup greeting spoken & displayed BEFORE activating listener."""
        self.display.show(self.config.startup_greeting, "Starting Optimistic...")
        self.tts.speak(self.config.startup_greeting, block=True)
        self.display.show(self.config.ready_prompt, "Initializing voice listener...")
        self.tts.speak(self.config.ready_prompt, block=True)

    def _speak_and_pause_mic(self, text: str, block: bool = True) -> None:
        if self.mic is not None:
            self.mic.pause()
        try:
            self.tts.speak(text, block=block)
        finally:
            if self.mic is not None:
                self.mic.resume()

    def process_utterance(self, transcript: str) -> ExecutionResult | None:
        """Process a recognized English utterance end-to-end."""
        should_process, cleaned, wake_triggered = self.wake_word.process_transcript(transcript)
        if wake_triggered and not should_process:
            self.display.show("Listening...", self.config.wake_response)
            self._speak_and_pause_mic(self.config.wake_response, block=True)
            return None
        if not should_process:
            return None

        self.display.show("Processing command...", cleaned)
        cmd = self.parser.parse(cleaned)

        # Handle pending confirmation turn
        if self.conversation.has_pending_confirmation():
            outcome, resolved_cmd = self.conversation.resolve_confirmation(cmd)
            if outcome == "cancelled":
                msg = "Action cancelled."
                self.display.show("Cancelled.", msg)
                self._speak_and_pause_mic(msg, block=True)
                self.display.show("Listening...", "")
                return ExecutionResult(success=True, message=msg)
            if outcome == "confirmed" and resolved_cmd is not None:
                return self._execute_confirmed(resolved_cmd)
            if resolved_cmd is not None:
                cmd = resolved_cmd

        # Handle mute / unmute assistant commands
        if cmd.intent == IntentType.MUTE_ASSISTANT:
            if self.mic is not None:
                self.mic.mute()
            msg = "Microphone muted."
            self.display.show("Muted", msg)
            self._speak_and_pause_mic(msg, block=True)
            return ExecutionResult(success=True, message=msg)

        if cmd.intent == IntentType.UNMUTE_ASSISTANT:
            if self.mic is not None:
                self.mic.unmute()
            msg = "Microphone unmuted. Listening."
            self.display.show("Listening...", msg)
            self._speak_and_pause_mic(msg, block=True)
            return ExecutionResult(success=True, message=msg)

        # Unrecognized or ambiguous commands
        if cmd.intent == IntentType.UNKNOWN:
            self.display.show("Command not recognized. Please try again.", cleaned)
            self._speak_and_pause_mic(self.responses.unknown_command_message(), block=True)
            self.display.show("Listening...", "")
            return ExecutionResult(success=False, message="Unknown command")

        if cmd.intent in (IntentType.AMBIGUOUS, IntentType.OPEN_WEBSITE_UNKNOWN):
            prompt = cmd.clarification_prompt or self.responses.unknown_command_message()
            self.display.show("Clarification needed", prompt)
            self._speak_and_pause_mic(prompt, block=True)
            self.display.show("Listening...", "")
            return ExecutionResult(success=False, message=prompt)

        # Sensitive / confirmation-required actions
        if cmd.requires_confirmation:
            self.conversation.set_pending_confirmation(cmd)
            self.display.show("Confirmation required", cmd.confirmation_prompt)
            self._speak_and_pause_mic(cmd.confirmation_prompt, block=True)
            self.display.show("Listening...", "Say yes or no")
            return ExecutionResult(success=True, message="Awaiting confirmation")

        return self._execute_confirmed(cmd)

    def _execute_confirmed(self, cmd: ParsedCommand) -> ExecutionResult:
        pre_msg = self.responses.pre_execution_message(cmd)
        self.display.show(pre_msg, cmd.raw_text)

        result = self.router.execute(cmd)
        self.conversation.update_context(cmd, result.success)

        if result.success:
            self._speak_and_pause_mic(pre_msg, block=True)
            self.display.show("Done.", pre_msg)
        else:
            err_msg = self.responses.failure_message(cmd, result.message)
            self.display.show("Action failed", err_msg)
            self._speak_and_pause_mic(err_msg, block=True)

        self.display.show("Listening...", "")
        return result

    def run_voice_loop(self) -> None:
        """Start continuous voice listening loop after mandatory startup greeting."""
        if self.stt is None:
            self.stt = SpeechRecognizer(
                engine=self.config.stt_engine,
                sample_rate=self.config.sample_rate,
                vosk_model_path=self.config.vosk_model_path,
                whisper_model_size=self.config.whisper_model_size,
                whisper_device=self.config.whisper_device,
                whisper_compute_type=self.config.whisper_compute_type,
                duplicate_cooldown_s=self.config.duplicate_command_cooldown_s,
            )

        self.startup()

        self.mic = MicrophoneStream(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_samples=self.config.chunk_samples,
            device=self.config.input_device,
        )
        self.mic.start()
        self._running = True
        self.display.show("Listening...", "Wake word active" if self.wake_word.enabled else "Continuous mode")

        try:
            while self._running:
                chunk = self.mic.read_chunk(timeout=0.2)
                if chunk is None:
                    continue
                utterance = self.vad.process_chunk(chunk)
                if utterance is None:
                    continue
                text = self.stt.transcribe(utterance)
                if text:
                    self.process_utterance(text)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def run_text_loop(self) -> None:
        """Interactive text-input loop (useful for testing without a physical mic)."""
        self.startup()
        self._running = True
        self.display.show("Listening...", "Text input mode")
        try:
            while self._running:
                user_input = input("Optimistic > ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    break
                if user_input:
                    self.process_utterance(user_input)
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._running = False
        if self.mic is not None:
            self.mic.stop()
        self.router.close()
        self.tts.shutdown()
        self.display.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimistic Voice-Controlled Smart Glasses Assistant")
    parser.add_argument("--mode", choices=["local", "remote_windows"], default=None)
    parser.add_argument("--wake-word", action="store_true", help="Enable 'Hey Optimistic' wake-word mode")
    parser.add_argument("--continuous", action="store_true", help="Enable continuous listening mode")
    parser.add_argument("--text-only", action="store_true", help="Run interactive text mode instead of microphone")
    parser.add_argument("--headless", action="store_true", help="Run browser and display in headless mode")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = OptimisticConfig()
    if args.mode:
        cfg.execution_mode = args.mode
    if args.wake_word:
        cfg.wake_word_enabled = True
    if args.continuous:
        cfg.wake_word_enabled = False
    if args.headless:
        cfg.browser_headless = True
        cfg.display_mode = "console"

    assistant = OptimisticAssistant(config=cfg)
    if args.text_only:
        assistant.run_text_loop()
    else:
        assistant.run_voice_loop()


if __name__ == "__main__":
    main()
