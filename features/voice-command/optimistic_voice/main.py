"""Main entrypoint for Optimistic Voice-Controlled Smart Glasses Assistant."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimistic_voice.assistant.command_parser import CommandParser
from optimistic_voice.assistant.command_router import CommandRouter
from optimistic_voice.assistant.conversation_manager import ConversationManager
from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand
from optimistic_voice.assistant.response_manager import ResponseManager
from optimistic_voice.audio.microphone import MicrophoneStream, list_audio_devices
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
        debug: bool = False,
    ) -> None:
        self.debug = debug
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
            output_device=self.config.tts_output_device,
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
        self._stop_vosk_model: Any = None
        self._init_stop_recognizer()

    def _init_stop_recognizer(self) -> None:
        """Pre-load lightweight Vosk model for instant sub-millisecond barge-in stop recognition."""
        try:
            model_path = self.config.vosk_model_path
            if model_path and Path(model_path).exists():
                import vosk
                vosk.SetLogLevel(-1)
                self._stop_vosk_model = vosk.Model(str(model_path))
        except Exception:
            self._stop_vosk_model = None

    def startup(self) -> None:
        """Mandatory startup greeting spoken & displayed BEFORE activating listener."""
        print(f"\n[OptiMistic] Audio Output: {self.tts.active_output_name}", flush=True)
        print(f"[OptiMistic] {self.config.startup_greeting}", flush=True)
        self.display.show(self.config.startup_greeting, "Starting OptiMistic...")
        self.tts.speak(self.config.startup_greeting, block=True)
        time.sleep(0.3)
        print(f"[OptiMistic] {self.config.ready_prompt}", flush=True)
        self.display.show(self.config.ready_prompt, "Initializing voice listener...")
        self.tts.speak(self.config.ready_prompt, block=True)
        time.sleep(0.25)

    def _speak_and_pause_mic(self, text: str, block: bool = True) -> bool:
        """Speak vocal output while actively monitoring for 'optimistic stop' to cut off speech instantly."""
        # If mic is not running (e.g. startup greeting or text mode), speak directly
        if self.mic is None or not self.mic.is_running:
            self.tts.speak(text, block=block)
            return True

        # Ensure mic is unpaused to capture the user's voice
        if self.mic.is_paused:
            self.mic.resume(settle_delay_s=0.0)

        # Enqueue speech asynchronously
        self.tts.speak(text, block=False)
        time.sleep(0.06)

        # Real-time stop recognizer
        stop_rec = None
        if self._stop_vosk_model is not None:
            try:
                import vosk
                grammar = json.dumps(["optimistic", "stop", "cancel", "quiet", "shut up", "halt", "[unk]"])
                stop_rec = vosk.KaldiRecognizer(self._stop_vosk_model, self.config.sample_rate, grammar)
            except Exception:
                stop_rec = None

        interrupted = False
        try:
            while self.tts.is_speaking and self._running:
                chunk = self.mic.read_chunk(timeout=0.04)
                if chunk is None:
                    continue

                if stop_rec is not None:
                    c_int16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                    matched = False
                    if stop_rec.AcceptWaveform(c_int16.tobytes()):
                        res = json.loads(stop_rec.Result()).get("text", "").lower()
                        if any(w in res for w in ("stop", "cancel", "quiet", "shut up", "halt")):
                            matched = True
                    else:
                        part = json.loads(stop_rec.PartialResult()).get("partial", "").lower()
                        if any(w in part for w in ("stop", "cancel", "quiet", "shut up", "halt")):
                            matched = True

                    if matched:
                        print("\n[OptiMistic] Stop command detected! Stopping speech immediately.", flush=True)
                        self.tts.stop()
                        interrupted = True
                        self.display.show("Stopped.", "Listening...")
                        break
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Stop monitor error: {e}", flush=True)
        finally:
            if interrupted:
                self.tts.stop()
                if self.mic is not None:
                    self.mic.resume(settle_delay_s=0.20)
            else:
                if block:
                    self.tts.wait_until_done()
                if self.mic is not None:
                    self.mic.resume(settle_delay_s=0.15)

        return not interrupted

    def process_utterance(self, transcript: str) -> ExecutionResult | None:
        """Process a recognized English utterance end-to-end."""
        should_process, cleaned, wake_triggered = self.wake_word.process_transcript(transcript)
        if wake_triggered and not should_process:
            print(f"[OptiMistic] {self.config.wake_response}", flush=True)
            self.display.show("Listening...", self.config.wake_response)
            self._speak_and_pause_mic(self.config.wake_response, block=True)
            print("[OptiMistic] Listening... (Ready for your command)\n", flush=True)
            self.display.show("Listening...", "Ready for your command")
            return None
        if not should_process:
            print(
                f"[OptiMistic] Ignored: \"{transcript}\" (Sentence must start with \"OptiMistic\", e.g. \"OptiMistic open google\")",
                flush=True,
            )
            self.display.show("Waiting...", "Say 'OptiMistic <command>'")
            return None

        self.display.show("Processing command...", cleaned)
        cmd = self.parser.parse(cleaned)
        print(f"[OptiMistic] Intent: {cmd.intent.name} (Confidence: {cmd.confidence:.2f})", flush=True)

        # Handle 'OptiMistic stop' / cancellation immediately
        if cmd.intent == IntentType.STOP_ASSISTANT:
            self.tts.stop()
            msg = "Stopped."
            print(f"[OptiMistic] {msg}\n", flush=True)
            self.display.show("Stopped.", "Say 'OptiMistic <command>'")
            return ExecutionResult(success=True, message=msg)

        # Handle low confidence / unrecognized commands
        if cmd.confidence < self.config.stt_min_confidence or cmd.intent == IntentType.UNKNOWN:
            prompt = cmd.clarification_prompt or "I didn't catch that command. Could you please repeat?"
            print(f"[OptiMistic] Clarification needed: {prompt}", flush=True)
            self.display.show("Clarification needed", prompt)
            self._speak_and_pause_mic(prompt, block=True)
            self.display.show("Listening...", "Say 'OptiMistic <command>'")
            print("[OptiMistic] Listening... (Say \"OptiMistic <command>\")\n", flush=True)
            return ExecutionResult(success=False, message=prompt)

        # Handle pending confirmation turn
        if self.conversation.has_pending_confirmation():
            outcome, resolved_cmd = self.conversation.resolve_confirmation(cmd)
            if outcome == "cancelled":
                msg = "Action cancelled."
                self.display.show("Cancelled.", msg)
                self._speak_and_pause_mic(msg, block=True)
                self.display.show("Listening...", "Say 'OptiMistic <command>'")
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
            self.display.show("Listening...", "Say 'OptiMistic <command>'")
            return ExecutionResult(success=False, message="Unknown command")

        if cmd.intent in (IntentType.AMBIGUOUS, IntentType.OPEN_WEBSITE_UNKNOWN):
            prompt = cmd.clarification_prompt or self.responses.unknown_command_message()
            self.display.show("Clarification needed", prompt)
            self._speak_and_pause_mic(prompt, block=True)
            self.display.show("Listening...", "Say 'OptiMistic <command>'")
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
        if cmd.intent == IntentType.LLM_QUERY:
            query = cmd.params.get("query", cmd.raw_text)
            print(f"[OptiMistic] Thinking...", flush=True)
            self.display.show("Thinking...", query)
            result = self.router.execute(cmd)
            self.conversation.update_context(cmd, result.success)

            if result.success:
                answer = result.message
                print(f"[OptiMistic] {answer}\n", flush=True)
                self.display.show(answer, query)
                self._speak_and_pause_mic(answer, block=True)
            else:
                err_msg = result.message or "I couldn't process that question right now."
                print(f"[OptiMistic] {err_msg}\n", flush=True)
                self.display.show("Notice", err_msg)
                self._speak_and_pause_mic(err_msg, block=True)

            self.display.show("Listening...", "Say 'OptiMistic <command>'")
            print("[OptiMistic] Listening... (Say \"OptiMistic <command>\")\n", flush=True)
            return result

        pre_msg = self.responses.pre_execution_message(cmd)
        print(f"[OptiMistic] Executing: {pre_msg}", flush=True)
        self.display.show(pre_msg, cmd.raw_text)

        result = self.router.execute(cmd)
        self.conversation.update_context(cmd, result.success)

        if result.success:
            self._speak_and_pause_mic(pre_msg, block=True)
            print("[OptiMistic] Done.", flush=True)
            self.display.show("Done.", pre_msg)
        else:
            err_msg = self.responses.failure_message(cmd, result.message)
            print(f"[OptiMistic] Failed: {err_msg}", flush=True)
            self.display.show("Action failed", err_msg)
            self._speak_and_pause_mic(err_msg, block=True)

        self.display.show("Listening...", "Say 'OptiMistic <command>'")
        print("[OptiMistic] Listening... (Say \"OptiMistic <command>\")\n", flush=True)
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
        mode_str = 'Wake prefix active (Say "OptiMistic <command>")' if self.wake_word.enabled else "Continuous listening mode"
        print(f"[OptiMistic] Microphone: {self.mic.device_name} ({self.config.sample_rate} Hz, mono)", flush=True)
        print(f"[OptiMistic] Speaker / Audio Output: {self.tts.active_output_name}", flush=True)
        print(f"[OptiMistic] Speech Recognition: {self.stt.active_engine}", flush=True)
        print(f"[OptiMistic] Wake Mode: {mode_str}", flush=True)
        print(f"[OptiMistic] Listening... (Say \"OptiMistic <command>\", e.g. \"OptiMistic open google\")\n", flush=True)
        self.display.show("Listening...", "Say 'OptiMistic <command>'")

        try:
            last_partial = ""
            last_debug_log = 0.0
            while self._running:
                chunk = self.mic.read_chunk(timeout=0.1)
                if chunk is None:
                    continue

                utterance = self.vad.process_chunk(chunk)
                rms = self.vad.current_rms

                # 1. State change: Speech started
                if self.vad.speech_started:
                    print("\n[OptiMistic] Speech detected...", flush=True)
                    self.display.show("Listening...", "Speech detected")
                    self.stt.start_stream()
                    for pre_chunk in self.vad._utterance_chunks:
                        self.stt.feed_stream_chunk(pre_chunk)

                # 2. While user is speaking: Feed live chunk and show partial transcript
                if self.vad.is_speaking and not self.vad.speech_started:
                    if self.config.live_partial_feedback:
                        partial = self.stt.feed_stream_chunk(chunk)
                        if partial and partial != last_partial:
                            print(f"[OptiMistic] Live: \"{partial}\"", flush=True)
                            self.display.show("Hearing...", partial)
                            last_partial = partial

                # 3. Idle level logging if in debug mode
                if not self.vad.is_speaking and self.debug:
                    now = time.monotonic()
                    if now - last_debug_log >= 2.5:
                        print(f"[DEBUG] Mic idle | rms={rms:.5f} | noise_floor={self.vad.noise_floor:.5f}", flush=True)
                        last_debug_log = now

                # 4. Utterance completed (end of speech after silence timeout)
                if utterance is not None:
                    print("[OptiMistic] Speech ended, finalizing...", flush=True)
                    final_text = self.stt.finish_stream(utterance)
                    last_partial = ""
                    if not final_text:
                        if self.debug:
                            print("[DEBUG] Audio captured but no words detected.", flush=True)
                        continue

                    print(f"\n[OptiMistic] Heard: \"{final_text}\"", flush=True)
                    self.process_utterance(final_text)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def run_text_loop(self) -> None:
        """Interactive text-input loop (useful for testing without a physical mic)."""
        self.startup()
        self._running = True
        self.display.show("Listening...", "Text input mode")
        print("[OptiMistic] Text input mode active. Enter commands starting with 'OptiMistic' (or 'exit' to quit):\n", flush=True)
        try:
            while self._running:
                user_input = input("OptiMistic > ").strip()
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
    parser.add_argument("--wake-word", action="store_true", help="Enable 'OptiMistic' prefix requirement (active by default)")
    parser.add_argument("--continuous", action="store_true", help="Enable continuous listening mode (disable 'OptiMistic' prefix requirement)")
    parser.add_argument("--text-only", action="store_true", help="Run interactive text mode instead of microphone")
    parser.add_argument("--headless", action="store_true", help="Run browser and display in headless mode")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument(
        "--stt",
        choices=["auto", "vosk", "faster_whisper", "mock"],
        default=None,
        help="Speech-to-text engine (faster_whisper is recommended for conversational accuracy)",
    )
    parser.add_argument("--silence-timeout", type=float, default=None, help="Silence timeout in seconds (default: 1.2s)")
    parser.add_argument("--vad-threshold", type=float, default=None, help="VAD energy threshold (default: 0.0030)")
    parser.add_argument("--confidence", type=float, default=None, help="Minimum intent confidence (default: 0.45)")
    parser.add_argument("--no-live", action="store_true", help="Disable live partial transcription feedback")
    parser.add_argument("--web", action="store_true", help="Launch interactive browser Web HUD on http://127.0.0.1:8000")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--debug", action="store_true", help="Print live microphone, VAD, and STT debug logs")
    parser.add_argument("--speaker", type=str, default=None, help="Audio output device for voice speech (e.g. laptop, usb, 0, 1)")
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "grok", "xai", "gemini", "groq", "ollama", "openai", "mock"],
        default=None,
        help="LLM provider for conversational intelligence (default: auto)",
    )
    parser.add_argument("--llm-key", type=str, default=None, help="API key for LLM provider (Grok, Gemini, or Groq)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_devices:
        devices = list_audio_devices()
        print("Input devices (Microphones):")
        for d in devices["inputs"]:
            print(f"  [{d['index']}] {d['name']} (ch={d['max_input_channels']}, sr={d['default_samplerate']})")
        print("\nOutput devices (Speakers / Audio Output):")
        try:
            import win32com.client
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            outputs = voice.GetAudioOutputs()
            for i in range(outputs.Count):
                print(f"  [{i}] {outputs.Item(i).GetDescription()}")
        except Exception:
            pass
        return

    cfg = OptimisticConfig()
    if args.speaker:
        cfg.tts_output_device = args.speaker
    if args.mode:
        cfg.execution_mode = args.mode
    if args.wake_word:
        cfg.wake_word_enabled = True
    if args.continuous:
        cfg.wake_word_enabled = False
    if args.stt:
        cfg.stt_engine = args.stt
    if args.device is not None:
        cfg.input_device = args.device
    if args.silence_timeout is not None:
        cfg.vad_silence_chunks = max(3, int(args.silence_timeout / (cfg.chunk_samples / cfg.sample_rate)))
    if args.vad_threshold is not None:
        cfg.vad_energy_threshold = args.vad_threshold
    if args.confidence is not None:
        cfg.stt_min_confidence = args.confidence
    if args.no_live:
        cfg.live_partial_feedback = False
    if args.headless:
        cfg.browser_headless = True
        cfg.display_mode = "console"
    if args.llm_provider:
        cfg.llm_provider = args.llm_provider
    if args.llm_key:
        if cfg.llm_provider in ("grok", "xai"):
            cfg.xai_api_key = args.llm_key
        elif cfg.llm_provider == "groq":
            cfg.groq_api_key = args.llm_key
        else:
            cfg.gemini_api_key = args.llm_key

    if args.web:
        from optimistic_voice.web_app import run_web_dashboard

        run_web_dashboard(config=cfg, open_browser=True)
        return

    assistant = OptimisticAssistant(config=cfg, debug=args.debug)
    if args.text_only:
        assistant.run_text_loop()
    else:
        assistant.run_voice_loop()


if __name__ == "__main__":
    main()
