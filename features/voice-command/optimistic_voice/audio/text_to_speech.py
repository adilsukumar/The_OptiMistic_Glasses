"""Offline Text-to-Speech engine with sequential speech queue (no overlapping audio)."""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import wave
import io
from pathlib import Path
from typing import Any, Callable


class TextToSpeechEngine:
    """Thread-safe TTS engine supporting Piper, pyttsx3, eSpeak NG, and mock output."""

    def __init__(
        self,
        engine: str = "auto",
        piper_model_path: Path | str | None = None,
        rate: int = 175,
        volume: float = 0.95,
        output_device: str | int | None = None,
        on_speak_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.requested_engine = engine.lower()
        self.piper_model_path = Path(piper_model_path) if piper_model_path else None
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.output_device = output_device
        self.active_output_name: str = "System Default"
        self.on_speak_callback = on_speak_callback

        self.active_engine: str = "mock"
        self.spoken_history: list[str] = []
        self._queue: queue.Queue[tuple[str, threading.Event | None]] = queue.Queue()
        self._stop_event = threading.Event()
        self._speaking_event = threading.Event()
        self._cancel_speech = threading.Event()
        self._current_sapi_voice: Any = None

        self._detect_engine()
        self._probe_initial_output()
        self._worker = threading.Thread(target=self._worker_loop, name="OptimisticTTS", daemon=True)
        self._worker.start()

    def _probe_initial_output(self) -> None:
        """Probe and set active audio output during initialization."""
        if self.active_engine == "mock":
            return
        try:
            import win32com.client
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._apply_sapi_output(voice)
        except Exception:
            pass

    def _detect_engine(self) -> None:
        if self.requested_engine == "mock":
            self.active_engine = "mock"
            return

        if self.requested_engine in ("auto", "piper"):
            if self._piper_available():
                self.active_engine = "piper"
                return
            if self.requested_engine == "piper":
                raise RuntimeError("Piper TTS requested but piper binary/package or model not found.")

        if self.requested_engine in ("auto", "pyttsx3"):
            try:
                import pyttsx3  # noqa: F401

                self.active_engine = "pyttsx3"
                return
            except Exception:
                if self.requested_engine == "pyttsx3":
                    raise RuntimeError("pyttsx3 requested but not available.")

        if self.requested_engine in ("auto", "espeak", "espeak-ng"):
            if shutil.which("espeak-ng") or shutil.which("espeak"):
                self.active_engine = "espeak"
                return

        self.active_engine = "mock"

    def _piper_available(self) -> bool:
        if not (self.piper_model_path and self.piper_model_path.exists()):
            return False
        if shutil.which("piper"):
            return True
        try:
            import piper  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def is_speaking(self) -> bool:
        return self._speaking_event.is_set()

    def speak(self, text: str, block: bool = False) -> None:
        """Enqueue text for sequential speech output."""
        clean_text = text.strip()
        if not clean_text:
            return
        done_event = threading.Event() if block else None
        self._queue.put((clean_text, done_event))
        if done_event is not None:
            done_event.wait()

    def wait_until_done(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        """Immediately interrupt any active speech and purge remaining queue items."""
        self._cancel_speech.set()

        # Discard any remaining queued utterances
        while not self._queue.empty():
            try:
                _, done_event = self._queue.get_nowait()
                if done_event is not None:
                    done_event.set()
                self._queue.task_done()
            except queue.Empty:
                break

        # Cancel active Windows SAPI SpVoice immediately
        if self._current_sapi_voice is not None:
            try:
                # SVSFPurgeBeforeSpeak = 2 clears active speech buffer
                self._current_sapi_voice.Speak("", 2)
            except Exception:
                pass

        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        self._speaking_event.clear()

    def _worker_loop(self) -> None:
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        pyttsx3_engine: Any = None
        sapi_voice: Any = None
        while not self._stop_event.is_set():
            try:
                text, done_event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._speaking_event.set()
            try:
                self.spoken_history.append(text)
                if self.on_speak_callback:
                    self.on_speak_callback(text)

                if self.active_engine == "piper":
                    self._speak_piper(text)
                elif self.active_engine == "pyttsx3":
                    sapi_voice, pyttsx3_engine = self._speak_windows(text, sapi_voice, pyttsx3_engine)
                elif self.active_engine == "espeak":
                    self._speak_espeak(text)
                else:
                    # Mock/headless mode prints nothing extra unless callback handles it
                    pass
            except Exception as e:
                # Log exception so it's not silently lost
                import traceback
                print(f"[TTS Error] {e}", flush=True)
            finally:
                self._speaking_event.clear()
                if done_event is not None:
                    done_event.set()
                self._queue.task_done()

        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def _apply_sapi_output(self, engine: Any) -> None:
        """Route Windows SAPI5 audio to the correct speaker (prefer Bluetooth earbuds when connected)."""
        try:
            tts = getattr(getattr(getattr(engine, "proxy", None), "_driver", None), "_tts", None)
            if tts is None and hasattr(engine, "GetAudioOutputs"):
                tts = engine
            if tts is None or not hasattr(tts, "GetAudioOutputs"):
                return

            outputs = tts.GetAudioOutputs()
            if not outputs or outputs.Count == 0:
                return

            target = self.output_device
            target_str = str(target).lower().strip() if target is not None else ""

            # 1. Explicit target matching
            if target_str and target_str not in ("auto", "default"):
                for i in range(outputs.Count):
                    desc = outputs.Item(i).GetDescription().lower()
                    if target_str in ("bluetooth", "bt", "earbuds", "headphones", "headset"):
                        if any(k in desc for k in ("headphone", "headset", "bluetooth", "cosmobuds", "wh-", "buds", "airpods")):
                            tts.AudioOutput = outputs.Item(i)
                            self.active_output_name = outputs.Item(i).GetDescription()
                            return
                    elif target_str in ("laptop", "internal", "speakers", "speaker"):
                        if any(k in desc for k in ("senary", "realtek", "high definition", "speakers")):
                            tts.AudioOutput = outputs.Item(i)
                            self.active_output_name = outputs.Item(i).GetDescription()
                            return
                    elif target_str == str(i) or target_str in desc:
                        tts.AudioOutput = outputs.Item(i)
                        self.active_output_name = outputs.Item(i).GetDescription()
                        return

            # 2. Smart auto-preference:
            # If Bluetooth headphones or earbuds are connected, ALWAYS prefer them so speech is heard in-ear!
            for i in range(outputs.Count):
                desc = outputs.Item(i).GetDescription().lower()
                if any(k in desc for k in ("cosmobuds", "headphone", "headset", "bluetooth", "wh-", "buds", "airpods")):
                    tts.AudioOutput = outputs.Item(i)
                    self.active_output_name = outputs.Item(i).GetDescription()
                    return

            # 3. If no Bluetooth, use laptop speakers (Senary, Realtek, etc.)
            for i in range(outputs.Count):
                desc = outputs.Item(i).GetDescription().lower()
                if any(k in desc for k in ("senary", "realtek", "high definition", "internal", "laptop")):
                    tts.AudioOutput = outputs.Item(i)
                    self.active_output_name = outputs.Item(i).GetDescription()
                    return

            if hasattr(tts, "AudioOutput") and tts.AudioOutput:
                self.active_output_name = tts.AudioOutput.GetDescription()
        except Exception:
            pass

    def _speak_windows(self, text: str, sapi_voice: Any, pyttsx3_engine: Any) -> tuple[Any, Any]:
        """Speak on Windows using fast native SAPI.SpVoice with fallback to pyttsx3."""
        # 1. Primary: Direct SAPI SpVoice for fast, glitch-free output and direct device routing
        self._cancel_speech.clear()
        try:
            if sapi_voice is None:
                import win32com.client
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._current_sapi_voice = sapi_voice
            self._apply_sapi_output(sapi_voice)
            sapi_voice.Volume = int(self.volume * 100)
            sapi_rate = int((self.rate - 175) / 25)
            sapi_voice.Rate = max(-10, min(10, sapi_rate))

            # SVSFlagsAsync = 1 allows responsive cancellation in tight 30ms polling
            sapi_voice.Speak(text, 1)
            while not sapi_voice.WaitUntilDone(30):
                if self._cancel_speech.is_set() or self._stop_event.is_set():
                    sapi_voice.Speak("", 2)  # SVSFPurgeBeforeSpeak
                    break
            return sapi_voice, pyttsx3_engine
        except Exception:
            pass
        finally:
            self._current_sapi_voice = None

        # 2. Fallback: pyttsx3
        try:
            if pyttsx3_engine is None:
                import pyttsx3
                pyttsx3_engine = pyttsx3.init()
            self._apply_sapi_output(pyttsx3_engine)
            pyttsx3_engine.setProperty("rate", self.rate)
            pyttsx3_engine.setProperty("volume", self.volume)
            pyttsx3_engine.say(text)
            pyttsx3_engine.runAndWait()
        except Exception:
            pass
        return sapi_voice, pyttsx3_engine

    def _speak_pyttsx3(self, text: str, engine: Any) -> Any:
        _, engine = self._speak_windows(text, None, engine)
        return engine

    def _speak_espeak(self, text: str) -> None:
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if binary:
            subprocess.run([binary, "-s", str(self.rate), text], check=False)

    def _speak_piper(self, text: str) -> None:
        piper_bin = shutil.which("piper")
        if piper_bin and self.piper_model_path:
            proc = subprocess.run(
                [piper_bin, "--model", str(self.piper_model_path), "--output_raw"],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.stdout:
                self._play_raw_pcm(proc.stdout, sample_rate=22050)
                return

        # Python piper-tts fallback
        from piper.voice import PiperVoice
        import numpy as np

        voice = PiperVoice.load(str(self.piper_model_path))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        buf.seek(0)
        with wave.open(buf, "rb") as wav_read:
            sr = wav_read.getframerate()
            raw = wav_read.readframes(wav_read.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            import sounddevice as sd

            sd.play(audio * self.volume, samplerate=sr, blocking=True)

    def _play_raw_pcm(self, raw_bytes: bytes, sample_rate: int = 22050) -> None:
        import numpy as np
        import sounddevice as sd

        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio * self.volume, samplerate=sample_rate, blocking=True)

    def set_rate(self, rate: int) -> None:
        """Configure speech rate (words per minute)."""
        self.rate = max(80, min(400, int(rate)))

    def set_volume(self, volume: float) -> None:
        """Configure output volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, float(volume)))

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
