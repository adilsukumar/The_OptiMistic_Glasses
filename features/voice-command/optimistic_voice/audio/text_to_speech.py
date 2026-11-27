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
        on_speak_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.requested_engine = engine.lower()
        self.piper_model_path = Path(piper_model_path) if piper_model_path else None
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.on_speak_callback = on_speak_callback

        self.active_engine: str = "mock"
        self.spoken_history: list[str] = []
        self._queue: queue.Queue[tuple[str, threading.Event | None]] = queue.Queue()
        self._stop_event = threading.Event()
        self._speaking_event = threading.Event()

        self._detect_engine()
        self._worker = threading.Thread(target=self._worker_loop, name="OptimisticTTS", daemon=True)
        self._worker.start()

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

    def _worker_loop(self) -> None:
        pyttsx3_engine: Any = None
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
                    pyttsx3_engine = self._speak_pyttsx3(text, pyttsx3_engine)
                elif self.active_engine == "espeak":
                    self._speak_espeak(text)
                else:
                    # Mock/headless mode prints nothing extra unless callback handles it
                    pass
            except Exception:
                # Fallback to espeak or silent record without crashing the loop
                pass
            finally:
                self._speaking_event.clear()
                if done_event is not None:
                    done_event.set()
                self._queue.task_done()

    def _speak_pyttsx3(self, text: str, engine: Any) -> Any:
        import pyttsx3

        if engine is None:
            engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        engine.say(text)
        engine.runAndWait()
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
