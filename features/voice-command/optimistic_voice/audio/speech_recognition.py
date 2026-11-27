"""Offline-capable Speech-to-Text engine supporting Vosk and faster-whisper."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


class DuplicateCommandFilter:
    """Prevents the same recognized command from firing repeatedly within a cooldown."""

    def __init__(self, cooldown_s: float = 2.0) -> None:
        self.cooldown_s = cooldown_s
        self._last_text: str = ""
        self._last_time: float = 0.0

    def is_duplicate(self, text: str, now: float | None = None) -> bool:
        normalized = " ".join(text.lower().strip().split())
        if not normalized:
            return True
        current = time.monotonic() if now is None else now
        if normalized == self._last_text and (current - self._last_time) < self.cooldown_s:
            return True
        self._last_text = normalized
        self._last_time = current
        return False


class SpeechRecognizer:
    """Unified offline STT wrapper for Vosk (Pi-friendly) and faster-whisper."""

    def __init__(
        self,
        engine: str = "auto",
        sample_rate: int = 16000,
        vosk_model_path: Path | str | None = None,
        whisper_model_size: str = "tiny.en",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        duplicate_cooldown_s: float = 2.0,
    ) -> None:
        self.requested_engine = engine.lower()
        self.sample_rate = sample_rate
        self.vosk_model_path = Path(vosk_model_path) if vosk_model_path else None
        self.whisper_model_size = whisper_model_size
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.dedup = DuplicateCommandFilter(cooldown_s=duplicate_cooldown_s)

        self.active_engine: str = "uninitialized"
        self._vosk_model: Any = None
        self._whisper_model: Any = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        if self.requested_engine == "mock":
            self.active_engine = "mock"
            return

        if self.requested_engine in ("auto", "vosk"):
            if self._try_init_vosk():
                self.active_engine = "vosk"
                return
            if self.requested_engine == "vosk":
                raise RuntimeError(
                    f"Vosk engine requested, but model or vosk package unavailable at {self.vosk_model_path}"
                )

        if self.requested_engine in ("auto", "faster_whisper", "faster-whisper"):
            if self._try_init_whisper():
                self.active_engine = "faster_whisper"
                return
            if self.requested_engine in ("faster_whisper", "faster-whisper"):
                raise RuntimeError("faster-whisper engine requested, but package/model unavailable.")

        self.active_engine = "unavailable"

    def _try_init_vosk(self) -> bool:
        try:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            if self.vosk_model_path and self.vosk_model_path.exists():
                self._vosk_model = Model(str(self.vosk_model_path))
                return True
        except Exception:
            pass
        return False

    def _try_init_whisper(self) -> bool:
        try:
            from faster_whisper import WhisperModel

            self._whisper_model = WhisperModel(
                self.whisper_model_size,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
            return True
        except Exception:
            return False

    def transcribe(self, audio_int16: np.ndarray, filter_duplicates: bool = True) -> str:
        """Transcribe an int16 mono PCM utterance into text."""
        if audio_int16.size == 0:
            return ""

        text = ""
        if self.active_engine == "vosk" and self._vosk_model is not None:
            from vosk import KaldiRecognizer

            rec = KaldiRecognizer(self._vosk_model, self.sample_rate)
            rec.AcceptWaveform(audio_int16.tobytes())
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
        elif self.active_engine == "faster_whisper" and self._whisper_model is not None:
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            segments, _ = self._whisper_model.transcribe(
                audio_f32,
                language="en",
                vad_filter=False,
                beam_size=1,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        elif self.active_engine == "mock":
            return ""
        else:
            raise RuntimeError(
                "No offline speech recognition model is available. "
                "Download a Vosk model into models/speech_models/ or install faster-whisper."
            )

        if filter_duplicates and self.dedup.is_duplicate(text):
            return ""
        return text
