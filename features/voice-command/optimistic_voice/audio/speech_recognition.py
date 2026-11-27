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
        self._stream_rec: Any = None
        self._stream_words: list[str] = []
        self._stream_buffer: list[np.ndarray] = []
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        if self.requested_engine == "mock":
            self.active_engine = "mock"
            return

        if self.requested_engine == "vosk":
            if self._try_init_vosk():
                self.active_engine = "vosk"
                return
            raise RuntimeError(
                f"Vosk engine requested, but model or vosk package unavailable at {self.vosk_model_path}"
            )

        if self.requested_engine in ("faster_whisper", "faster-whisper"):
            if self._try_init_whisper():
                self.active_engine = "faster_whisper"
                return
            raise RuntimeError("faster-whisper engine requested, but package/model unavailable.")

        # In "auto" mode: prefer faster_whisper for superior recognition accuracy if available,
        # otherwise fall back to offline Vosk.
        if self.requested_engine == "auto":
            if self._try_init_whisper():
                self.active_engine = "faster_whisper"
                return
            if self._try_init_vosk():
                self.active_engine = "vosk"
                return

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
            chunk_size = 4000
            audio_bytes = audio_int16.tobytes()
            words: list[str] = []
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                if rec.AcceptWaveform(chunk):
                    part = json.loads(rec.Result()).get("text", "").strip()
                    if part:
                        words.append(part)
            final_part = json.loads(rec.FinalResult()).get("text", "").strip()
            if final_part:
                words.append(final_part)
            text = " ".join(words).strip()
        elif self.active_engine == "faster_whisper" and self._whisper_model is not None:
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            segments, _ = self._whisper_model.transcribe(
                audio_f32,
                language="en",
                initial_prompt="OptiMistic. OptiMistic, what is the weather? OptiMistic open Google. OptiMistic search YouTube. OptiMistic stop.",
                vad_filter=False,
                beam_size=2,
                condition_on_previous_text=False,
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

    def start_stream(self) -> None:
        """Start a live streaming recognition session."""
        self._stream_words = []
        self._stream_buffer = []
        if self.active_engine == "vosk" and self._vosk_model is not None:
            from vosk import KaldiRecognizer

            self._stream_rec = KaldiRecognizer(self._vosk_model, self.sample_rate)
        else:
            self._stream_rec = None

    def feed_stream_chunk(self, chunk: np.ndarray) -> str:
        """Feed a float32 or int16 audio chunk, returning live partial text if available."""
        if chunk.size == 0:
            return ""

        if chunk.dtype != np.int16:
            c_int16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            c_int16 = chunk

        self._stream_buffer.append(c_int16)

        if self.active_engine == "vosk" and self._stream_rec is not None:
            chunk_bytes = c_int16.tobytes()
            if self._stream_rec.AcceptWaveform(chunk_bytes):
                part = json.loads(self._stream_rec.Result()).get("text", "").strip()
                if part:
                    self._stream_words.append(part)
                return " ".join(self._stream_words).strip()
            else:
                partial = json.loads(self._stream_rec.PartialResult()).get("partial", "").strip()
                combined = (" ".join(self._stream_words) + " " + partial).strip()
                return combined

        return ""

    def finish_stream(
        self,
        final_audio_int16: np.ndarray | None = None,
        filter_duplicates: bool = True,
    ) -> str:
        """Finalize the active stream and return the complete transcribed text."""
        text = ""
        if self.active_engine == "vosk" and self._stream_rec is not None:
            final_part = json.loads(self._stream_rec.FinalResult()).get("text", "").strip()
            if final_part:
                self._stream_words.append(final_part)
            text = " ".join(self._stream_words).strip()
            self._stream_rec = None
            self._stream_words = []
            self._stream_buffer = []
        elif self.active_engine == "faster_whisper" and self._whisper_model is not None:
            if final_audio_int16 is not None and final_audio_int16.size > 0:
                audio_to_use = final_audio_int16
            elif self._stream_buffer:
                audio_to_use = np.concatenate(self._stream_buffer)
            else:
                audio_to_use = np.array([], dtype=np.int16)
            self._stream_buffer = []
            if audio_to_use.size > 0:
                text = self.transcribe(audio_to_use, filter_duplicates=False)
        else:
            if final_audio_int16 is not None:
                text = self.transcribe(final_audio_int16, filter_duplicates=False)

        if filter_duplicates and self.dedup.is_duplicate(text):
            return ""
        return text
