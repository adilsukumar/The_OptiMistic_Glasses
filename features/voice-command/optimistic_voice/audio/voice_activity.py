"""Voice Activity Detection (VAD) and lightweight noise reduction for Raspberry Pi."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


def remove_dc_and_reduce_noise(
    samples: np.ndarray,
    noise_floor: float = 0.005,
) -> np.ndarray:
    """Apply DC-offset removal and soft noise gating to float32 mono audio [-1, 1]."""
    if samples.size == 0:
        return samples
    audio = samples.astype(np.float32, copy=False)
    audio = audio - np.mean(audio)
    if noise_floor > 0.0:
        gate = noise_floor * 1.35
        mask = np.abs(audio) < gate
        if np.any(mask):
            audio = audio.copy()
            audio[mask] *= 0.2
    return np.clip(audio, -1.0, 1.0)


@dataclass
class VoiceActivityDetector:
    """Adaptive energy-based VAD that segments speech utterances cleanly."""

    sample_rate: int = 16000
    energy_threshold: float = 0.012
    speech_pad_chunks: int = 2
    silence_chunks: int = 8
    max_utterance_s: float = 15.0
    chunk_samples: int = 1600
    noise_reduction_enabled: bool = True

    _noise_floor: float = field(default=0.005, init=False)
    _in_speech: bool = field(default=False, init=False)
    _silence_count: int = field(default=0, init=False)
    _pre_buffer: deque[np.ndarray] = field(init=False)
    _utterance_chunks: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._pre_buffer = deque(maxlen=max(1, self.speech_pad_chunks))
        self._noise_floor = max(0.002, self.energy_threshold * 0.4)

    @property
    def is_speaking(self) -> bool:
        return self._in_speech

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def reset(self) -> None:
        self._in_speech = False
        self._silence_count = 0
        self._pre_buffer.clear()
        self._utterance_chunks.clear()

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """Process a float32 mono audio chunk.

        Returns a complete int16 utterance array when speech ends, else None.
        """
        if chunk.size == 0:
            return None

        cleaned = (
            remove_dc_and_reduce_noise(chunk, self._noise_floor)
            if self.noise_reduction_enabled
            else chunk.astype(np.float32, copy=False)
        )
        rms = float(np.sqrt(np.mean(cleaned * cleaned)))
        dynamic_threshold = max(self.energy_threshold, self._noise_floor * 2.4)

        if not self._in_speech:
            # Slowly adapt noise floor when idle
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * min(rms, dynamic_threshold)
            self._pre_buffer.append(cleaned)
            if rms >= dynamic_threshold:
                self._in_speech = True
                self._silence_count = 0
                self._utterance_chunks = list(self._pre_buffer)
                self._pre_buffer.clear()
            return None

        self._utterance_chunks.append(cleaned)
        if rms >= dynamic_threshold * 0.75:
            self._silence_count = 0
        else:
            self._silence_count += 1

        max_chunks = max(1, int((self.max_utterance_s * self.sample_rate) / max(1, self.chunk_samples)))
        if self._silence_count >= self.silence_chunks or len(self._utterance_chunks) >= max_chunks:
            full_audio = np.concatenate(self._utterance_chunks)
            self.reset()
            return (np.clip(full_audio, -1.0, 1.0) * 32767.0).astype(np.int16)

        return None
