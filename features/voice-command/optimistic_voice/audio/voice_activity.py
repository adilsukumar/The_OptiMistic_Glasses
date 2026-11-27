"""Voice Activity Detection (VAD), noise reduction, and AGC for Raspberry Pi & Windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


def high_pass_filter(samples: np.ndarray, alpha: float = 0.90) -> np.ndarray:
    """Lightweight pure-NumPy 1st-order pre-emphasis/high-pass filter to cut low-end rumble."""
    if samples.size < 2:
        return samples.astype(np.float32, copy=False)
    x = samples.astype(np.float32, copy=False)
    x = x - np.mean(x)
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


def remove_dc_and_reduce_noise(
    samples: np.ndarray,
    noise_floor: float = 0.003,
) -> np.ndarray:
    """Apply DC-offset removal, high-pass rumble filtering, and gentle soft noise gating."""
    if samples.size == 0:
        return samples
    audio = samples.astype(np.float32, copy=False)
    audio = audio - np.mean(audio)
    audio = high_pass_filter(audio, alpha=0.95)
    if noise_floor > 0.0:
        gate = noise_floor * 1.15
        mask = np.abs(audio) < gate
        if np.any(mask):
            audio = audio.copy()
            audio[mask] *= 0.35
    return np.clip(audio, -1.0, 1.0)


def normalize_utterance(
    samples: np.ndarray,
    target_peak: float = 0.75,
    max_gain: float = 15.0,
) -> np.ndarray:
    """Apply Automatic Gain Control (AGC) so quiet mics reach optimal STT levels."""
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak < 1e-4:
        return samples
    gain = min(max_gain, target_peak / peak)
    if gain > 1.05:
        return np.clip(samples * gain, -1.0, 1.0)
    return np.clip(samples, -1.0, 1.0)


@dataclass
class VoiceActivityDetector:
    """Adaptive energy-based VAD tuned for sensitive pickup in noisy environments."""

    sample_rate: int = 16000
    energy_threshold: float = 0.004
    speech_pad_chunks: int = 4
    silence_chunks: int = 7
    max_utterance_s: float = 15.0
    chunk_samples: int = 1600
    noise_reduction_enabled: bool = True

    _noise_floor: float = field(default=0.002, init=False)
    _in_speech: bool = field(default=False, init=False)
    _silence_count: int = field(default=0, init=False)
    _pre_buffer: deque[np.ndarray] = field(init=False)
    _utterance_chunks: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._pre_buffer = deque(maxlen=max(1, self.speech_pad_chunks))
        self._noise_floor = max(0.001, self.energy_threshold * 0.5)

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

        Returns a normalized int16 utterance array when speech ends, else None.
        """
        if chunk.size == 0:
            return None

        raw = chunk.astype(np.float32, copy=False)
        raw = raw - np.mean(raw)
        # Measure speech-band energy after removing <100 Hz fan/hum rumble
        filtered = high_pass_filter(raw, alpha=0.96) if self.noise_reduction_enabled else raw
        rms = float(np.sqrt(np.mean(filtered * filtered)))

        # Sensitive SNR threshold: 1.55x stationary noise floor
        dynamic_threshold = max(self.energy_threshold, self._noise_floor * 1.55)

        if not self._in_speech:
            # Asymmetric noise-floor tracking: drops quickly in quiet, rises slowly in noise
            if rms < self._noise_floor:
                self._noise_floor = 0.85 * self._noise_floor + 0.15 * rms
            else:
                self._noise_floor = 0.98 * self._noise_floor + 0.02 * min(rms, dynamic_threshold)

            self._pre_buffer.append(filtered)
            if rms >= dynamic_threshold:
                self._in_speech = True
                self._silence_count = 0
                self._utterance_chunks = list(self._pre_buffer)
                self._pre_buffer.clear()
            return None

        self._utterance_chunks.append(filtered)
        if rms >= dynamic_threshold * 0.65:
            self._silence_count = 0
        else:
            self._silence_count += 1

        max_chunks = max(1, int((self.max_utterance_s * self.sample_rate) / max(1, self.chunk_samples)))
        if self._silence_count >= self.silence_chunks or len(self._utterance_chunks) >= max_chunks:
            full_audio = np.concatenate(self._utterance_chunks)
            if self.noise_reduction_enabled:
                full_audio = remove_dc_and_reduce_noise(full_audio, self._noise_floor * 0.8)
            full_audio = normalize_utterance(full_audio)
            self.reset()
            return (np.clip(full_audio, -1.0, 1.0) * 32767.0).astype(np.int16)

        return None
