"""Real-time microphone capture with mute/pause support."""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional in headless environments
    sd = None  # type: ignore[assignment]


def list_audio_devices() -> dict[str, list[dict[str, Any]]]:
    """Return available input and output audio devices."""
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    if sd is None:
        return {"inputs": inputs, "outputs": outputs}
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            entry = {
                "index": idx,
                "name": dev.get("name", f"Device {idx}"),
                "max_input_channels": dev.get("max_input_channels", 0),
                "max_output_channels": dev.get("max_output_channels", 0),
                "default_samplerate": dev.get("default_samplerate", 16000),
            }
            if entry["max_input_channels"] > 0:
                inputs.append(entry)
            if entry["max_output_channels"] > 0:
                outputs.append(entry)
    except Exception:
        pass
    return {"inputs": inputs, "outputs": outputs}


class MicrophoneStream:
    """Thread-safe real-time microphone audio stream."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_samples: int = 1600,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = chunk_samples
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream: Any = None
        self._running = False
        self._muted = threading.Event()
        self._paused = threading.Event()

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def is_running(self) -> bool:
        return self._running

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        # Drain stale audio accumulated while paused/speaking
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._paused.clear()

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if self._muted.is_set() or self._paused.is_set():
            return
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        try:
            self._queue.put_nowait(mono.astype(np.float32, copy=False))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(mono.astype(np.float32, copy=False))
            except queue.Empty:
                pass

    def start(self) -> None:
        if self._running:
            return
        if sd is None:
            raise RuntimeError("sounddevice is not installed; cannot open microphone.")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    def read_chunk(self, timeout: float = 0.25) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
