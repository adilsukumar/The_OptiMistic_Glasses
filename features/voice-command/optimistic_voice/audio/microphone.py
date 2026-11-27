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
        self.device_name = "Default Microphone"
        self.native_samplerate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream: Any = None
        self._running = False
        self._muted = threading.Event()
        self._paused = threading.Event()
        self._last_rms = 0.0

        if sd is not None:
            try:
                # 1. If explicit string requested (e.g. "bluetooth", "headset", or device name substring)
                if self.device is not None and isinstance(self.device, str) and not self.device.isdigit():
                    dev_str = self.device.lower()
                    for idx, dev in enumerate(sd.query_devices()):
                        if dev.get("max_input_channels", 0) > 0:
                            dname = dev.get("name", "").lower()
                            if dev_str in dname or (
                                dev_str in ("bluetooth", "bt", "headset", "earbuds")
                                and any(k in dname for k in ("headset", "bluetooth", "cosmobuds", "wh-", "buds", "airpods"))
                            ):
                                self.device = idx
                                break

                # 2. If no specific device requested or auto:
                # Auto-detect Bluetooth headset microphone first
                if self.device is None or str(self.device).lower() == "auto":
                    devices = sd.query_devices()
                    for idx, dev in enumerate(devices):
                        if dev.get("max_input_channels", 0) > 0:
                            dname = dev.get("name", "").lower()
                            if any(k in dname for k in ("cosmobuds", "headset", "wh-", "buds", "airpods", "bluetooth")) and "mapper" not in dname:
                                self.device = idx
                                break

                    # If no Bluetooth, auto-detect USB microphone
                    if self.device is None or str(self.device).lower() == "auto":
                        for idx, dev in enumerate(devices):
                            if dev.get("max_input_channels", 0) > 0:
                                dname = dev.get("name", "").lower()
                                if ("usb" in dname or "ab13x" in dname) and "mapper" not in dname:
                                    self.device = idx
                                    break

                target_dev = int(self.device) if (self.device is not None and str(self.device).isdigit()) else (
                    self.device if (self.device is not None and str(self.device).lower() != "auto") else sd.default.device[0]
                )
                dev_info = sd.query_devices(target_dev)
                self.device_name = dev_info.get("name", f"Device {target_dev}")
                self.native_samplerate = int(dev_info.get("default_samplerate", sample_rate))
                self.device = target_dev
            except Exception:
                pass

    @property
    def is_muted(self) -> bool:
        return self._muted.is_set()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_rms(self) -> float:
        return self._last_rms

    def mute(self) -> None:
        self._muted.set()

    def unmute(self) -> None:
        self._muted.clear()

    def pause(self) -> None:
        self._paused.set()
        if self._stream is not None:
            try:
                if getattr(self._stream, "active", False):
                    self._stream.stop()
            except Exception:
                pass

    def resume(self, settle_delay_s: float = 0.20) -> None:
        """Resume recording and drain stale audio/speaker echo."""
        import time
        if self._stream is not None:
            try:
                if getattr(self._stream, "stopped", False):
                    self._stream.start()
            except Exception:
                pass
        if settle_delay_s > 0:
            time.sleep(settle_delay_s)
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
        chunk_f32 = mono.astype(np.float32, copy=False)
        self._last_rms = float(np.sqrt(np.mean(chunk_f32 * chunk_f32))) if chunk_f32.size > 0 else 0.0
        try:
            self._queue.put_nowait(chunk_f32)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk_f32)
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
