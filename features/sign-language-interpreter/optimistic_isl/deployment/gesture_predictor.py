"""Standalone TFLite gesture predictor for Raspberry Pi deployment."""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np


def _load_interpreter(model_path: Path, num_threads: int = 4):
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(model_path), num_threads=num_threads)
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(model_path), num_threads=num_threads)
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=str(model_path), num_threads=num_threads)


def _interpolate_short_dropouts(sequence: np.ndarray, max_gap: int = 4) -> np.ndarray:
    out = sequence.copy()
    for flag_idx, start_col, end_col in ((126, 0, 63), (127, 63, 126)):
        valid = np.where(out[:, flag_idx] > 0.5)[0]
        for k in range(len(valid) - 1):
            i0, i1 = int(valid[k]), int(valid[k + 1])
            gap = i1 - i0 - 1
            if 0 < gap <= max_gap:
                v0, v1 = out[i0, start_col:end_col], out[i1, start_col:end_col]
                for step in range(1, gap + 1):
                    a = step / (gap + 1)
                    out[i0 + step, start_col:end_col] = (1.0 - a) * v0 + a * v1
                    out[i0 + step, flag_idx] = 1.0
    return out


class PiGesturePredictor:
    """Lightweight TFLite predictor with temporal smoothing for Raspberry Pi."""

    def __init__(
        self,
        model_path: Path | None = None,
        labels_path: Path | None = None,
        sequence_length: int = 30,
        confidence_threshold: float = 0.70,
        smoothing_window: int = 7,
        min_consistent_frames: int = 4,
        ema_alpha: float = 0.45,
        num_threads: int = 4,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        model_path = model_path or (base_dir / "optimistic_model.tflite")
        labels_path = labels_path or (base_dir / "labels.json")

        with open(labels_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

        num_classes = int(label_map["num_classes"])
        self.class_names: List[str] = [
            label_map["id_to_display"][str(i)] for i in range(num_classes)
        ]
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.min_consistent_frames = min_consistent_frames
        self.ema_alpha = ema_alpha
        self.buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self._votes: Deque[Optional[int]] = deque(maxlen=smoothing_window)
        self._ema_probs: Optional[np.ndarray] = None

        self._interpreter = _load_interpreter(model_path, num_threads=num_threads)
        self._interpreter.allocate_tensors()
        self._in = self._interpreter.get_input_details()[0]
        self._out = self._interpreter.get_output_details()[0]

    def push_frame(self, feature_vec: np.ndarray) -> Dict[str, Any]:
        self.buffer.append(np.asarray(feature_vec, dtype=np.float32))
        ts = datetime.now(timezone.utc).isoformat()
        if len(self.buffer) < self.sequence_length:
            return {
                "gesture": "Gesture not recognized",
                "confidence": 0.0,
                "recognized": False,
                "timestamp": ts,
            }

        seq = np.stack(list(self.buffer), axis=0)
        if int(np.sum((seq[:, 126] > 0.5) | (seq[:, 127] > 0.5))) < 4:
            self._ema_probs = None
            self._votes.clear()
            return {
                "gesture": "Gesture not recognized",
                "confidence": 0.0,
                "recognized": False,
                "timestamp": ts,
            }

        seq = _interpolate_short_dropouts(seq)
        sample = np.expand_dims(seq, axis=0).astype(np.float32)
        in_scale, in_zero = self._in.get("quantization", (0.0, 0))
        if self._in["dtype"] == np.int8 and in_scale > 0:
            sample = np.round(sample / in_scale + in_zero).clip(-128, 127).astype(np.int8)

        t0 = time.perf_counter()
        self._interpreter.set_tensor(self._in["index"], sample)
        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(self._out["index"])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        out_scale, out_zero = self._out.get("quantization", (0.0, 0))
        if self._out["dtype"] == np.int8 and out_scale > 0:
            probs = (probs.astype(np.float32) - out_zero) * out_scale
        probs = probs.astype(np.float32)

        self._ema_probs = (
            probs.copy()
            if self._ema_probs is None
            else (self.ema_alpha * probs + (1.0 - self.ema_alpha) * self._ema_probs)
        )
        top_id = int(np.argmax(self._ema_probs))
        top_conf = float(self._ema_probs[top_id])
        self._votes.append(top_id if top_conf >= self.confidence_threshold else None)

        valid = [v for v in self._votes if v is not None]
        recognized = False
        gesture = "Gesture not recognized"
        if valid:
            winner_id, count = Counter(valid).most_common(1)[0]
            if count >= self.min_consistent_frames:
                winner_conf = float(self._ema_probs[winner_id])
                if winner_conf >= self.confidence_threshold:
                    recognized = True
                    gesture = self.class_names[winner_id]
                    top_conf = winner_conf

        return {
            "gesture": gesture,
            "confidence": round(top_conf, 4),
            "recognized": recognized,
            "timestamp": ts,
            "latency_ms": round(latency_ms, 2),
        }
