"""TFLite / Keras gesture predictor with rolling temporal landmark buffer."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from dataset.extract_landmarks import interpolate_short_dropouts
from inference.prediction_smoother import PredictionSmoother
import config

logger = logging.getLogger("gesture_predictor")


def _load_tflite_interpreter(tflite_path: Path):
    """Load TFLite Interpreter using lightweight runtime first, falling back to TF."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(tflite_path))
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(tflite_path))
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=str(tflite_path))


class GesturePredictor:
    """Maintains a temporal landmark buffer and runs TFLite gesture classification."""

    def __init__(
        self,
        model_path: Path = config.TFLITE_MODEL_PATH,
        labels_path: Path = config.DEPLOYMENT_LABELS_PATH,
        sequence_length: int = config.DATASET_CONFIG.sequence_length,
        confidence_threshold: float = config.INFERENCE_CONFIG.confidence_threshold,
        smoothing_window: int = config.INFERENCE_CONFIG.smoothing_window,
        min_consistent_frames: int = config.INFERENCE_CONFIG.min_consistent_frames,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels JSON not found: {labels_path}")

        with open(labels_path, "r", encoding="utf-8") as f:
            self.label_mapping = json.load(f)

        num_classes = int(self.label_mapping["num_classes"])
        self.class_names: List[str] = [
            self.label_mapping["id_to_display"][str(i)] for i in range(num_classes)
        ]

        self.sequence_length = sequence_length
        self.buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self.smoother = PredictionSmoother(
            class_names=self.class_names,
            confidence_threshold=confidence_threshold,
            window_size=smoothing_window,
            min_consistent_frames=min_consistent_frames,
            ema_alpha=config.INFERENCE_CONFIG.ema_alpha,
            unrecognized_label=config.INFERENCE_CONFIG.unrecognized_label,
        )

        self._interpreter = _load_tflite_interpreter(model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]
        logger.info("Loaded TFLite model from %s (%d classes)", model_path, num_classes)

    def reset(self) -> None:
        self.buffer.clear()
        self.smoother.reset()

    def predict_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Run raw TFLite inference on a (T, D) sequence and return class probabilities."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.shape != (self.sequence_length, config.DATASET_CONFIG.feature_dim):
            raise ValueError(
                f"Expected sequence shape ({self.sequence_length}, "
                f"{config.DATASET_CONFIG.feature_dim}), got {seq.shape}"
            )

        sample = np.expand_dims(seq, axis=0)
        in_dtype = self._input_details["dtype"]
        in_scale, in_zero = self._input_details.get("quantization", (0.0, 0))
        if in_dtype == np.int8 and in_scale > 0:
            sample = np.round(sample / in_scale + in_zero).clip(-128, 127).astype(np.int8)

        self._interpreter.set_tensor(self._input_details["index"], sample)
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._output_details["index"])[0]

        out_scale, out_zero = self._output_details.get("quantization", (0.0, 0))
        if self._output_details["dtype"] == np.int8 and out_scale > 0:
            out = (out.astype(np.float32) - out_zero) * out_scale
        return out.astype(np.float32)

    def push_frame(self, frame_features: np.ndarray, run_inference: bool = True) -> Dict[str, Any]:
        """Push one 128-dim frame feature vector into buffer and return structured prediction."""
        self.buffer.append(np.asarray(frame_features, dtype=np.float32))
        now_iso = datetime.now(timezone.utc).isoformat()

        if len(self.buffer) < self.sequence_length:
            return {
                "gesture": config.INFERENCE_CONFIG.unrecognized_label,
                "confidence": 0.0,
                "recognized": False,
                "timestamp": now_iso,
                "buffer_ready": False,
                "buffer_fill": len(self.buffer),
                "buffer_capacity": self.sequence_length,
                "latency_ms": 0.0,
            }

        seq = np.stack(list(self.buffer), axis=0)
        active_frames = int(np.sum((seq[:, 126] > 0.5) | (seq[:, 127] > 0.5)))
        if active_frames < 4:
            self.smoother.reset()
            return {
                "gesture": config.INFERENCE_CONFIG.unrecognized_label,
                "confidence": 0.0,
                "recognized": False,
                "timestamp": now_iso,
                "buffer_ready": True,
                "buffer_fill": len(self.buffer),
                "buffer_capacity": self.sequence_length,
                "latency_ms": 0.0,
            }

        if not run_inference:
            return {
                "gesture": config.INFERENCE_CONFIG.unrecognized_label,
                "confidence": 0.0,
                "recognized": False,
                "timestamp": now_iso,
                "buffer_ready": True,
                "buffer_fill": len(self.buffer),
                "buffer_capacity": self.sequence_length,
                "latency_ms": 0.0,
                "skipped_stride": True,
            }

        seq_interp = interpolate_short_dropouts(seq)
        t0 = time.perf_counter()
        probs = self.predict_sequence(seq_interp)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        smoothed = self.smoother.update(probs)

        return {
            "gesture": smoothed.gesture,
            "confidence": smoothed.confidence,
            "recognized": smoothed.recognized,
            "timestamp": now_iso,
            "buffer_ready": True,
            "buffer_fill": len(self.buffer),
            "buffer_capacity": self.sequence_length,
            "consistent_frames": smoothed.consistent_count,
            "latency_ms": round(latency_ms, 3),
        }

    def get_buffered_sequence(self) -> Optional[np.ndarray]:
        """Return a copy of the current buffered (T, D) sequence if full."""
        if len(self.buffer) < self.sequence_length:
            return None
        return np.stack(list(self.buffer), axis=0).astype(np.float32)
