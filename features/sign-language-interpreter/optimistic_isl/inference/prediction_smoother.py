"""Temporal prediction smoothing and confidence thresholding."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence

import numpy as np


@dataclass
class SmoothedPrediction:
    class_id: Optional[int]
    gesture: str
    confidence: float
    recognized: bool
    consistent_count: int


class PredictionSmoother:
    """Stabilizes frame-by-frame gesture predictions with EMA + majority buffer."""

    def __init__(
        self,
        class_names: Sequence[str],
        confidence_threshold: float = 0.70,
        window_size: int = 7,
        min_consistent_frames: int = 4,
        ema_alpha: float = 0.45,
        unrecognized_label: str = "Gesture not recognized",
    ) -> None:
        if min_consistent_frames > window_size:
            raise ValueError("min_consistent_frames cannot exceed window_size")
        self.class_names = list(class_names)
        self.confidence_threshold = float(confidence_threshold)
        self.window_size = int(window_size)
        self.min_consistent_frames = int(min_consistent_frames)
        self.ema_alpha = float(ema_alpha)
        self.unrecognized_label = unrecognized_label

        self._ema_probs: Optional[np.ndarray] = None
        self._label_buffer: Deque[Optional[int]] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._ema_probs = None
        self._label_buffer.clear()

    def update(self, raw_probs: np.ndarray) -> SmoothedPrediction:
        """Update smoother with latest probability distribution (num_classes,)."""
        probs = np.asarray(raw_probs, dtype=np.float32)
        if probs.ndim != 1 or len(probs) != len(self.class_names):
            raise ValueError(
                f"Expected probs shape ({len(self.class_names)},), got {probs.shape}"
            )

        if self._ema_probs is None:
            self._ema_probs = probs.copy()
        else:
            self._ema_probs = (
                self.ema_alpha * probs + (1.0 - self.ema_alpha) * self._ema_probs
            )

        top_id = int(np.argmax(self._ema_probs))
        top_conf = float(self._ema_probs[top_id])

        candidate = top_id if top_conf >= self.confidence_threshold else None
        self._label_buffer.append(candidate)

        valid_votes = [lbl for lbl in self._label_buffer if lbl is not None]
        if not valid_votes:
            return SmoothedPrediction(
                class_id=None,
                gesture=self.unrecognized_label,
                confidence=round(top_conf, 4),
                recognized=False,
                consistent_count=0,
            )

        winner_id, count = Counter(valid_votes).most_common(1)[0]
        if count >= self.min_consistent_frames:
            winner_conf = float(self._ema_probs[winner_id])
            if winner_conf >= self.confidence_threshold:
                return SmoothedPrediction(
                    class_id=winner_id,
                    gesture=self.class_names[winner_id],
                    confidence=round(winner_conf, 4),
                    recognized=True,
                    consistent_count=count,
                )

        return SmoothedPrediction(
            class_id=None,
            gesture=self.unrecognized_label,
            confidence=round(top_conf, 4),
            recognized=False,
            consistent_count=count,
        )
