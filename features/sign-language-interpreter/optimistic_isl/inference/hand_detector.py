"""Real-time MediaPipe hand detection and landmark normalization for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from dataset.extract_landmarks import (
    HAND_CONNECTIONS,
    MediaPipeHandExtractor,
    build_frame_feature_vector,
)
import config

logger = logging.getLogger("hand_detector")


class RealTimeHandDetector:
    """Low-latency wrapper around MediaPipe Hands for real-time camera frames."""

    def __init__(
        self,
        task_model_path: Path = config.HAND_LANDMARKER_TASK_PATH,
        detection_width: int = 480,
        min_detection_confidence: float = config.DATASET_CONFIG.min_hand_detection_confidence,
        min_presence_confidence: float = config.DATASET_CONFIG.min_hand_presence_confidence,
        min_tracking_confidence: float = config.DATASET_CONFIG.min_tracking_confidence,
        foreground_selector: Optional[Callable] = None,
    ) -> None:
        self._foreground_selector = foreground_selector
        self.detection_width = detection_width
        self._extractor = MediaPipeHandExtractor(
            task_model_path=task_model_path,
            num_hands=4 if foreground_selector else config.DATASET_CONFIG.num_hands,
            min_detection_confidence=min_detection_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_bgr_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
        """Detect hands in a BGR frame and return (128-dim feature, detected_bool, raw_hands)."""
        h, w = frame_bgr.shape[:2]
        if self.detection_width and w > self.detection_width:
            scale = self.detection_width / float(w)
            resized = cv2.resize(
                frame_bgr,
                (self.detection_width, max(1, int(h * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = frame_bgr

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self._foreground_selector is None:
            left_hand, right_hand, raw_hands = self._extractor.detect_rgb(rgb)
        else:
            left_hand, right_hand = self._foreground_selector(self._extractor.detect_labeled_rgb(rgb))
            raw_hands = [hand for hand in (left_hand, right_hand) if hand is not None]
        features = build_frame_feature_vector(left_hand, right_hand)
        detected = (left_hand is not None) or (right_hand is not None)
        return features, detected, raw_hands

    @staticmethod
    def draw_landmarks(
        frame_bgr: np.ndarray,
        raw_hands: List[np.ndarray],
    ) -> np.ndarray:
        """Draw 21-point hand landmarks and skeleton connections on a BGR frame."""
        h, w = frame_bgr.shape[:2]
        for hand_pts in raw_hands:
            px_pts = [
                (int(np.clip(pt[0], 0.0, 1.0) * w), int(np.clip(pt[1], 0.0, 1.0) * h))
                for pt in hand_pts
            ]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame_bgr, px_pts[a], px_pts[b], (255, 180, 0), 2)
            for idx, (cx, cy) in enumerate(px_pts):
                color = (0, 255, 120) if idx == 0 else (0, 90, 255)
                cv2.circle(frame_bgr, (cx, cy), 4 if idx == 0 else 3, color, -1)
        return frame_bgr

    def close(self) -> None:
        self._extractor.close()
