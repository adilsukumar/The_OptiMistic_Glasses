"""Standalone lightweight MediaPipe hand detector for Raspberry Pi deployment."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def normalize_hand_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks_xyz, dtype=np.float32)
    centered = pts - pts[0:1, :]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-5:
        scale = 1.0
    return (centered / scale).reshape(-1).astype(np.float32)


def build_frame_feature_vector(
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
) -> np.ndarray:
    feat = np.zeros(128, dtype=np.float32)
    if left_hand is not None:
        feat[0:63] = normalize_hand_landmarks(left_hand)
        feat[126] = 1.0
    if right_hand is not None:
        feat[63:126] = normalize_hand_landmarks(right_hand)
        feat[127] = 1.0
    return feat


class PiHandDetector:
    """Low-latency MediaPipe hand detector optimized for Raspberry Pi 4 / Pi 5."""

    def __init__(
        self,
        task_model_path: Path | None = None,
        detection_width: int = 320,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self.detection_width = detection_width
        self._mode = "tasks" if hasattr(mp, "tasks") else "solutions"

        if self._mode == "tasks":
            if task_model_path is None:
                task_model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
            if not task_model_path.exists():
                raise FileNotFoundError(
                    f"Missing {task_model_path}. Run setup_pi.sh or copy hand_landmarker.task here."
                )
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(task_model_path)),
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
        else:
            self._detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,  # Lite model for Raspberry Pi speed
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )

    def process_bgr_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
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
        left_hand: Optional[np.ndarray] = None
        right_hand: Optional[np.ndarray] = None
        raw_hands: List[np.ndarray] = []

        if self._mode == "tasks":
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            res = self._detector.detect(mp_img)
            for idx, lms in enumerate(res.hand_landmarks or []):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                raw_hands.append(coords)
                label = (
                    res.handedness[idx][0].category_name
                    if res.handedness and idx < len(res.handedness)
                    else "Right"
                )
                if label.lower() == "left":
                    if left_hand is None:
                        left_hand = coords
                    elif right_hand is None:
                        right_hand = coords
                else:
                    if right_hand is None:
                        right_hand = coords
                    elif left_hand is None:
                        left_hand = coords
        else:
            res = self._detector.process(rgb)
            if res.multi_hand_landmarks:
                handedness = res.multi_handedness or []
                for idx, hand_lm in enumerate(res.multi_hand_landmarks):
                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                        dtype=np.float32,
                    )
                    raw_hands.append(coords)
                    label = (
                        handedness[idx].classification[0].label
                        if idx < len(handedness)
                        else "Right"
                    )
                    if label.lower() == "left":
                        if left_hand is None:
                            left_hand = coords
                        elif right_hand is None:
                            right_hand = coords
                    else:
                        if right_hand is None:
                            right_hand = coords
                        elif left_hand is None:
                            left_hand = coords

        feat = build_frame_feature_vector(left_hand, right_hand)
        detected = (left_hand is not None) or (right_hand is not None)
        return feat, detected, raw_hands

    def close(self) -> None:
        if hasattr(self._detector, "close"):
            self._detector.close()
