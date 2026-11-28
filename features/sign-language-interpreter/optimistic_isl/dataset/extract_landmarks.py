"""MediaPipe hand landmark extraction, normalization, and temporal sequence builder.

Designed to produce identical normalized feature vectors offline during dataset
preprocessing and online during real-time Raspberry Pi / webcam inference.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger("extract_landmarks")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def normalize_hand_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    """Translate to wrist origin and scale-normalize 21 (x, y, z) hand landmarks.

    Args:
        landmarks_xyz: Array of shape (21, 3).

    Returns:
        Flattened float32 array of shape (63,) in [-1, 1] relative to wrist.
    """
    pts = np.asarray(landmarks_xyz, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {pts.shape}")
    centered = pts - pts[0:1, :]
    norms = np.linalg.norm(centered, axis=1)
    scale = float(np.max(norms))
    if scale < 1e-5:
        scale = 1.0
    normalized = centered / scale
    return normalized.reshape(-1).astype(np.float32)


def build_frame_feature_vector(
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
) -> np.ndarray:
    """Assemble the 128-dim per-frame feature vector from optional left/right hands.

    Layout:
        [0:63]   -> Normalized Left hand (21 x 3) or zeros
        [63:126] -> Normalized Right hand (21 x 3) or zeros
        [126]    -> Left hand presence flag (1.0 or 0.0)
        [127]    -> Right hand presence flag (1.0 or 0.0)
    """
    feat = np.zeros(config.DATASET_CONFIG.feature_dim, dtype=np.float32)
    if left_hand is not None:
        feat[0:63] = normalize_hand_landmarks(left_hand)
        feat[126] = 1.0
    if right_hand is not None:
        feat[63:126] = normalize_hand_landmarks(right_hand)
        feat[127] = 1.0
    return feat


def interpolate_short_dropouts(
    sequence: np.ndarray,
    max_gap: int = config.DATASET_CONFIG.max_interpolation_gap,
) -> np.ndarray:
    """Linearly interpolate short missing-hand gaps per hand slot independently."""
    out = sequence.copy()
    num_frames = out.shape[0]
    for flag_idx, start_col, end_col in ((126, 0, 63), (127, 63, 126)):
        present = out[:, flag_idx] > 0.5
        if not np.any(present):
            continue
        valid_indices = np.where(present)[0]
        for k in range(len(valid_indices) - 1):
            i0 = int(valid_indices[k])
            i1 = int(valid_indices[k + 1])
            gap = i1 - i0 - 1
            if 0 < gap <= max_gap:
                v0 = out[i0, start_col:end_col]
                v1 = out[i1, start_col:end_col]
                for step in range(1, gap + 1):
                    alpha = step / (gap + 1)
                    out[i0 + step, start_col:end_col] = (1.0 - alpha) * v0 + alpha * v1
                    out[i0 + step, flag_idx] = 1.0
    return out


def resample_sequence(
    sequence: np.ndarray,
    target_length: int = config.DATASET_CONFIG.sequence_length,
) -> np.ndarray:
    """Resample a variable-length (T, D) landmark sequence to fixed (target_length, D)."""
    T, D = sequence.shape
    if T == 0:
        return np.zeros((target_length, D), dtype=np.float32)
    if T == target_length:
        return sequence.astype(np.float32)

    if T == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    src_idx = np.linspace(0, T - 1, num=target_length)
    low = np.floor(src_idx).astype(int)
    high = np.minimum(low + 1, T - 1)
    alpha = (src_idx - low)[:, None].astype(np.float32)

    resampled = (1.0 - alpha) * sequence[low] + alpha * sequence[high]
    # Keep binary presence flags crisp
    resampled[:, 126] = (resampled[:, 126] >= 0.5).astype(np.float32)
    resampled[:, 127] = (resampled[:, 127] >= 0.5).astype(np.float32)
    return resampled.astype(np.float32)


def trim_inactive_ends(sequence: np.ndarray, min_active_frames: int = 6) -> np.ndarray:
    """Trim leading and trailing frames where neither hand is detected."""
    active = (sequence[:, 126] > 0.5) | (sequence[:, 127] > 0.5)
    idx = np.where(active)[0]
    if len(idx) == 0:
        return sequence
    start, end = int(idx[0]), int(idx[-1]) + 1
    if end - start >= min_active_frames:
        return sequence[start:end]
    return sequence


class MediaPipeHandExtractor:
    """Wrapper supporting MediaPipe Tasks HandLandmarker (1.x) and legacy solutions.hands."""

    def __init__(
        self,
        task_model_path: Path = config.HAND_LANDMARKER_TASK_PATH,
        num_hands: int = config.DATASET_CONFIG.num_hands,
        min_detection_confidence: float = config.DATASET_CONFIG.min_hand_detection_confidence,
        min_presence_confidence: float = config.DATASET_CONFIG.min_hand_presence_confidence,
        min_tracking_confidence: float = config.DATASET_CONFIG.min_tracking_confidence,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self._mode = "tasks" if hasattr(mp, "tasks") else "solutions"

        if self._mode == "tasks":
            if not task_model_path.exists():
                raise FileNotFoundError(
                    f"MediaPipe task model not found at {task_model_path}. "
                    "Run dataset/download_dataset.py first."
                )
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(model_asset_path=str(task_model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
        else:
            self._detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def detect_rgb(
        self, rgb_frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        """Detect hands in an RGB image.

        Returns:
            (left_hand_xyz, right_hand_xyz, raw_2d_landmarks_list)
        """
        hands = self.detect_labeled_rgb(rgb_frame)
        slots = {"left": None, "right": None}
        for label, coords in hands:
            # Preserve the legacy dataset and feature extraction behavior.
            other = "right" if label == "left" else "left"
            if slots[label] is None:
                slots[label] = coords
            elif slots[other] is None:
                slots[other] = coords
        return slots["left"], slots["right"], [coords for _, coords in hands]

    def detect_labeled_rgb(self, rgb_frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        # Keep the original MediaPipe label for foreground selection.
        if self._mode == "tasks":
            result = self._detector.detect(self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            ))
            landmarks = result.hand_landmarks or []
            categories = result.handedness or []
            labels = [items[0].category_name if items else "Right" for items in categories]
        else:
            result = self._detector.process(rgb_frame)
            landmarks = [item.landmark for item in result.multi_hand_landmarks or []]
            labels = [item.classification[0].label for item in result.multi_handedness or []]

        return [
            ("left" if index < len(labels) and labels[index].lower() == "left" else "right",
             np.array([[lm.x, lm.y, lm.z] for lm in points], dtype=np.float32))
            for index, points in enumerate(landmarks)
        ]

    def close(self) -> None:
        if hasattr(self._detector, "close"):
            self._detector.close()


def extract_video_sequence(
    video_path: Path,
    extractor: MediaPipeHandExtractor,
    max_sampled_frames: int = 90,
    target_length: int = config.DATASET_CONFIG.sequence_length,
) -> Dict[str, Any]:
    """Extract normalized temporal hand-landmark sequence from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": False, "reason": "cannot_open_video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        # Fallback: read sequentially if metadata frame count is missing
        frames_rgb: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames_rgb:
            return {"ok": False, "reason": "empty_or_corrupt_video"}
        if len(frames_rgb) > max_sampled_frames:
            idxs = np.linspace(0, len(frames_rgb) - 1, num=max_sampled_frames, dtype=int)
            frames_rgb = [frames_rgb[i] for i in idxs]
    else:
        sample_count = min(total_frames, max_sampled_frames)
        target_indices = set(
            np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()
        )
        frames_rgb = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx in target_indices:
                # Downscale high-res frames (e.g. 1080p) while preserving aspect ratio for speed
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    frame = cv2.resize(
                        frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA
                    )
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
        cap.release()

    if len(frames_rgb) < 4:
        return {"ok": False, "reason": "too_few_decoded_frames"}

    raw_features: List[np.ndarray] = []
    sample_raw_hand: Optional[np.ndarray] = None
    detected_frames = 0

    for rgb in frames_rgb:
        left_hand, right_hand, raw_hands = extractor.detect_rgb(rgb)
        if left_hand is not None or right_hand is not None:
            detected_frames += 1
            if sample_raw_hand is None and raw_hands:
                sample_raw_hand = raw_hands[0]
        raw_features.append(build_frame_feature_vector(left_hand, right_hand))

    seq = np.stack(raw_features, axis=0)
    missing_ratio = 1.0 - (detected_frames / len(frames_rgb))
    if missing_ratio > config.DATASET_CONFIG.max_missing_frame_ratio:
        return {
            "ok": False,
            "reason": "excessive_missing_hands",
            "missing_ratio": round(missing_ratio, 4),
            "decoded_frames": len(frames_rgb),
        }

    seq = trim_inactive_ends(seq)
    seq = interpolate_short_dropouts(seq)
    fixed_seq = resample_sequence(seq, target_length=target_length)

    return {
        "ok": True,
        "sequence": fixed_seq,
        "decoded_frames": len(frames_rgb),
        "total_frames": total_frames if total_frames > 0 else len(frames_rgb),
        "fps": fps,
        "width": width,
        "height": height,
        "detected_frames": detected_frames,
        "missing_ratio": round(missing_ratio, 4),
        "sample_raw_hand": sample_raw_hand,
    }
