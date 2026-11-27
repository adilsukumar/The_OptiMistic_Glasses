"""Unit tests for landmark normalization, interpolation, resampling, and splitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.extract_landmarks import (
    build_frame_feature_vector,
    interpolate_short_dropouts,
    normalize_hand_landmarks,
    resample_sequence,
    trim_inactive_ends,
)
from dataset.split_dataset import stratified_leakage_resistant_split


def test_normalize_translation_and_scale_invariance() -> None:
    rng = np.random.default_rng(0)
    hand = rng.normal(0.0, 0.1, size=(21, 3)).astype(np.float32) + np.array([0.4, 0.6, 0.1], dtype=np.float32)
    norm_a = normalize_hand_landmarks(hand)

    # Translate and scale the same hand
    hand_transformed = (hand - np.array([0.2, 0.3, 0.0], dtype=np.float32)) * 2.5
    norm_b = normalize_hand_landmarks(hand_transformed)

    assert norm_a.shape == (63,)
    assert np.allclose(norm_a[:3], 0.0, atol=1e-6)
    assert np.allclose(norm_a, norm_b, atol=1e-5)
    assert np.max(np.linalg.norm(norm_a.reshape(21, 3), axis=1)) <= 1.0 + 1e-5


def test_left_right_slotting_and_interpolation() -> None:
    seq = np.zeros((8, 128), dtype=np.float32)
    hand = np.ones((21, 3), dtype=np.float32)
    hand[0] = 0.0

    seq[0] = build_frame_feature_vector(left_hand=hand, right_hand=None)
    seq[3] = build_frame_feature_vector(left_hand=hand * 2.0, right_hand=None)
    seq[7] = build_frame_feature_vector(left_hand=hand, right_hand=hand)

    assert seq[0, 126] == 1.0 and seq[0, 127] == 0.0
    assert seq[1, 126] == 0.0

    interp = interpolate_short_dropouts(seq, max_gap=3)
    # Gap of 2 frames (1, 2) between 0 and 3 should be filled
    assert interp[1, 126] == 1.0
    assert interp[2, 126] == 1.0
    # Gap of 3 frames (4, 5, 6) between 3 and 7 should also be filled
    assert interp[5, 126] == 1.0
    # Right hand only appears at frame 7, so earlier frames stay 0
    assert interp[5, 127] == 0.0


def test_trim_and_resample_sequence() -> None:
    seq = np.zeros((15, 128), dtype=np.float32)
    seq[3:12, 126] = 1.0
    seq[3:12, :63] = 0.25
    trimmed = trim_inactive_ends(seq, min_active_frames=5)
    assert trimmed.shape == (9, 128)

    resampled = resample_sequence(trimmed, target_length=30)
    assert resampled.shape == (30, 128)
    assert np.all(resampled[:, 126] == 1.0)


def test_leakage_resistant_split_deduplicates_and_has_zero_overlap() -> None:
    rng = np.random.default_rng(42)
    records = []
    seqs = []
    for cls_id in range(3):
        for take in range(10):
            records.append(
                {
                    "video_path": f"Cat/Sign_{cls_id}/MVI_{1000 + cls_id * 100 + take}.MOV",
                    "label": f"Sign_{cls_id}",
                    "label_id": cls_id,
                }
            )
            seqs.append(rng.normal(0.0, 1.0, size=(30, 128)).astype(np.float32))

    # Inject an exact sequence duplicate under a different path
    records.append(
        {
            "video_path": "Cat/Sign_0/MVI_9999.MOV",
            "label": "Sign_0",
            "label_id": 0,
        }
    )
    seqs.append(seqs[0].copy())

    res = stratified_leakage_resistant_split(records, np.stack(seqs, axis=0))
    assert res["removed_duplicates"] == 1
    tr, va, te = set(res["train_idx"]), set(res["val_idx"]), set(res["test_idx"])
    assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0
    assert len(tr) + len(va) + len(te) == 30
