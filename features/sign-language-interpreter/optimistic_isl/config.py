"""Central configuration for the Optimistic ISL recognition system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_ZIPS_DIR = DATASET_DIR / "raw_zips"
RAW_VIDEOS_DIR = DATASET_DIR / "raw_videos"
HF_METADATA_PATH = DATASET_DIR / "include_hf_metadata.json"
INSPECTION_REPORT_PATH = DATASET_DIR / "dataset_inspection_report.json"

PROCESSED_DIR = PROJECT_ROOT / "processed_dataset"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "validation"
TEST_DIR = PROCESSED_DIR / "test"
LABEL_MAPPING_PATH = PROCESSED_DIR / "label_mapping.json"
DATASET_STATS_PATH = PROCESSED_DIR / "dataset_statistics.json"
LANDMARK_VIZ_DIR = PROCESSED_DIR / "visualizations"

MODELS_DIR = PROJECT_ROOT / "models"
MEDIAPIPE_MODEL_DIR = MODELS_DIR / "mediapipe"
HAND_LANDMARKER_TASK_PATH = MEDIAPIPE_MODEL_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

SAVED_MODELS_DIR = MODELS_DIR / "saved"
BEST_MODEL_PATH = SAVED_MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = SAVED_DIR_FINAL = SAVED_MODELS_DIR / "final_model.keras"
TRAINING_HISTORY_PATH = SAVED_MODELS_DIR / "training_history.json"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
METRICS_PATH = EVALUATION_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = EVALUATION_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = EVALUATION_DIR / "training_curves.png"

DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
TFLITE_MODEL_PATH = DEPLOYMENT_DIR / "optimistic_model.tflite"
DEPLOYMENT_LABELS_PATH = DEPLOYMENT_DIR / "labels.json"

UNRECOGNIZED_SAMPLES_DIR = PROJECT_ROOT / "unrecognized_samples"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class DatasetConfig:
    hf_dataset_id: str = "ai4bharat/INCLUDE"
    zenodo_record_url: str = "https://zenodo.org/api/records/4010759"
    # Default categories for practical lightweight smart-glasses training;
    # set download_all=True or pass --all in download_dataset.py for full INCLUDE.
    default_categories: List[str] = field(
        default_factory=lambda: ["Seasons", "Greetings", "Pronouns"]
    )
    sequence_length: int = 30
    num_hands: int = 2
    landmarks_per_hand: int = 21
    coords_per_landmark: int = 3
    # 63 (left) + 63 (right) + 2 (hand presence flags) = 128 features per frame
    feature_dim: int = 128
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_missing_frame_ratio: float = 0.65
    max_interpolation_gap: int = 4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class TrainingConfig:
    architecture: str = "tcn_temporal"  # options: tcn_temporal, gru_temporal, mlp_static, hybrid_cnn_gru
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout_rate: float = 0.30
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 8
    use_class_weights: bool = True
    augment_training: bool = True
    random_seed: int = 42


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.70
    smoothing_window: int = 7
    min_consistent_frames: int = 4
    ema_alpha: float = 0.45
    inference_stride: int = 2
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    websocket_host: str = "127.0.0.1"
    websocket_port: int = 8765
    unrecognized_label: str = "Gesture not recognized"


DATASET_CONFIG = DatasetConfig()
TRAINING_CONFIG = TrainingConfig()
INFERENCE_CONFIG = InferenceConfig()


def ensure_directories() -> None:
    """Create standard project output directories if they do not exist."""
    for path in (
        RAW_ZIPS_DIR,
        RAW_VIDEOS_DIR,
        PROCESSED_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        LANDMARK_VIZ_DIR,
        MEDIAPIPE_MODEL_DIR,
        SAVED_MODELS_DIR,
        EVALUATION_DIR,
        DEPLOYMENT_DIR,
        LOGS_DIR,
    ):
        os.makedirs(path, exist_ok=True)
