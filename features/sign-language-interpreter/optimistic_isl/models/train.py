"""Training pipeline for Optimistic ISL recognition models."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.model_architecture import ARCHITECTURE_DESCRIPTIONS, build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def set_global_seed(seed: int) -> None:
    """Fix random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split_arrays(split_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    x_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Split arrays not found in {split_dir}. Run dataset/preprocess.py first."
        )
    return np.load(x_path).astype(np.float32), np.load(y_path).astype(np.int32)


def augment_landmark_batch(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply realistic landmark-domain augmentations (jitter, scale, time-shift)."""
    X_aug = X.copy()
    B, T, D = X_aug.shape

    for i in range(B):
        # Preserve hand presence masks
        mask_l = X_aug[i, :, 126:127] > 0.5
        mask_r = X_aug[i, :, 127:128] > 0.5

        # Random spatial scale jitter [0.92, 1.08]
        scale = rng.uniform(0.92, 1.08)
        X_aug[i, :, :63] *= scale * mask_l
        X_aug[i, :, 63:126] *= scale * mask_r

        # Small coordinate noise on active hands only
        noise_l = rng.normal(0.0, 0.012, size=(T, 63)).astype(np.float32)
        noise_r = rng.normal(0.0, 0.012, size=(T, 63)).astype(np.float32)
        X_aug[i, :, :63] += noise_l * mask_l
        X_aug[i, :, 63:126] += noise_r * mask_r

        # Small temporal shift [-2, +2] frames with edge padding
        shift = int(rng.integers(-2, 3))
        if shift > 0:
            X_aug[i, shift:] = X_aug[i, :-shift]
            X_aug[i, :shift] = X_aug[i, 0:1]
        elif shift < 0:
            s = -shift
            X_aug[i, :-s] = X_aug[i, s:]
            X_aug[i, -s:] = X_aug[i, -1:]

    return X_aug.astype(np.float32), y


class LandmarkSequenceDataset(keras.utils.Sequence):
    """Keras Sequence yielding augmented batches for training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.X) / self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        xb = self.X[batch_idx]
        yb = self.y[batch_idx]
        if self.augment:
            xb, yb = augment_landmark_batch(xb, yb, self.rng)
        return xb, yb

    def on_epoch_end(self) -> None:
        self.rng.shuffle(self.indices)


def plot_training_curves(history_dict: Dict[str, List[float]], out_path: Path) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(epochs, history_dict["loss"], label="Train Loss", color="#2563eb", linewidth=2)
    if "val_loss" in history_dict:
        ax1.plot(epochs, history_dict["val_loss"], label="Val Loss", color="#ef4444", linewidth=2)
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sparse Categorical Crossentropy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history_dict["accuracy"], label="Train Accuracy", color="#2563eb", linewidth=2)
    if "val_accuracy" in history_dict:
        ax2.plot(epochs, history_dict["val_accuracy"], label="Val Accuracy", color="#10b981", linewidth=2)
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def train_single_architecture(
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    class_weight_dict: Dict[int, float] | None,
    epochs: int,
    batch_size: int,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train one model architecture and optionally persist checkpoints & history."""
    cfg = config.TRAINING_CONFIG
    set_global_seed(cfg.random_seed)

    seq_len, feat_dim = X_train.shape[1], X_train.shape[2]
    model = build_model(
        architecture=architecture,
        sequence_length=seq_len,
        feature_dim=feat_dim,
        num_classes=num_classes,
        dropout_rate=cfg.dropout_rate,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    logger.info(
        "Built '%s' model: %d parameters",
        architecture,
        model.count_params(),
    )

    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.reduce_lr_patience,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    if save_artifacts:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(config.BEST_MODEL_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            )
        )

    train_seq = LandmarkSequenceDataset(
        X_train,
        y_train,
        batch_size=batch_size,
        augment=cfg.augment_training,
        seed=cfg.random_seed,
    )

    history = model.fit(
        train_seq,
        validation_data=(X_val, y_val),
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    history_dict = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    best_val_acc = float(max(history_dict.get("val_accuracy", [0.0])))
    best_val_loss = float(min(history_dict.get("val_loss", [0.0])))

    if save_artifacts:
        model.save(config.FINAL_MODEL_PATH)
        if not config.BEST_MODEL_PATH.exists():
            model.save(config.BEST_MODEL_PATH)
        payload = {
            "architecture": architecture,
            "num_parameters": int(model.count_params()),
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "epochs_ran": len(history_dict["loss"]),
            "history": history_dict,
        }
        with open(config.TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if config.LABEL_MAPPING_PATH.exists():
            shutil.copy2(
                config.LABEL_MAPPING_PATH,
                config.SAVED_MODELS_DIR / "label_mapping.json",
            )
        plot_training_curves(history_dict, config.TRAINING_CURVES_PATH)
        logger.info("Saved best_model.keras, final_model.keras, and training_history.json")

    return {
        "architecture": architecture,
        "num_parameters": int(model.count_params()),
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history_dict["loss"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Optimistic ISL recognition model.")
    parser.add_argument(
        "--arch",
        default=config.TRAINING_CONFIG.architecture,
        choices=list(ARCHITECTURE_DESCRIPTIONS.keys()),
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare mlp_static, gru_temporal, and tcn_temporal on validation set, then train best.",
    )
    parser.add_argument("--epochs", type=int, default=config.TRAINING_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=config.TRAINING_CONFIG.batch_size)
    args = parser.parse_args()

    config.ensure_directories()
    gpus = tf.config.list_physical_devices("GPU")
    logger.info("TensorFlow %s | Physical GPUs: %s", tf.__version__, gpus)

    X_train, y_train = load_split_arrays(config.TRAIN_DIR)
    X_val, y_val = load_split_arrays(config.VAL_DIR)

    with open(config.LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    num_classes = int(label_mapping["num_classes"])

    class_weight_dict = None
    if config.TRAINING_CONFIG.use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    if args.compare:
        candidates = ["mlp_static", "gru_temporal", "tcn_temporal"]
        comparison = []
        for cand in candidates:
            logger.info("--- Benchmarking architecture: %s ---", cand)
            res = train_single_architecture(
                architecture=cand,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=num_classes,
                class_weight_dict=class_weight_dict,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_artifacts=False,
            )
            comparison.append(res)
        with open(config.SAVED_MODELS_DIR / "architecture_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        best_arch = max(comparison, key=lambda r: (r["best_val_accuracy"], -r["best_val_loss"]))["architecture"]
        logger.info("Selected best architecture from comparison: %s", best_arch)
    else:
        best_arch = args.arch

    final_res = train_single_architecture(
        architecture=best_arch,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_classes=num_classes,
        class_weight_dict=class_weight_dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_artifacts=True,
    )
    print(json.dumps(final_res, indent=2))


if __name__ == "__main__":
    main()
