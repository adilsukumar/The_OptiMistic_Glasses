"""Independent test-set evaluation for Optimistic ISL recognition model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """Plot normalized & count confusion matrix."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    fig_size = max(8, min(20, n * 0.65))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), np.maximum(row_sums, 1.0))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    if n <= 25:
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                if val > 0:
                    color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    ax.set_ylabel("True Sign")
    ax.set_xlabel("Predicted Sign")
    ax.set_title("Optimistic ISL — Independent Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute (num_classes, num_classes) confusion matrix in pure NumPy."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_names: List[str],
    original_names: List[str],
) -> Dict[str, Any]:
    """Compute accuracy, macro/weighted metrics, per-class metrics, and report text."""
    num_classes = len(display_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    total = int(len(y_true))
    acc = float(np.mean(y_true == y_pred)) if total > 0 else 0.0

    per_class = []
    precisions, recalls, f1s, supports = [], [], [], []
    lines = [
        f"{'Sign':<26} {'Precision':>9} {'Recall':>9} {'F1-Score':>9} {'Support':>8}",
        "-" * 65,
    ]
    for idx in range(num_classes):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        per_class.append(
            {
                "class_id": idx,
                "original_label": original_names[idx],
                "display_label": display_names[idx],
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "support": support,
            }
        )
        lines.append(
            f"{display_names[idx]:<26} {prec:9.4f} {rec:9.4f} {f1:9.4f} {support:8d}"
        )

    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    sup_arr = np.array(supports, dtype=np.float64)
    weighted_f1 = (
        float(np.sum(np.array(f1s) * sup_arr) / sup_arr.sum())
        if sup_arr.sum() > 0
        else 0.0
    )
    lines.append("-" * 65)
    lines.append(f"{'Accuracy':<26} {'':>9} {'':>9} {acc:9.4f} {total:8d}")
    lines.append(f"{'Macro Avg':<26} {macro_p:9.4f} {macro_r:9.4f} {macro_f1:9.4f} {total:8d}")
    lines.append(f"{'Weighted Avg':<26} {'':>9} {'':>9} {weighted_f1:9.4f} {total:8d}")

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "report_text": "\n".join(lines),
    }


def evaluate_model(
    model_path: Path = config.BEST_MODEL_PATH,
    confidence_threshold: float = config.INFERENCE_CONFIG.confidence_threshold,
) -> Dict[str, Any]:
    """Evaluate trained Keras model on validation and independent test sets."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    config.ensure_directories()
    model = keras.models.load_model(model_path)

    with open(config.LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    num_classes = int(label_mapping["num_classes"])
    display_names = [label_mapping["id_to_display"][str(i)] for i in range(num_classes)]
    original_names = [label_mapping["id_to_label"][str(i)]["original_label"] for i in range(num_classes)]

    # Validation metrics (for clear separation from test metrics)
    X_val = np.load(config.VAL_DIR / "X.npy").astype(np.float32)
    y_val = np.load(config.VAL_DIR / "y.npy").astype(np.int32)
    val_probs = model.predict(X_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_summary = compute_classification_metrics(y_val, val_preds, display_names, original_names)
    val_metrics = {
        "num_samples": int(len(y_val)),
        "accuracy": val_summary["accuracy"],
        "macro_f1": val_summary["macro_f1"],
    }

    # Independent Test Set Evaluation
    X_test = np.load(config.TEST_DIR / "X.npy").astype(np.float32)
    y_test = np.load(config.TEST_DIR / "y.npy").astype(np.int32)
    test_probs = model.predict(X_test, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_confs = np.max(test_probs, axis=1)

    test_summary = compute_classification_metrics(y_test, test_preds, display_names, original_names)
    cm = test_summary["confusion_matrix"]
    plot_confusion_matrix(cm, display_names, config.CONFUSION_MATRIX_PATH)

    per_class = test_summary["per_class"]
    weakest_signs = sorted(per_class, key=lambda c: (c["f1_score"], c["recall"]))[:5]

    accepted_mask = test_confs >= confidence_threshold
    accepted_acc = (
        float(np.mean(y_test[accepted_mask] == test_preds[accepted_mask]))
        if np.any(accepted_mask)
        else 0.0
    )

    metrics = {
        "model_path": str(model_path),
        "validation_performance": val_metrics,
        "test_performance": {
            "num_samples": int(len(y_test)),
            "accuracy": test_summary["accuracy"],
            "macro_precision": test_summary["macro_precision"],
            "macro_recall": test_summary["macro_recall"],
            "macro_f1": test_summary["macro_f1"],
            "weighted_f1": test_summary["weighted_f1"],
            "confidence_threshold": confidence_threshold,
            "thresholded_coverage": round(float(np.mean(accepted_mask)), 4),
            "thresholded_accuracy": round(accepted_acc, 4),
        },
        "per_class_metrics": per_class,
        "weakest_performing_signs": weakest_signs,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": test_summary["report_text"],
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Evaluation complete | Val Acc=%.4f | Test Acc=%.4f | Test Macro-F1=%.4f",
        val_metrics["accuracy"],
        metrics["test_performance"]["accuracy"],
        metrics["test_performance"]["macro_f1"],
    )
    print(test_summary["report_text"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained ISL model on test set.")
    parser.add_argument("--model", type=Path, default=config.BEST_MODEL_PATH)
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.INFERENCE_CONFIG.confidence_threshold,
    )
    args = parser.parse_args()
    evaluate_model(model_path=args.model, confidence_threshold=args.threshold)


if __name__ == "__main__":
    main()
