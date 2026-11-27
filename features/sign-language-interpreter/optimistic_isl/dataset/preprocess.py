"""End-to-end preprocessing pipeline for the INCLUDE ISL dataset.

1. Discovers unique raw videos (deduplicated by canonical path & file hash).
2. Extracts MediaPipe 21-point hand landmarks per frame (left & right hands).
3. Normalizes coordinates, interpolates short dropouts, and builds fixed-length
   temporal sequences.
4. Saves label_mapping.json, dataset_statistics.json, visualizations, and
   leakage-resistant train/validation/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dataset.extract_landmarks import (
    HAND_CONNECTIONS,
    MediaPipeHandExtractor,
    extract_video_sequence,
)
from dataset.split_dataset import save_splits, stratified_leakage_resistant_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess")

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def clean_sign_display_name(raw_label: str) -> str:
    """Convert raw INCLUDE folder label (e.g. '48. Hello') to clean display word ('HELLO')."""
    cleaned = re.sub(r"^(?:\d+|Ex)\.\s*", "", raw_label.strip())
    return cleaned.strip().upper()


def compute_file_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_hf_include50_lookup() -> Dict[str, bool]:
    """Load include_50 flags by canonical video_path from cached HF metadata if available."""
    if not config.HF_METADATA_PATH.exists():
        return {}
    with open(config.HF_METADATA_PATH, "r", encoding="utf-8") as f:
        hf = json.load(f)
    lookup: Dict[str, bool] = {}
    for rows in hf.values():
        for r in rows:
            vp = r["video_path"]
            lookup[vp] = lookup.get(vp, False) or bool(r.get("include_50", False))
    return lookup


def discover_unique_videos(raw_videos_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Scan raw_videos_dir and deduplicate by canonical relative path and file MD5."""
    video_paths = sorted(
        p
        for p in raw_videos_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    inc50_lookup = load_hf_include50_lookup()
    seen_hashes = set()
    discovered: List[Dict[str, Any]] = []
    skipped_hash_dups = 0

    for vf in video_paths:
        rel = vf.relative_to(raw_videos_dir).as_posix()
        parts = rel.split("/")
        if len(parts) >= 3:
            parent_label = parts[0]
            raw_label = parts[1]
        elif len(parts) == 2:
            parent_label = "Uncategorized"
            raw_label = parts[0]
        else:
            continue

        file_md5 = compute_file_md5(vf)
        if file_md5 in seen_hashes:
            skipped_hash_dups += 1
            continue
        seen_hashes.add(file_md5)

        discovered.append(
            {
                "abs_path": vf,
                "video_path": rel,
                "parent_label": parent_label,
                "label": raw_label,
                "display_label": clean_sign_display_name(raw_label),
                "include_50": inc50_lookup.get(rel, False),
                "file_md5": file_md5,
            }
        )

    return discovered, skipped_hash_dups


def generate_visualizations(
    records: List[Dict[str, Any]],
    sample_hands: List[Tuple[str, np.ndarray]],
    split_summary: Dict[str, Any],
) -> None:
    """Save class distribution and sample 21-point MediaPipe hand landmark visualizations."""
    config.LANDMARK_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Class distribution plot across train / val / test
    labels = sorted({r["display_label"] for r in records})
    raw_by_disp = {r["display_label"]: r["label"] for r in records}

    tr_counts = [split_summary["train"]["class_counts"].get(raw_by_disp[l], 0) for l in labels]
    va_counts = [split_summary["validation"]["class_counts"].get(raw_by_disp[l], 0) for l in labels]
    te_counts = [split_summary["test"]["class_counts"].get(raw_by_disp[l], 0) for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 6))
    ax.bar(x, tr_counts, label="Train", color="#2563eb")
    ax.bar(x, va_counts, bottom=tr_counts, label="Validation", color="#f59e0b")
    bottom_te = np.array(tr_counts) + np.array(va_counts)
    ax.bar(x, te_counts, bottom=bottom_te, label="Test", color="#10b981")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Video Sequences")
    ax.set_title("Optimistic ISL — Processed Dataset Class Distribution by Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LANDMARK_VIZ_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    # 2. Sample 21-point hand landmark skeletons
    if sample_hands:
        n = min(6, len(sample_hands))
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows))
        axes_arr = np.atleast_1d(axes).ravel()
        for i in range(len(axes_arr)):
            ax = axes_arr[i]
            if i < n:
                title, pts = sample_hands[i]
                xs, ys = pts[:, 0], pts[:, 1]
                for a, b in HAND_CONNECTIONS:
                    ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#3b82f6", linewidth=2)
                ax.scatter(xs, ys, c="#ef4444", s=28, zorder=3)
                ax.scatter([xs[0]], [ys[0]], c="#10b981", s=55, zorder=4, label="Wrist")
                ax.set_title(title, fontsize=10)
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.grid(alpha=0.25)
            else:
                ax.axis("off")
        plt.suptitle("Sample Extracted 21-Point Hand Landmarks", fontsize=12)
        plt.tight_layout()
        plt.savefig(config.LANDMARK_VIZ_DIR / "sample_landmarks.png", dpi=160)
        plt.close(fig)


def run_preprocessing(include_50_only: bool = False) -> Dict[str, Any]:
    """Execute the full preprocessing pipeline on downloaded INCLUDE videos."""
    config.ensure_directories()
    discovered, skipped_hash_dups = discover_unique_videos(config.RAW_VIDEOS_DIR)
    if include_50_only:
        discovered = [d for d in discovered if d["include_50"]]

    if not discovered:
        raise RuntimeError(
            f"No videos found in {config.RAW_VIDEOS_DIR}. "
            "Run dataset/download_dataset.py first."
        )

    logger.info(
        "Discovered %d unique videos (skipped %d MD5 duplicate files).",
        len(discovered),
        skipped_hash_dups,
    )

    extractor = MediaPipeHandExtractor()
    cache_dir = config.PROCESSED_DIR / "landmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    valid_records: List[Dict[str, Any]] = []
    sequences: List[np.ndarray] = []
    rejected: List[Dict[str, Any]] = []
    sample_hands: List[Tuple[str, np.ndarray]] = []
    seen_sample_classes = set()

    try:
        for idx, item in enumerate(discovered, start=1):
            cache_file = cache_dir / f"{item['file_md5']}.npz"
            if cache_file.exists():
                cached = np.load(cache_file, allow_pickle=False)
                meta_json = json.loads(str(cached["meta_json"]))
                if not meta_json.get("ok", False):
                    res = meta_json
                else:
                    res = {
                        **meta_json,
                        "sequence": cached["sequence"],
                        "sample_raw_hand": (
                            cached["sample_raw_hand"]
                            if "sample_raw_hand" in cached
                            else None
                        ),
                    }
            else:
                res = extract_video_sequence(item["abs_path"], extractor)
                meta_only = {
                    k: v
                    for k, v in res.items()
                    if k not in ("sequence", "sample_raw_hand")
                }
                if res["ok"]:
                    save_kwargs = {
                        "sequence": res["sequence"],
                        "meta_json": np.array(json.dumps(meta_only)),
                    }
                    if res["sample_raw_hand"] is not None:
                        save_kwargs["sample_raw_hand"] = res["sample_raw_hand"]
                    np.savez_compressed(cache_file, **save_kwargs)
                else:
                    np.savez_compressed(
                        cache_file, meta_json=np.array(json.dumps(meta_only))
                    )
            if not res["ok"]:
                rejected.append({"video_path": item["video_path"], **res})
                continue

            rec = {
                "video_path": item["video_path"],
                "parent_label": item["parent_label"],
                "label": item["label"],
                "display_label": item["display_label"],
                "include_50": item["include_50"],
                "file_md5": item["file_md5"],
                "fps": res["fps"],
                "width": res["width"],
                "height": res["height"],
                "total_frames": res["total_frames"],
                "decoded_frames": res["decoded_frames"],
                "detected_frames": res["detected_frames"],
                "missing_ratio": res["missing_ratio"],
            }
            valid_records.append(rec)
            sequences.append(res["sequence"])

            if (
                res["sample_raw_hand"] is not None
                and item["display_label"] not in seen_sample_classes
                and len(sample_hands) < 6
            ):
                sample_hands.append((item["display_label"], res["sample_raw_hand"]))
                seen_sample_classes.add(item["display_label"])

            if idx % 25 == 0 or idx == len(discovered):
                logger.info(
                    "Processed %d/%d videos (valid=%d, rejected=%d)",
                    idx,
                    len(discovered),
                    len(valid_records),
                    len(rejected),
                )
    finally:
        extractor.close()

    if not valid_records:
        raise RuntimeError("All videos were rejected during landmark extraction.")

    # Build deterministic label mapping preserving original INCLUDE labels
    unique_raw_labels = sorted({r["label"] for r in valid_records})
    label_to_id = {lbl: idx for idx, lbl in enumerate(unique_raw_labels)}
    id_to_label: Dict[str, Dict[str, Any]] = {}
    for lbl, idx in label_to_id.items():
        first_rec = next(r for r in valid_records if r["label"] == lbl)
        id_to_label[str(idx)] = {
            "id": idx,
            "original_label": lbl,
            "display_label": first_rec["display_label"],
            "parent_label": first_rec["parent_label"],
        }

    for rec in valid_records:
        rec["label_id"] = label_to_id[rec["label"]]

    label_mapping = {
        "num_classes": len(unique_raw_labels),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "id_to_display": {str(i): info["display_label"] for i, info in enumerate(id_to_label.values())},
    }
    with open(config.LABEL_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    X_all = np.stack(sequences, axis=0).astype(np.float32)
    split_indices = stratified_leakage_resistant_split(valid_records, X_all)
    split_summary = save_splits(valid_records, X_all, split_indices)

    generate_visualizations(valid_records, sample_hands, split_summary)

    fps_vals = [r["fps"] for r in valid_records if r["fps"] > 0]
    missing_ratios = [r["missing_ratio"] for r in valid_records]
    reject_reasons = dict(Counter(r["reason"] for r in rejected))

    stats = {
        "total_discovered_unique_videos": len(discovered),
        "skipped_md5_duplicates": skipped_hash_dups,
        "valid_sequences": len(valid_records),
        "rejected_sequences": len(rejected),
        "rejection_reasons": reject_reasons,
        "removed_sequence_duplicates": split_indices["removed_duplicates"],
        "num_classes": len(unique_raw_labels),
        "sequence_shape": [
            config.DATASET_CONFIG.sequence_length,
            config.DATASET_CONFIG.feature_dim,
        ],
        "mean_missing_hand_ratio": round(float(np.mean(missing_ratios)), 4),
        "mean_fps": round(float(np.mean(fps_vals)), 2) if fps_vals else 0.0,
        "splits": split_summary,
        "rejected_samples_log": rejected[:50],
    }
    with open(config.DATASET_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Preprocessing complete. Statistics saved to %s", config.DATASET_STATS_PATH)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess INCLUDE dataset videos.")
    parser.add_argument(
        "--include-50",
        action="store_true",
        help="Filter to INCLUDE-50 subset only.",
    )
    args = parser.parse_args()
    stats = run_preprocessing(include_50_only=args.include_50)
    print(json.dumps({k: v for k, v in stats.items() if k != "rejected_samples_log"}, indent=2))


if __name__ == "__main__":
    main()
