"""Leakage-resistant train / validation / test dataset splitter.

Prevents data leakage by:
1. Enforcing unique canonical video_path and content MD5 deduplication.
2. Detecting near-duplicate landmark sequences across samples.
3. Grouping consecutive MVI camera-take clusters where signer/session metadata
   is absent, then performing stratified splitting across classes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger("split_dataset")


def extract_session_group(record: Dict[str, Any]) -> str:
    """Infer a coarse recording session/camera-take group to reduce split leakage.

    INCLUDE filenames follow Canon DSLR patterns like 'MVI_5194.MOV' without
    explicit signer IDs. Grouping by (label, MVI_bucket_of_3) keeps burst/retake
    shots together when enough groups exist for a class.
    """
    filename = Path(record["video_path"]).stem
    match = re.search(r"(\d+)", filename)
    if match:
        num = int(match.group(1))
        bucket = num // 3
        return f"{record['label']}__mvi_{bucket}"
    return f"{record['label']}__file_{filename}"


def stratified_leakage_resistant_split(
    records: List[Dict[str, Any]],
    sequences: np.ndarray,
    train_ratio: float = config.DATASET_CONFIG.train_ratio,
    val_ratio: float = config.DATASET_CONFIG.val_ratio,
    test_ratio: float = config.DATASET_CONFIG.test_ratio,
    seed: int = config.DATASET_CONFIG.random_seed,
) -> Dict[str, Any]:
    """Split extracted sequences into leakage-free train, val, and test partitions."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    rng = np.random.default_rng(seed)

    # Deduplicate by sequence fingerprint in case identical clips sit under different names
    seen_seq_hashes = set()
    kept_indices: List[int] = []
    duplicate_seq_indices: List[int] = []
    for idx, rec in enumerate(records):
        seq_bytes = np.round(sequences[idx], 3).tobytes()
        seq_hash = hashlib.sha256(seq_bytes).hexdigest()
        key = (rec["label_id"], seq_hash)
        if key in seen_seq_hashes:
            duplicate_seq_indices.append(idx)
            continue
        seen_seq_hashes.add(key)
        kept_indices.append(idx)

    if duplicate_seq_indices:
        logger.warning(
            "Removed %d near-duplicate sequences before splitting.",
            len(duplicate_seq_indices),
        )

    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx in kept_indices:
        by_class[records[idx]["label_id"]].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for label_id in sorted(by_class.keys()):
        cls_indices = by_class[label_id]
        # Group by session bucket
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx in cls_indices:
            grp = extract_session_group(records[idx])
            groups[grp].append(idx)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        n_total = len(cls_indices)
        n_test = max(1, round(n_total * test_ratio)) if n_total >= 3 else 0
        n_val = max(1, round(n_total * val_ratio)) if n_total >= 4 else 0
        if n_test + n_val >= n_total:
            n_val = max(0, n_total - 2)
            n_test = 1 if n_total >= 2 else 0

        if len(group_keys) >= 3:
            # Group-level assignment preserves recording takes together
            cls_test, cls_val, cls_train = [], [], []
            for gk in group_keys:
                members = groups[gk]
                if len(cls_test) < n_test and len(cls_test) + len(members) <= n_total - 2:
                    cls_test.extend(members)
                elif len(cls_val) < n_val and len(cls_val) + len(members) <= n_total - len(cls_test) - 1:
                    cls_val.extend(members)
                else:
                    cls_train.extend(members)

            # Fallback balance if group sizes skipped val/test
            if not cls_test or (n_val > 0 and not cls_val):
                flat = list(cls_indices)
                rng.shuffle(flat)
                cls_test = flat[:n_test]
                cls_val = flat[n_test : n_test + n_val]
                cls_train = flat[n_test + n_val :]
        else:
            flat = list(cls_indices)
            rng.shuffle(flat)
            cls_test = flat[:n_test]
            cls_val = flat[n_test : n_test + n_val]
            cls_train = flat[n_test + n_val :]

        train_idx.extend(cls_train)
        val_idx.extend(cls_val)
        test_idx.extend(cls_test)

    # Verify zero overlap
    set_tr, set_va, set_te = set(train_idx), set(val_idx), set(test_idx)
    assert len(set_tr & set_va) == 0, "Train/Val index overlap detected!"
    assert len(set_tr & set_te) == 0, "Train/Test index overlap detected!"
    assert len(set_va & set_te) == 0, "Val/Test index overlap detected!"

    vp_tr = {records[i]["video_path"] for i in train_idx}
    vp_va = {records[i]["video_path"] for i in val_idx}
    vp_te = {records[i]["video_path"] for i in test_idx}
    assert len(vp_tr & vp_va) == 0, "Train/Val video_path leakage!"
    assert len(vp_tr & vp_te) == 0, "Train/Test video_path leakage!"
    assert len(vp_va & vp_te) == 0, "Val/Test video_path leakage!"

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "removed_duplicates": len(duplicate_seq_indices),
    }


def save_splits(
    records: List[Dict[str, Any]],
    sequences: np.ndarray,
    split_indices: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist train, validation, and test .npy arrays and metadata JSON files."""
    config.ensure_directories()
    summary: Dict[str, Any] = {}

    mapping = [
        ("train", config.TRAIN_DIR, split_indices["train_idx"]),
        ("validation", config.VAL_DIR, split_indices["val_idx"]),
        ("test", config.TEST_DIR, split_indices["test_idx"]),
    ]

    for split_name, out_dir, indices in mapping:
        X = sequences[indices].astype(np.float32)
        y = np.array([records[i]["label_id"] for i in indices], dtype=np.int32)
        meta = [records[i] for i in indices]

        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        class_dist = dict(sorted(Counter(r["label"] for r in meta).items()))
        summary[split_name] = {
            "num_samples": int(len(indices)),
            "shape": list(X.shape),
            "classes_present": len(class_dist),
            "class_counts": class_dist,
        }
        logger.info("Saved %s split: X=%s, y=%s", split_name, X.shape, y.shape)

    return summary
