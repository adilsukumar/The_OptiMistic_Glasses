"""Inspect the ai4bharat/INCLUDE dataset metadata and local video files.

Fetches Hugging Face metadata (via datasets-server API or local cache), checks
Zenodo archive availability/licensing, detects duplicate video_paths and label
mismatches, and optionally inspects downloaded video files using OpenCV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inspect_dataset")

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def fetch_hf_metadata(
    dataset_id: str = config.DATASET_CONFIG.hf_dataset_id,
    cache_path: Path = config.HF_METADATA_PATH,
    force_refresh: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load or download all split rows from Hugging Face datasets-server."""
    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached HF metadata from %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Fetching HF metadata for %s...", dataset_id)
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        offset = 0
        rows: List[Dict[str, Any]] = []
        while True:
            url = (
                f"https://datasets-server.huggingface.co/rows?"
                f"dataset={dataset_id}&config=default&split={split}"
                f"&offset={offset}&length=100"
            )
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            batch = [r["row"] for r in payload.get("rows", [])]
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            total = payload.get("num_rows_total", 0)
            if total and offset >= total:
                break
        all_rows[split] = rows
        logger.info("Fetched split '%s': %d rows", split, len(rows))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    return all_rows


def fetch_zenodo_manifest(
    zenodo_url: str = config.DATASET_CONFIG.zenodo_record_url,
) -> Optional[Dict[str, Any]]:
    """Query Zenodo API for INCLUDE video archive manifest and license."""
    try:
        with urllib.request.urlopen(zenodo_url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = [
            {
                "key": item["key"],
                "size_bytes": item["size"],
                "size_mb": round(item["size"] / 1e6, 2),
                "url": item["links"]["self"],
                "checksum": item.get("checksum", ""),
            }
            for item in data.get("files", [])
        ]
        meta = data.get("metadata", {})
        return {
            "title": meta.get("title"),
            "license": meta.get("license", {}).get("id"),
            "doi": data.get("doi"),
            "total_files": len(files),
            "total_zip_gb": round(
                sum(f["size_bytes"] for f in files if f["key"].endswith(".zip")) / 1e9, 2
            ),
            "files": files,
        }
    except Exception as exc:
        logger.warning("Could not query Zenodo manifest (offline?): %s", exc)
        return None


def analyze_hf_metadata(hf_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze Hugging Face INCLUDE splits, duplicates, mismatches, and class counts."""
    split_counts = {sp: len(rows) for sp, rows in hf_data.items()}
    vp_entries: Dict[str, List[Tuple[str, str, str, bool]]] = defaultdict(list)
    mismatches: List[Dict[str, str]] = []

    for split, rows in hf_data.items():
        for row in rows:
            vp = row["video_path"]
            parts = vp.split("/")
            folder_parent = parts[0] if len(parts) >= 3 else row["parent_label"]
            folder_label = parts[1] if len(parts) >= 3 else row["label"]
            vp_entries[vp].append((split, row["label"], row["parent_label"], bool(row["include_50"])))
            if folder_label != row["label"] or folder_parent != row["parent_label"]:
                mismatches.append(
                    {
                        "split": split,
                        "hf_parent": row["parent_label"],
                        "hf_label": row["label"],
                        "folder_parent": folder_parent,
                        "folder_label": folder_label,
                        "video_path": vp,
                    }
                )

    split_vps = {sp: {r["video_path"] for r in rows} for sp, rows in hf_data.items()}
    cross_split_leakage = {
        "train_val_overlap": len(split_vps.get("train", set()) & split_vps.get("val", set())),
        "train_test_overlap": len(split_vps.get("train", set()) & split_vps.get("test", set())),
        "val_test_overlap": len(split_vps.get("val", set()) & split_vps.get("test", set())),
    }

    # Canonical deduplicated catalog using directory structure as ground truth
    unique_catalog: Dict[str, Dict[str, Any]] = {}
    for vp, entries in vp_entries.items():
        parts = vp.split("/")
        parent_label = parts[0] if len(parts) >= 3 else entries[0][2]
        label = parts[1] if len(parts) >= 3 else entries[0][1]
        unique_catalog[vp] = {
            "video_path": vp,
            "parent_label": parent_label,
            "label": label,
            "include_50": any(e[3] for e in entries),
        }

    category_counts = Counter(v["parent_label"] for v in unique_catalog.values())
    label_counts = Counter(v["label"] for v in unique_catalog.values())
    inc50_counts = Counter(v["label"] for v in unique_catalog.values() if v["include_50"])
    extensions = Counter(Path(vp).suffix.upper() for vp in unique_catalog)

    return {
        "hf_raw_split_counts": split_counts,
        "hf_total_rows": sum(split_counts.values()),
        "duplicate_video_paths_count": sum(1 for v in vp_entries.values() if len(v) > 1),
        "hf_label_folder_mismatches_count": len(mismatches),
        "hf_label_folder_mismatches_sample": mismatches[:15],
        "hf_cross_split_leakage": cross_split_leakage,
        "unique_video_count": len(unique_catalog),
        "unique_parent_categories": len(category_counts),
        "unique_folder_classes": len(label_counts),
        "include_50_unique_videos": sum(1 for v in unique_catalog.values() if v["include_50"]),
        "include_50_classes": len(inc50_counts),
        "extensions": dict(extensions),
        "category_video_counts": dict(sorted(category_counts.items())),
        "min_samples_per_class": min(label_counts.values()) if label_counts else 0,
        "max_samples_per_class": max(label_counts.values()) if label_counts else 0,
    }


def inspect_local_videos(raw_videos_dir: Path = config.RAW_VIDEOS_DIR, max_sample_probe: int = 30) -> Dict[str, Any]:
    """Scan local raw_videos_dir and probe video stream metadata with OpenCV."""
    import cv2

    if not raw_videos_dir.exists():
        return {"available": False, "video_count": 0}

    video_files: List[Path] = [
        p
        for p in sorted(raw_videos_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not video_files:
        return {"available": False, "video_count": 0}

    by_category: Counter = Counter()
    by_class: Counter = Counter()
    for vf in video_files:
        rel = vf.relative_to(raw_videos_dir)
        parts = rel.parts
        if len(parts) >= 3:
            by_category[parts[0]] += 1
            by_class[parts[1]] += 1
        elif len(parts) == 2:
            by_class[parts[0]] += 1

    # Probe evenly spaced sample videos for resolution, fps, frame count, duration
    step = max(1, len(video_files) // max_sample_probe)
    probed = []
    corrupted = []
    for vf in video_files[::step][:max_sample_probe]:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            corrupted.append(str(vf))
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None or frame_count <= 0:
            corrupted.append(str(vf))
            continue
        duration_sec = round(frame_count / fps, 3) if fps > 0 else None
        probed.append(
            {
                "file": str(vf.relative_to(raw_videos_dir)),
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "duration_sec": duration_sec,
            }
        )

    return {
        "available": True,
        "video_count": len(video_files),
        "local_categories": dict(sorted(by_category.items())),
        "local_classes_count": len(by_class),
        "local_classes": dict(sorted(by_class.items())),
        "probed_count": len(probed),
        "corrupted_in_probe": corrupted,
        "sample_probes": probed[:10],
        "resolutions": dict(Counter(f"{p['width']}x{p['height']}" for p in probed)),
        "fps_values": dict(Counter(p["fps"] for p in probed)),
        "frame_count_range": (
            [min(p["frame_count"] for p in probed), max(p["frame_count"] for p in probed)]
            if probed
            else []
        ),
        "duration_range_sec": (
            [
                min(p["duration_sec"] for p in probed if p["duration_sec"] is not None),
                max(p["duration_sec"] for p in probed if p["duration_sec"] is not None),
            ]
            if any(p["duration_sec"] is not None for p in probed)
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ai4bharat/INCLUDE dataset.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch metadata from HuggingFace.")
    parser.add_argument("--skip-zenodo", action="store_true", help="Skip Zenodo API check.")
    args = parser.parse_args()

    config.ensure_directories()
    hf_data = fetch_hf_metadata(force_refresh=args.refresh)
    hf_summary = analyze_hf_metadata(hf_data)
    zenodo_summary = None if args.skip_zenodo else fetch_zenodo_manifest()
    local_summary = inspect_local_videos()

    report = {
        "dataset_id": config.DATASET_CONFIG.hf_dataset_id,
        "license": "CC-BY-4.0",
        "hf_metadata_summary": hf_summary,
        "zenodo_manifest_summary": (
            {k: v for k, v in zenodo_summary.items() if k != "files"}
            if zenodo_summary
            else None
        ),
        "local_videos_summary": local_summary,
    }

    with open(config.INSPECTION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Dataset inspection report saved to %s", config.INSPECTION_REPORT_PATH)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
