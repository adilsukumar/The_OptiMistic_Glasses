"""Reproducible download script for ai4bharat/INCLUDE videos (Zenodo) and MediaPipe task model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_dataset")


def fetch_zenodo_files(
    zenodo_url: str = config.DATASET_CONFIG.zenodo_record_url,
) -> List[Dict]:
    """Fetch the list of downloadable files from Zenodo record 4010759."""
    req = urllib.request.Request(
        zenodo_url,
        headers={"User-Agent": "Optimistic-ISL/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("files", [])


def verify_md5(file_path: Path, expected_checksum: str) -> bool:
    """Verify file MD5 against Zenodo 'md5:<hex>' checksum string."""
    if not expected_checksum or not expected_checksum.startswith("md5:"):
        return True
    expected_hex = expected_checksum.split(":", 1)[1].strip().lower()
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual_hex = hasher.hexdigest().lower()
    if actual_hex != expected_hex:
        logger.warning(
            "Checksum mismatch for %s: expected %s, got %s",
            file_path.name,
            expected_hex,
            actual_hex,
        )
        return False
    return True


def download_file_with_resume(
    url: str,
    dest: Path,
    expected_size: Optional[int] = None,
    max_retries: int = 5,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file with HTTP Range resume and exponential backoff retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        existing_size = dest.stat().st_size if dest.exists() else 0
        if expected_size and existing_size == expected_size:
            return dest
        if expected_size and existing_size > expected_size:
            logger.warning("Corrupt partial file %s; restarting.", dest.name)
            dest.unlink()
            existing_size = 0

        headers = {"User-Agent": "Optimistic-ISL/1.0"}
        mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            logger.info(
                "Resuming %s from %.1f MB...",
                dest.name,
                existing_size / 1e6,
            )
        else:
            logger.info("Downloading %s...", dest.name)

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if mode == "ab" and status == 200:
                    # Server ignored Range header; restart from 0
                    mode = "wb"
                    existing_size = 0
                downloaded = existing_size
                last_log = time.time()
                with open(dest, mode) as out_f:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        out_f.write(buf)
                        downloaded += len(buf)
                        if time.time() - last_log >= 10.0:
                            pct = (
                                f" ({100.0 * downloaded / expected_size:.1f}%)"
                                if expected_size
                                else ""
                            )
                            logger.info(
                                "  %s: %.1f MB%s",
                                dest.name,
                                downloaded / 1e6,
                                pct,
                            )
                            last_log = time.time()
            if expected_size is None or dest.stat().st_size == expected_size:
                return dest
        except Exception as exc:
            wait_sec = min(30, 2 ** attempt)
            logger.warning(
                "Download attempt %d/%d failed for %s: %s. Retrying in %ds...",
                attempt,
                max_retries,
                dest.name,
                exc,
                wait_sec,
            )
            time.sleep(wait_sec)

    raise RuntimeError(f"Failed to download {dest.name} after {max_retries} attempts.")


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a Zenodo INCLUDE archive cleanly into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s into %s...", zip_path.name, target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def ensure_mediapipe_model() -> Path:
    """Download the official MediaPipe HandLandmarker task asset if missing."""
    dst = config.HAND_LANDMARKER_TASK_PATH
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    logger.info("Downloading MediaPipe hand_landmarker.task...")
    download_file_with_resume(config.HAND_LANDMARKER_URL, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download INCLUDE dataset video archives from Zenodo."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=config.DATASET_CONFIG.default_categories,
        help="Category prefixes to download (e.g., Seasons Greetings Pronouns).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download ALL 44 INCLUDE zip archives (~56 GB compressed).",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete each .zip archive after successful extraction to save disk space.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available Zenodo zip archives and exit.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent archive downloads (default: 2).",
    )
    args = parser.parse_args()

    config.ensure_directories()
    ensure_mediapipe_model()

    files = fetch_zenodo_files()
    zip_files = sorted(
        [f for f in files if f["key"].endswith(".zip")],
        key=lambda x: x["key"],
    )

    if args.list_only:
        for f in zip_files:
            print(f"{f['key']:35s} {f['size']/1e6:8.2f} MB  {f.get('checksum','')}")
        return

    if args.all:
        selected = zip_files
    else:
        prefixes = tuple(f"{cat}_" for cat in args.categories)
        selected = [f for f in zip_files if f["key"].startswith(prefixes)]

    if not selected:
        raise ValueError(f"No Zenodo zip archives matched categories={args.categories}")

    total_gb = sum(f["size"] for f in selected) / 1e9
    logger.info(
        "Selected %d zip archive(s) totaling %.2f GB: %s",
        len(selected),
        total_gb,
        [f["key"] for f in selected],
    )

    def _process_item(item: Dict) -> str:
        zip_path = config.RAW_ZIPS_DIR / item["key"]
        download_file_with_resume(
            url=item["links"]["self"],
            dest=zip_path,
            expected_size=item["size"],
        )
        if not verify_md5(zip_path, item.get("checksum", "")):
            logger.warning("Re-downloading %s due to MD5 mismatch...", item["key"])
            zip_path.unlink(missing_ok=True)
            download_file_with_resume(
                url=item["links"]["self"],
                dest=zip_path,
                expected_size=item["size"],
            )
            if not verify_md5(zip_path, item.get("checksum", "")):
                raise RuntimeError(f"Checksum verification failed for {item['key']}")

        extract_zip(zip_path, config.RAW_VIDEOS_DIR)
        if args.delete_zips:
            zip_path.unlink(missing_ok=True)
            logger.info("Deleted archive %s after extraction.", item["key"])
        return item["key"]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done_key in pool.map(_process_item, selected):
            logger.info("Completed archive: %s", done_key)

    logger.info("Download & extraction complete.")


if __name__ == "__main__":
    main()
