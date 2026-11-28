"""Master Indian Sign Language (ISL) Database & Lookup Engine.
Contains 541 curated, authentic ISL signs based on ISLRTC and INCLUDE standards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_PATH = Path(__file__).resolve().parent / "data" / "isl_master_vocabulary.json"
if not _DATA_PATH.exists():
    _DATA_PATH = Path(__file__).resolve().parent / "web" / "isl_vocabulary.json"

with open(_DATA_PATH, "r", encoding="utf-8") as f:
    ISL_MASTER_VOCABULARY: List[Dict[str, Any]] = json.load(f)

# Fast O(1) indices
BY_ID: Dict[str, Dict[str, Any]] = {}
BY_GLOSS: Dict[str, Dict[str, Any]] = {}
BY_CATEGORY: Dict[str, List[Dict[str, Any]]] = {}

for entry in ISL_MASTER_VOCABULARY:
    BY_ID[entry["id"]] = entry
    gloss = entry.get("gloss")
    if gloss:
        g_up = gloss.upper()
        # Ensure full vocabulary words have precedence over single alphabet letter entries
        if g_up not in BY_GLOSS or entry.get("category") != "ISL Alphabet":
            BY_GLOSS[g_up] = entry
    cat = entry.get("category", "General")
    if cat not in BY_CATEGORY:
        BY_CATEGORY[cat] = []
    BY_CATEGORY[cat].append(entry)


def get_sign(key: str) -> Optional[Dict[str, Any]]:
    """Lookup a sign by ID or GLOSS."""
    if not key:
        return None
    k_upper = key.strip().upper()
    if k_upper in BY_GLOSS:
        return BY_GLOSS[k_upper]
    k_lower = key.strip().lower()
    if k_lower in BY_ID:
        return BY_ID[k_lower]
    # Check normalized gloss replacements
    normalized = k_upper.replace(" ", "_").replace("-", "_")
    if normalized in BY_GLOSS:
        return BY_GLOSS[normalized]
    return None


def search_signs(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Search signs across English, Hindi, Gloss, and tags."""
    if not query:
        return ISL_MASTER_VOCABULARY[:limit]
    q = query.strip().lower()
    results = []
    for entry in ISL_MASTER_VOCABULARY:
        score = 0
        if q == entry["id"] or q == entry["gloss"].lower():
            score += 100
        elif q in entry["english"].lower():
            score += 50
        elif q in entry.get("hindi", ""):
            score += 40
        elif any(q in tag for tag in entry.get("tags", [])):
            score += 20
        elif q in entry.get("category", "").lower():
            score += 10
        if score > 0:
            results.append((score, entry))
    results.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in results[:limit]]


def get_categories() -> Dict[str, int]:
    """Return dictionary of categories and count of signs in each."""
    return {cat: len(signs) for cat, signs in BY_CATEGORY.items()}


def get_total_signs_count() -> int:
    """Return the total number of signs in the master dictionary."""
    return len(ISL_MASTER_VOCABULARY)
