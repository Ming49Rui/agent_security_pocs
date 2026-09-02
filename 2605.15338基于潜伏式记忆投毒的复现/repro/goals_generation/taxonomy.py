"""Taxonomy and seed data loader for goals v2 pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Optional

_DIR = Path(__file__).parent
_TAXONOMY_PATH = _DIR / "taxonomy.json"
_SEEDS_PATH = _DIR / "seeds.json"

# Lazy caches
_taxonomy_cache: Optional[List[Dict]] = None
_seeds_cache: Optional[Dict] = None


def _load_taxonomy() -> List[Dict]:
    global _taxonomy_cache
    if _taxonomy_cache is None:
        with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            _taxonomy_cache = json.load(f)
    return _taxonomy_cache


def _load_seeds() -> Dict:
    global _seeds_cache
    if _seeds_cache is None:
        with open(_SEEDS_PATH, "r", encoding="utf-8") as f:
            _seeds_cache = json.load(f)
    return _seeds_cache


def get_all_subcategories() -> List[Dict]:
    """Return all 5 subcategory definitions."""
    return _load_taxonomy().copy()


def get_subcategory(subcategory_id: str) -> Optional[Dict]:
    """Get a single subcategory by ID (e.g. '1a', '2b', '3')."""
    for subcat in _load_taxonomy():
        if subcat["subcategory_id"] == subcategory_id:
            return subcat
    return None


def filter_subcategories(subcategory_ids: List[str]) -> List[Dict]:
    """Filter subcategories by a list of IDs."""
    id_set = set(subcategory_ids)
    return [s for s in _load_taxonomy() if s["subcategory_id"] in id_set]


def _find_seed_group(subcategory_id: str) -> tuple[str, Dict]:
    """Find the seed group that contains a given subcategory ID."""
    seeds_data = _load_seeds()
    for seed_type, group in seeds_data.items():
        if subcategory_id in group["subcategories"]:
            return seed_type, group
    raise KeyError(f"No seeds found for subcategory '{subcategory_id}'")


def get_seeds(subcategory_id: str) -> List[str]:
    """Get the 20 domain seeds for a subcategory."""
    _, group = _find_seed_group(subcategory_id)
    return group["seeds"]


def get_seed_type(subcategory_id: str) -> str:
    """Get the seed type label for a subcategory (e.g. 'industry', 'topic_domain')."""
    seed_type, _ = _find_seed_group(subcategory_id)
    return seed_type


def get_all_subcategory_ids() -> List[str]:
    """Return all subcategory IDs in order."""
    return [s["subcategory_id"] for s in _load_taxonomy()]


def get_num_seeds(subcategory_id: str) -> int:
    """Return how many seeds a subcategory has."""
    return len(get_seeds(subcategory_id))
