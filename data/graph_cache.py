"""
Lightweight graph/feature caching utilities.

Caches are stored under data/cache/ and keyed by a simple tag string.
Serialization uses joblib for reliability and speed.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from joblib import dump, load


class GraphCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.joblib")

    # Graphs
    def save_graphs(self, key: str, graphs: List[Any], nodes: List[int], time_keys: List[str]) -> str:
        path = self._path(f"{key}_graphs")
        dump({"graphs": graphs, "nodes": nodes, "time_keys": time_keys}, path)
        return path

    def load_graphs(self, key: str) -> Tuple[List[Any], List[int], List[str]]:
        path = self._path(f"{key}_graphs")
        obj = load(path)
        return obj["graphs"], obj["nodes"], obj["time_keys"]

    def graphs_exist(self, key: str) -> bool:
        return os.path.exists(self._path(f"{key}_graphs"))

    # Features
    def save_features(self, key: str, payload: Dict[str, Any]) -> str:
        path = self._path(f"{key}_features")
        dump(payload, path)
        return path

    def load_features(self, key: str) -> Dict[str, Any]:
        path = self._path(f"{key}_features")
        return load(path)

    def features_exist(self, key: str) -> bool:
        return os.path.exists(self._path(f"{key}_features"))

