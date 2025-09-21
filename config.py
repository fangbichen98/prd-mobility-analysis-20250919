"""
Configuration for the Shenzhen OD dynamic embedding project.

This module centralizes file paths and key parameters used across the
data loading, feature engineering and subsequent modeling stages.

Notes
- Edge weight column is set to `cu_freq` per user requirement.
- All snapshots will be forced to share a unified node set derived from
  all edges and (optionally) grid metadata.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# Project root inferred from this file's location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Paths:
    data_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "data"))
    mobility_edges_dir: str = field(init=False)
    cache_dir: str = field(init=False)
    # Default to PRD metadata for current task; can be overridden at runtime if needed.
    # Note: analysis scripts read CONFIG.paths.grid_metadata_csv for maps/exports.
    grid_metadata_csv: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "data", "PRD_grid_metadata.csv"))

    def __post_init__(self) -> None:
        self.mobility_edges_dir = os.path.join(self.data_dir, "mobility_edges")
        self.cache_dir = os.path.join(self.data_dir, "cache")


@dataclass
class LoaderParams:
    # Column names in mobility edge CSVs
    date_col: str = "date_dt"
    hour_col: str = "time_"
    src_col: str = "o_grid"
    dst_col: str = "d_grid"
    weight_col: str = "cu_freq"  # per user confirmation

    # Whether to include all nodes from grid metadata in the unified node set
    # For PRD scale, avoid inflating the node set with unused grid ids.
    include_metadata_nodes: bool = False

    # Optional performance/robustness knobs
    # PRD daily CSVs are large (~600MB each); enable chunked reads by default to reduce peak memory.
    use_chunks: bool = True
    chunk_rows: int = 2_000_000
    enforce_int_node_ids: bool = True
    # Limit nodes to top-N by total (in+out) flow for tractability on PRD
    top_n_nodes_by_flow: int = 30000


@dataclass
class CacheParams:
    enable_cache: bool = True
    graphs_key: str = "graphs_v1"
    feats_key: str = "features_v1"


@dataclass
class FeatureParams:
    # Add or remove features as needed; names are descriptive only
    compute_pagerank: bool = True
    pagerank_alpha: float = 0.85
    epsilon_smoothing: float = 1.0  # for weekday/weekend ratio smoothing


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    loader: LoaderParams = field(default_factory=LoaderParams)
    cache: CacheParams = field(default_factory=CacheParams)
    feats: FeatureParams = field(default_factory=FeatureParams)


# Convenience singleton used by scripts
CONFIG = Config()
