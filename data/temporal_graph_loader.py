"""
Temporal graph loader that builds a time-sorted list of networkx DiGraph
snapshots from hourly OD mobility CSVs. All snapshots share a unified
node set derived from edges (and optionally from metadata).
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import networkx as nx


class TemporalGraphLoader:
    def __init__(
        self,
        mobility_edges_dir: str,
        date_col: str = "date_dt",
        hour_col: str = "time_",
        src_col: str = "o_grid",
        dst_col: str = "d_grid",
        weight_col: str = "cu_freq",
        include_metadata_nodes: bool = True,
        grid_metadata_csv: Optional[str] = None,
        use_chunks: bool = False,
        chunk_rows: int = 2_000_000,
        enforce_int_node_ids: bool = True,
        top_n_nodes_by_flow: int | None = 10000,
    ) -> None:
        self.mobility_edges_dir = mobility_edges_dir
        self.date_col = date_col
        self.hour_col = hour_col
        self.src_col = src_col
        self.dst_col = dst_col
        self.weight_col = weight_col
        self.include_metadata_nodes = include_metadata_nodes
        self.grid_metadata_csv = grid_metadata_csv
        self.use_chunks = use_chunks
        self.chunk_rows = chunk_rows
        self.enforce_int_node_ids = enforce_int_node_ids
        # Keep only the top-N most active nodes (by total in+out flow) if set.
        # This is critical to make PRD-scale runs tractable for DySAT.
        self.top_n_nodes_by_flow = top_n_nodes_by_flow

    def _list_csvs(self) -> List[str]:
        files = [
            os.path.join(self.mobility_edges_dir, f)
            for f in os.listdir(self.mobility_edges_dir)
            if f.lower().endswith(".csv")
        ]
        files.sort()
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.mobility_edges_dir}")
        return files

    def _read_metadata_nodes(self) -> List[int]:
        if not (self.include_metadata_nodes and self.grid_metadata_csv and os.path.exists(self.grid_metadata_csv)):
            return []
        df = pd.read_csv(self.grid_metadata_csv, usecols=["grid_id"])
        nodes = df["grid_id"].dropna().astype(int).tolist()
        return nodes

    def _normalize_node_ids(self, s: pd.Series) -> pd.Series:
        if self.enforce_int_node_ids:
            # Coerce to int safely
            return pd.to_numeric(s, errors="coerce").dropna().astype(int)
        return s

    def _read_edges_concat(self, files: List[str]) -> pd.DataFrame:
        cols = [self.date_col, self.hour_col, self.src_col, self.dst_col, self.weight_col]
        if not self.use_chunks:
            dfs = [pd.read_csv(fp, usecols=cols) for fp in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            parts = []
            for fp in files:
                for chunk in pd.read_csv(fp, usecols=cols, chunksize=self.chunk_rows):
                    parts.append(chunk)
            df = pd.concat(parts, ignore_index=True)

        # Normalize node id types
        df[self.src_col] = self._normalize_node_ids(df[self.src_col])
        df[self.dst_col] = self._normalize_node_ids(df[self.dst_col])
        # Drop any rows with NaN due to coercion
        df = df.dropna(subset=[self.src_col, self.dst_col])
        df[self.src_col] = df[self.src_col].astype(int)
        df[self.dst_col] = df[self.dst_col].astype(int)
        return df

    def _build_time_keys(self, df: pd.DataFrame) -> List[str]:
        # Time key format: YYYYMMDD-HH zero-padded hour
        tk = (df[self.date_col].astype(int).astype(str) + "-" + df[self.hour_col].astype(int).astype(str).str.zfill(2))
        return tk.tolist()

    def _sorted_unique_time_keys(self, df: pd.DataFrame) -> List[str]:
        df = df.copy()
        df["__time_key__"] = (
            df[self.date_col].astype(int).astype(str)
            + "-"
            + df[self.hour_col].astype(int).astype(str).str.zfill(2)
        )
        keys = sorted(df["__time_key__"].unique())
        return keys

    def load(self) -> Tuple[List[nx.DiGraph], List[int], List[str]]:
        files = self._list_csvs()
        df = self._read_edges_concat(files)

        # Optional pruning to top-N by flow to control graph size
        if self.top_n_nodes_by_flow is not None and int(self.top_n_nodes_by_flow) > 0:
            out_sum = df.groupby(self.src_col, as_index=False)[self.weight_col].sum().rename(columns={self.src_col: 'node', self.weight_col: 'out'})
            in_sum = df.groupby(self.dst_col, as_index=False)[self.weight_col].sum().rename(columns={self.dst_col: 'node', self.weight_col: 'in'})
            deg = out_sum.merge(in_sum, on='node', how='outer').fillna(0.0)
            deg['tot'] = deg['out'].astype(float) + deg['in'].astype(float)
            keep = set(deg.sort_values('tot', ascending=False)['node'].astype(int).head(int(self.top_n_nodes_by_flow)).tolist())
            df = df[df[self.src_col].isin(keep) & df[self.dst_col].isin(keep)].reset_index(drop=True)

        # Determine unified node set from edges (+ optional metadata)
        edge_nodes = pd.unique(pd.concat([df[self.src_col], df[self.dst_col]], ignore_index=True)).astype(int)
        nodes = set(edge_nodes.tolist())
        nodes.update(self._read_metadata_nodes())
        node_list = sorted(nodes)

        # Build per-time graphs
        time_keys = self._sorted_unique_time_keys(df)
        graphs: List[nx.DiGraph] = []

        # Group by (date, hour) to avoid string comparisons inside the loop
        grp = df.groupby([self.date_col, self.hour_col], sort=True, as_index=False)
        # Build a mapping for quick look-up from tuple to time_key string
        def tk_of(dt: int, hr: int) -> str:
            return f"{int(dt)}-{int(hr):02d}"

        groups: Dict[Tuple[int, int], pd.DataFrame] = { (int(k[0]), int(k[1])): g for k, g in grp }

        for tk in time_keys:
            date_s, hour_s = tk.split("-")
            key = (int(date_s), int(hour_s))
            gdf = groups.get(key)
            G = nx.DiGraph()
            G.add_nodes_from(node_list)
            if gdf is not None and not gdf.empty:
                # Aggregate possible duplicate edges (same o,d within same hour)
                agg = gdf.groupby([self.src_col, self.dst_col], as_index=False)[self.weight_col].sum()
                edges = list(zip(agg[self.src_col].tolist(), agg[self.dst_col].tolist(), agg[self.weight_col].tolist()))
                for u, v, w in edges:
                    G.add_edge(int(u), int(v), weight=float(w))
            graphs.append(G)

        return graphs, node_list, time_keys

