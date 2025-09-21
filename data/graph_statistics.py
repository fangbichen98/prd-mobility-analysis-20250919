"""
Compute static, aggregated node features across a sequence of directed graphs.

Features (per node)
- total_inflow, total_outflow, total_flow
- active_hours_in, active_hours_out
- weekday_weekend_ratio_in, weekday_weekend_ratio_out
- pagerank (on aggregated weighted digraph) [optional]

Output is standardized (z-score) together with original (raw) values.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GraphStatistics:
    def __init__(
        self,
        pagerank: bool = True,
        pagerank_alpha: float = 0.85,
        epsilon_smoothing: float = 1.0,
    ) -> None:
        self.pagerank = pagerank
        self.pagerank_alpha = pagerank_alpha
        self.eps = epsilon_smoothing

    @staticmethod
    def _parse_weekday_from_time_keys(time_keys: Sequence[str]) -> List[bool]:
        # time_key format: YYYYMMDD-HH
        weekend = []
        for tk in time_keys:
            date_s, _ = tk.split("-")
            y, m, d = int(date_s[:4]), int(date_s[4:6]), int(date_s[6:8])
            wd = dt.date(y, m, d).weekday()  # 0=Mon ... 6=Sun
            weekend.append(wd >= 5)
        return weekend

    def compute_static_node_features(
        self,
        graphs: Sequence[nx.DiGraph],
        nodes: Sequence[int],
        time_keys: Sequence[str],
    ) -> Dict[str, object]:
        n = len(nodes)
        T = len(graphs)
        node_index = {nid: i for i, nid in enumerate(nodes)}

        inflow = np.zeros(n, dtype=float)
        outflow = np.zeros(n, dtype=float)
        active_in = np.zeros(n, dtype=float)
        active_out = np.zeros(n, dtype=float)

        # Keep aggregated edge weights for building a summary graph later
        agg_edges: Dict[Tuple[int, int], float] = {}

        weekend_flags = np.array(self._parse_weekday_from_time_keys(time_keys), dtype=bool)
        weekend_in = np.zeros(n, dtype=float)
        weekend_out = np.zeros(n, dtype=float)
        weekday_in = np.zeros(n, dtype=float)
        weekday_out = np.zeros(n, dtype=float)

        for t, G in enumerate(graphs):
            # For activity flags, check if node has any non-zero in/out at this snapshot
            # Accumulate flows
            # Outgoing
            has_out = np.zeros(n, dtype=bool)
            has_in = np.zeros(n, dtype=bool)

            for u, v, data in G.edges(data=True):
                w = float(data.get("weight", 1.0))
                ui = node_index[u]
                vi = node_index[v]
                outflow[ui] += w
                inflow[vi] += w
                has_out[ui] = True
                has_in[vi] = True
                key = (u, v)
                agg_edges[key] = agg_edges.get(key, 0.0) + w

            active_out += has_out.astype(float)
            active_in += has_in.astype(float)

            if weekend_flags[t]:
                weekend_out += outflow * 0  # placeholders for vectorized shape
                weekend_in += inflow * 0
                # Accumulate per-snapshot flows again to the right bins
                for u, v, data in G.edges(data=True):
                    w = float(data.get("weight", 1.0))
                    weekend_out[node_index[u]] += w
                    weekend_in[node_index[v]] += w
            else:
                for u, v, data in G.edges(data=True):
                    w = float(data.get("weight", 1.0))
                    weekday_out[node_index[u]] += w
                    weekday_in[node_index[v]] += w

        # Build aggregated graph for centrality
        aggG = nx.DiGraph()
        aggG.add_nodes_from(nodes)
        for (u, v), w in agg_edges.items():
            if w != 0:
                aggG.add_edge(u, v, weight=w)

        features = {}
        features["total_inflow"] = inflow
        features["total_outflow"] = outflow
        features["total_flow"] = inflow + outflow
        features["active_hours_in"] = active_in
        features["active_hours_out"] = active_out
        # Ratios with smoothing to avoid division by zero
        wkr = (weekday_in + self.eps) / (weekend_in + self.eps)
        wkr_out = (weekday_out + self.eps) / (weekend_out + self.eps)
        features["weekday_weekend_ratio_in"] = wkr
        features["weekday_weekend_ratio_out"] = wkr_out

        if self.pagerank:
            # networkx pagerank can handle dangling nodes; use weight attr
            pr = nx.pagerank(aggG, alpha=self.pagerank_alpha, weight="weight")
            pr_vec = np.array([pr.get(nid, 0.0) for nid in nodes], dtype=float)
            features["pagerank"] = pr_vec

        # Assemble matrix
        feat_names = list(features.keys())
        X = np.vstack([features[k] for k in feat_names]).T  # shape (N, D)

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        raw_df = pd.DataFrame(X, index=nodes, columns=feat_names)

        return {
            "node_ids": list(nodes),
            "feature_names": feat_names,
            "X": X_std,
            "raw_df": raw_df,
            "scaler_mean_": scaler.mean_.tolist(),
            "scaler_scale_": scaler.scale_.tolist(),
        }

