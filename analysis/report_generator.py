"""
Report generation utilities (Phase 4) to analyze and interpret global patterns.

This module consumes outputs from Phase 3 (U_final, U_global) together with
the temporal graphs (or OD edges) to produce:
- Pattern profiles: For each global pattern, a typical 24-hour inflow/outflow
  curve using U_final as weights.
- TIE stability metrics: Per-node measures of temporal stability based on
  how rapidly memberships change.
- Global distribution: Counts and proportions of nodes per dominant pattern.

Exports CSV/JSON files under outputs/experiment_{tag}/report/.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class ReportPaths:
    out_dir: str  # outputs/experiment_{tag}/report/


class ReportGenerator:
    """Generate reports and metrics from clustering memberships and graphs.

    Typical usage:
        rg = ReportGenerator(ReportPaths(out_dir))
        inflow, outflow = rg.compute_hourly_flows_from_graphs(graphs, nodes)
        profiles = rg.generate_pattern_profiles(U_final, inflow, outflow, time_keys)
        tie = rg.calculate_tie_stability(U_final)
        dist = rg.analyze_global_pattern_distribution(U_global)
    """

    def __init__(self, paths: ReportPaths) -> None:
        self.paths = paths
        ensure_dir(self.paths.out_dir)

    # ---------- Data helpers ----------
    @staticmethod
    def compute_hourly_flows_from_graphs(
        graphs: Sequence[nx.DiGraph], nodes: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-node inflow/outflow per hour from graph snapshots.

        Returns
        - inflow: (N, T)
        - outflow: (N, T)
        """
        N = len(nodes)
        T = len(graphs)
        node_index = {nid: i for i, nid in enumerate(nodes)}
        inflow = np.zeros((N, T), dtype=np.float64)
        outflow = np.zeros((N, T), dtype=np.float64)
        for t, G in enumerate(graphs):
            for u, v, d in G.edges(data=True):
                w = float(d.get("weight", 1.0))
                outflow[node_index[u], t] += w
                inflow[node_index[v], t] += w
        return inflow, outflow

    # ---------- Pattern profiles ----------
    def generate_pattern_profiles(
        self,
        U_final: np.ndarray,
        inflow: np.ndarray,
        outflow: np.ndarray,
        time_keys: Sequence[str],
    ) -> Dict[str, object]:
        """Compute weighted average curves per pattern using U_final.

        For pattern k and snapshot t:
            curve_k[t] = sum_i U_final[i,t,k] * flow[i,t] / sum_i U_final[i,t,k]

        Additionally, collapse to typical 24-hour curves by averaging
        across days for each hour-of-day.
        Exports:
        - pattern_profiles_timekey.csv
        - pattern_profiles_hourly24.csv
        - pattern_names.json (heuristic naming based on peaks)
        """
        N, T, K = U_final.shape
        assert inflow.shape == (N, T) and outflow.shape == (N, T)

        eps = 1e-8
        wsum = U_final.sum(axis=0) + eps  # (T, K)
        inflow_w = (U_final.transpose(1, 2, 0) @ inflow).transpose(0, 2, 1)  # (T,K,N)@ (N,T)-> broadcast is tricky
        # Re-implement more explicitly to avoid confusion
        inflow_wtk = np.zeros((T, K), dtype=np.float64)
        outflow_wtk = np.zeros((T, K), dtype=np.float64)
        for t in range(T):
            W_t = U_final[:, t, :]  # (N,K)
            inflow_wtk[t, :] = (W_t.T @ inflow[:, t]).astype(np.float64)
            outflow_wtk[t, :] = (W_t.T @ outflow[:, t]).astype(np.float64)
        inflow_avg = inflow_wtk / wsum  # (T,K)
        outflow_avg = outflow_wtk / wsum

        # Export per time_key
        df_time = []
        for t, tk in enumerate(time_keys):
            date_s, hour_s = tk.split("-")
            hour = int(hour_s)
            for k in range(K):
                df_time.append({
                    "pattern": k,
                    "time_key": tk,
                    "date": int(date_s),
                    "hour": hour,
                    "inflow": float(inflow_avg[t, k]),
                    "outflow": float(outflow_avg[t, k]),
                })
        df_time = pd.DataFrame(df_time)
        df_time.to_csv(os.path.join(self.paths.out_dir, "pattern_profiles_timekey.csv"), index=False)

        # Collapse to 24-hour averages
        df_time["hour"] = df_time["hour"].astype(int)
        df24 = df_time.groupby(["pattern", "hour"], as_index=False)[["inflow", "outflow"]].mean()
        df24.to_csv(os.path.join(self.paths.out_dir, "pattern_profiles_hourly24.csv"), index=False)

        # Heuristic naming based on peak structure on 24h curves (use outflow)
        pattern_names = {}
        for k in range(K):
            curve = df24[df24["pattern"] == k].sort_values("hour")["outflow"].to_numpy()
            name = self._name_pattern_from_curve(curve)
            pattern_names[str(k)] = name
        with open(os.path.join(self.paths.out_dir, "pattern_names.json"), "w", encoding="utf-8") as f:
            json.dump(pattern_names, f, ensure_ascii=False, indent=2)

        return {
            "timekey_curves_csv": os.path.join(self.paths.out_dir, "pattern_profiles_timekey.csv"),
            "hourly24_curves_csv": os.path.join(self.paths.out_dir, "pattern_profiles_hourly24.csv"),
            "pattern_names_json": os.path.join(self.paths.out_dir, "pattern_names.json"),
        }

    @staticmethod
    def _name_pattern_from_curve(curve: np.ndarray) -> str:
        """Assign a simple name based on peaks/shape for a 24h curve.

        Heuristics (on normalized curve):
        - Two clear peaks at ~8-10 and ~17-20 -> "commute_double_peak"
        - Peak after 22 -> "night_active"
        - Peak mid-day 11-15 -> "midday_peak"
        - Otherwise -> "flat_or_single_peak"
        """
        if curve.size != 24:
            # Fallback to generic naming
            return "pattern"
        x = curve.astype(float)
        if x.max() > 0:
            x = x / (x.max() + 1e-9)
        # Locate rough peaks
        peaks = [h for h in range(1, 23) if x[h] > x[h - 1] and x[h] > x[h + 1] and x[h] > 0.4]
        if any(8 <= h <= 10 for h in peaks) and any(17 <= h <= 20 for h in peaks):
            return "commute_double_peak"
        if any(h >= 22 for h in peaks):
            return "night_active"
        if any(11 <= h <= 15 for h in peaks):
            return "midday_peak"
        return "flat_or_single_peak"

    # ---------- TIE stability ----------
    def calculate_tie_stability(self, U_final: np.ndarray) -> pd.DataFrame:
        """Compute temporal stability indicators per node.

        Metrics
        - tie_switch_rate: fraction of time steps where argmax membership
          changes (0=stable, 1=changes every step).
        - tie_total_variation: mean total variation distance between
          consecutive membership vectors: mean_t 0.5*||u_t - u_{t-1}||_1.
        Returns a DataFrame with columns: node_idx, tie_switch_rate, tie_total_variation
        and writes tie_metrics.csv in report dir.
        """
        N, T, K = U_final.shape
        arg = U_final.argmax(axis=2)  # (N,T)
        switches = (arg[:, 1:] != arg[:, :-1]).sum(axis=1)
        switch_rate = switches / max(1, (T - 1))

        # Total variation over probabilities
        tv = np.zeros(N, dtype=np.float64)
        for i in range(N):
            diffs = np.abs(U_final[i, 1:, :] - U_final[i, :-1, :])  # (T-1,K)
            tv[i] = 0.5 * diffs.sum(axis=1).mean() if T > 1 else 0.0

        df = pd.DataFrame({
            "node_idx": np.arange(N, dtype=int),
            "tie_switch_rate": switch_rate,
            "tie_total_variation": tv,
        })
        df.to_csv(os.path.join(self.paths.out_dir, "tie_metrics.csv"), index=False)
        return df

    # ---------- Global distribution ----------
    def analyze_global_pattern_distribution(self, U_global: np.ndarray) -> pd.DataFrame:
        """Summarize global memberships across nodes.

        Returns a DataFrame with per-pattern dominant counts and shares,
        plus mean membership (soft proportion). Writes global_distribution.csv.
        """
        N, K = U_global.shape
        arg = U_global.argmax(axis=1)
        rows = []
        for k in range(K):
            count = int((arg == k).sum())
            share = count / max(1, N)
            soft_mean = float(U_global[:, k].mean())
            rows.append({"pattern": k, "count": count, "share": share, "soft_mean": soft_mean})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.paths.out_dir, "global_distribution.csv"), index=False)
        return df

    # ---------- Node summary (grid_id, lon, lat, cluster, metrics) ----------
    def export_node_summary(
        self,
        U_global: np.ndarray,
        node_ids: Sequence[int],
        grid_metadata_csv: str,
        tie_df: pd.DataFrame | None = None,
    ) -> str:
        """Export a node-level CSV with coordinates and cluster info.

        Columns: grid_id, lon, lat, dominant_pattern, plus optional
        tie metrics merged by node_idx.
        """
        import pandas as pd  # local import for robustness

        assert U_global.shape[0] == len(node_ids)
        dom = U_global.argmax(axis=1)
        meta = pd.read_csv(grid_metadata_csv, usecols=["grid_id", "lon", "lat"]).set_index("grid_id")
        df = pd.DataFrame({
            "grid_id": node_ids,
            "dominant_pattern": dom,
        })
        # attach lon/lat if available
        df = df.join(meta, on="grid_id")

        # merge tie metrics if provided
        if tie_df is not None and "node_idx" in tie_df.columns:
            tie_cols = [c for c in tie_df.columns if c != "node_idx"]
            tie_merge = tie_df.copy()
            tie_merge = tie_merge.rename(columns={"node_idx": "_node_idx"})
            tie_merge.index = tie_merge["_node_idx"].to_numpy()
            tie_merge = tie_merge.drop(columns=["_node_idx"])  # keep only metrics
            # bring metrics in the same order as node_ids
            metrics = tie_merge.reindex(range(len(node_ids)))
            metrics.index = df.index
            df = pd.concat([df, metrics], axis=1)

        out = os.path.join(self.paths.out_dir, "node_summary.csv")
        df.to_csv(out, index=False)
        return out
