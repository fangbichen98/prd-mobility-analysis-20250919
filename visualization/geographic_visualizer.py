"""
Geographic visualizations for global/temporal patterns and node metrics.

This module renders simple scatter maps using lon/lat from grid metadata.
It avoids heavy GIS dependencies to remain portable. All figures are
saved under a provided figure directory.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class GeographicVisualizer:
    """Render maps for dominant patterns and spatial metrics.

    Assumes node IDs correspond to `grid_id` in the metadata CSV
    containing columns: grid_id, lon, lat.
    """

    def __init__(
        self,
        figure_dir: str,
        grid_metadata_csv: str,
        nodes: Sequence[int],
        time_keys: Optional[Sequence[str]] = None,
        pattern_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.figure_dir = figure_dir
        ensure_dir(self.figure_dir)
        self.nodes = list(nodes)
        self.time_keys = list(time_keys) if time_keys is not None else None
        self.pattern_names = pattern_names or {}

        meta = pd.read_csv(grid_metadata_csv, usecols=["grid_id", "lon", "lat"])
        self.meta = meta.set_index("grid_id")
        # Pre-compute coordinates aligned to nodes
        lons, lats, mask = self._aligned_coords()
        self._lons = lons
        self._lats = lats
        self._valid_mask = mask

    def _aligned_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lons = np.full(len(self.nodes), np.nan, dtype=float)
        lats = np.full(len(self.nodes), np.nan, dtype=float)
        for i, nid in enumerate(self.nodes):
            if nid in self.meta.index:
                row = self.meta.loc[nid]
                lons[i] = float(row["lon"])
                lats[i] = float(row["lat"])
        mask = ~np.isnan(lons) & ~np.isnan(lats)
        return lons, lats, mask

    def _discrete_cmap(self, K: int) -> ListedColormap:
        base = plt.get_cmap("tab20" if K > 10 else "tab10")
        colors = base.colors[:K]
        return ListedColormap(colors)

    def _set_true_aspect(self, ax) -> None:
        """Set axis aspect so that plotted lon/lat approximate true metric scale.

        On a Plate Carrée view, 1 degree of latitude ~ 111 km, and 1 degree of
        longitude ~ cos(phi) * 111 km where phi is the latitude. To achieve
        approximately true distances on the screen, set the y/x data aspect
        ratio to 1/cos(mean_lat).
        """
        valid_lats = self._lats[self._valid_mask]
        if valid_lats.size == 0 or np.isnan(valid_lats).all():
            return
        mean_lat = float(np.nanmean(valid_lats))
        ratio = 1.0 / max(1e-6, np.cos(np.deg2rad(mean_lat)))
        ax.set_aspect(ratio, adjustable="box")

    def plot_global_dominant_pattern_map(self, U_global: np.ndarray, fname: str = "global_dominant_pattern_map.png") -> str:
        """Plot dominant global pattern per node as a scatter map."""
        assert U_global.shape[0] == len(self.nodes)
        K = U_global.shape[1]
        dom = U_global.argmax(axis=1)

        lons = self._lons[self._valid_mask]
        lats = self._lats[self._valid_mask]
        dom_v = dom[self._valid_mask]

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(lons, lats, c=dom_v, s=6, cmap=self._discrete_cmap(K), linewidths=0, alpha=0.9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title("Global Dominant Pattern")
        cbar = plt.colorbar(sc, ax=ax, ticks=range(K))
        cbar.ax.set_yticklabels([self.pattern_names.get(k, str(k)) for k in range(K)])

        # Enforce true-ish aspect in lon/lat.
        self._set_true_aspect(ax)

        out = os.path.join(self.figure_dir, fname)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_hourly_dominant_pattern_map(self, U_final: np.ndarray, hour: int, fname: Optional[str] = None) -> str:
        """Plot dominant pattern at a specific hour-of-day aggregated across days.

        If time_keys are provided, picks all snapshots with the given hour
        and averages memberships over them. Otherwise, uses snapshot index
        `hour` directly if within bounds.
        """
        N, T, K = U_final.shape
        assert N == len(self.nodes)

        if self.time_keys is not None:
            hours = [int(tk.split("-")[1]) for tk in self.time_keys]
            idx = [t for t, h in enumerate(hours) if h == int(hour)]
            if not idx:
                # Fallback: use modulo across T
                idx = [h for h in range(T) if (h % 24) == int(hour % 24)]
            U_agg = U_final[:, idx, :].mean(axis=1) if idx else U_final.mean(axis=1)
        else:
            if 0 <= hour < T:
                U_agg = U_final[:, hour, :]
            else:
                U_agg = U_final.mean(axis=1)

        dom = U_agg.argmax(axis=1)
        lons = self._lons[self._valid_mask]
        lats = self._lats[self._valid_mask]
        dom_v = dom[self._valid_mask]

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(lons, lats, c=dom_v, s=6, cmap=self._discrete_cmap(K), linewidths=0, alpha=0.9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"Dominant Pattern at Hour {int(hour):02d}")
        cbar = plt.colorbar(sc, ax=ax, ticks=range(K))
        cbar.ax.set_yticklabels([self.pattern_names.get(k, str(k)) for k in range(K)])

        if fname is None:
            fname = f"hourly_dominant_pattern_map_hour{int(hour):02d}.png"
        # Enforce true-ish aspect in lon/lat.
        self._set_true_aspect(ax)
        out = os.path.join(self.figure_dir, fname)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_metric_map(self, node_metrics: pd.DataFrame, metric_col: str = "tie_total_variation", fname: Optional[str] = None) -> str:
        """Plot a continuous metric (per node) as a scatter heatmap."""
        if "node_idx" not in node_metrics.columns:
            raise ValueError("node_metrics must contain column 'node_idx'")
        if metric_col not in node_metrics.columns:
            raise ValueError(f"node_metrics missing metric column: {metric_col}")

        vals = np.full(len(self.nodes), np.nan, dtype=float)
        idx = node_metrics["node_idx"].to_numpy().astype(int)
        m = node_metrics[metric_col].to_numpy().astype(float)
        ok = (idx >= 0) & (idx < len(self.nodes))
        vals[idx[ok]] = m[ok]

        lons = self._lons[self._valid_mask]
        lats = self._lats[self._valid_mask]
        v = vals[self._valid_mask]

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(lons, lats, c=v, s=6, cmap="viridis", linewidths=0, alpha=0.9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(metric_col)
        plt.colorbar(sc, ax=ax)

        if fname is None:
            fname = f"metric_map_{metric_col}.png"
        # Enforce true-ish aspect in lon/lat.
        self._set_true_aspect(ax)
        out = os.path.join(self.figure_dir, fname)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out
