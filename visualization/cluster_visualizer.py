"""
Cluster visualizations for pattern curves and distribution.

Generates line charts for 24-hour pattern signatures and pie/bar charts
for pattern proportions. Figures are saved under a provided figure dir.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class ClusterVisualizer:
    def __init__(self, figure_dir: str, pattern_names: Optional[Dict[str, str]] = None) -> None:
        self.figure_dir = figure_dir
        ensure_dir(self.figure_dir)
        self.pattern_names = pattern_names or {}

    def _name(self, k: int) -> str:
        return self.pattern_names.get(str(k), str(k))

    def plot_pattern_signature_curves(self, hourly24_csv: str, names_json: Optional[str] = None, metric: str = "outflow") -> str:
        """Plot 24h curves per pattern from CSV produced by ReportGenerator.

        metric: one of 'inflow' or 'outflow'.
        """
        if names_json and os.path.exists(names_json):
            with open(names_json, "r", encoding="utf-8") as f:
                self.pattern_names.update(json.load(f))

        df = pd.read_csv(hourly24_csv)
        if metric not in df.columns:
            raise ValueError(f"metric '{metric}' not found in {hourly24_csv}")

        K = int(df["pattern"].max()) + 1 if "pattern" in df.columns else None

        fig, ax = plt.subplots(figsize=(9, 5))
        for k, g in df.groupby("pattern"):
            g = g.sort_values("hour")
            ax.plot(g["hour"].to_numpy(), g[metric].to_numpy(), marker="o", label=self._name(int(k)))
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour")
        ax.set_ylabel(metric)
        ax.set_title(f"Pattern Signature Curves ({metric})")
        ax.legend(ncol=2, fontsize=8)
        out = os.path.join(self.figure_dir, f"pattern_signature_curves_{metric}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_global_distribution_pie(self, distribution_csv: Optional[str] = None, U_global: Optional[np.ndarray] = None) -> str:
        """Plot distribution of patterns as pie (<=10) or bar (>10).

        The function accepts either a CSV produced by ReportGenerator
        (global_distribution.csv) or a raw U_global to compute counts.
        """
        if distribution_csv:
            df = pd.read_csv(distribution_csv)
            labels = [self._name(int(k)) for k in df["pattern"].to_list()]
            counts = df["count"].to_numpy()
        elif U_global is not None:
            arg = U_global.argmax(axis=1)
            K = U_global.shape[1]
            counts = np.array([(arg == k).sum() for k in range(K)], dtype=int)
            labels = [self._name(k) for k in range(K)]
        else:
            raise ValueError("Either distribution_csv or U_global must be provided")

        K = len(labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        if K <= 10:
            ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False)
            ax.set_title("Global Pattern Distribution (pie)")
            out = os.path.join(self.figure_dir, "global_distribution_pie.png")
        else:
            y = np.arange(K)
            ax.barh(y, counts)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("count")
            ax.set_title("Global Pattern Distribution (bar)")
            out = os.path.join(self.figure_dir, "global_distribution_bar.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

