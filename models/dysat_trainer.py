"""
Trainer for the simplified DySAT model using self-supervised link
reconstruction across time snapshots.

Loss
- For each snapshot t, sample positive edges from the graph and the same
  number of negative node pairs uniformly at random. Predict link logits
  via dot product of per-time embeddings and optimize BCE loss.

Saves
- Model checkpoints under models/checkpoints/
"""
from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dysat import DySAT, DySATConfig


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_neg_ratio: float = 1.0  # negatives per positive
    max_pos_per_snap: int = 50000  # cap positives per snapshot to limit compute
    log_every: int = 1
    device: str = "cpu"


def sample_edges(G: nx.DiGraph, max_count: int) -> List[Tuple[int, int]]:
    edges = list(G.edges())
    if len(edges) <= max_count:
        return edges
    return random.sample(edges, max_count)


def sample_non_edges(n: int, pos_edges: Sequence[Tuple[int, int]], count: int) -> List[Tuple[int, int]]:
    pos_set = set(pos_edges)
    pairs = []
    tries = 0
    while len(pairs) < count and tries < count * 10:
        u = random.randrange(0, n)
        v = random.randrange(0, n)
        tries += 1
        if u == v:
            continue
        if (u, v) in pos_set:
            continue
        pairs.append((u, v))
    if len(pairs) < count:
        # Fallback to fill remaining with random even if duplicates
        for _ in range(count - len(pairs)):
            u = random.randrange(0, n)
            v = random.randrange(0, n)
            pairs.append((u, v))
    return pairs


class DySATTrainer:
    def __init__(
        self,
        model: DySAT,
        adj_list: List[torch.Tensor],
        x: torch.Tensor,
        graphs_nx: Sequence[nx.DiGraph],
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.adj_list = [a.coalesce().to(device) for a in adj_list]
        self.x = x.to(device)
        self.graphs_nx = graphs_nx
        self.device = device

    def _snapshot_embeddings(self) -> torch.Tensor:
        """Compute H (N, T, D)."""
        self.model.eval()
        with torch.no_grad():
            H = self.model(self.adj_list, self.x)
        return H

    def train(self, cfg: TrainConfig, nodes: Sequence[int]) -> None:
        opt = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        bce = nn.BCEWithLogitsLoss()
        n = len(nodes)
        node_to_idx = {nid: i for i, nid in enumerate(nodes)}

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            opt.zero_grad()

            H = self.model(self.adj_list, self.x)  # (N, T, D)
            N, T, D = H.shape
            loss_total = 0.0

            # Per-snapshot link reconstruction
            for t, G in enumerate(self.graphs_nx):
                pos_edges_raw = sample_edges(G, cfg.max_pos_per_snap)
                # Map raw node ids to contiguous indices aligned with embeddings
                pos_edges = []
                for (u, v) in pos_edges_raw:
                    ui = node_to_idx.get(u)
                    vi = node_to_idx.get(v)
                    if ui is None or vi is None:
                        continue
                    pos_edges.append((ui, vi))
                num_neg = int(len(pos_edges) * cfg.batch_neg_ratio)
                neg_edges = sample_non_edges(n, pos_edges, num_neg)

                z = H[:, t, :]  # (N, D)
                if len(pos_edges) == 0:
                    continue
                u_pos = torch.tensor([e[0] for e in pos_edges], device=self.device)
                v_pos = torch.tensor([e[1] for e in pos_edges], device=self.device)
                u_neg = torch.tensor([e[0] for e in neg_edges], device=self.device)
                v_neg = torch.tensor([e[1] for e in neg_edges], device=self.device)

                pos_logit = (z[u_pos] * z[v_pos]).sum(dim=1)
                neg_logit = (z[u_neg] * z[v_neg]).sum(dim=1)
                y_pos = torch.ones_like(pos_logit)
                y_neg = torch.zeros_like(neg_logit)
                loss = bce(pos_logit, y_pos) + bce(neg_logit, y_neg)
                loss_total = loss_total + loss

            loss_total.backward()
            opt.step()

            if epoch % cfg.log_every == 0:
                print(f"[DySATTrainer] epoch={epoch} loss={loss_total.item():.4f}")

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device), strict=strict)

    def get_embeddings(self) -> torch.Tensor:
        return self._snapshot_embeddings()
