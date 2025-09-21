"""
Simplified DySAT-like model implemented in PyTorch.

Structure
- Structural encoder (per snapshot): one or two graph propagation layers
  using normalized adjacency (D^(-1/2) (A+I) D^(-1/2)). This approximates
  the structural attention in the original DySAT with a lightweight
  propagation + MLP, suitable for environments without graph libs.
- Temporal encoder: multi-head self-attention over the time axis applied
  independently for each node, producing temporally-contextualized node
  embeddings per snapshot.

Output
- H: tensor of shape (N, T, D_out) where N is the number of nodes,
  T the number of snapshots, and D_out the final embedding size.

Note
- This is a pragmatic implementation focusing on end-to-end usability.
  It captures the core idea (structure + temporal attention) with
  reasonable computational complexity and minimal dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_mm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multiply sparse COO adjacency by dense matrix x.

    Expects adj as torch.sparse_coo_tensor and x as dense (N, F).
    """
    return torch.sparse.mm(adj, x)


class GraphProp(nn.Module):
    """Single graph propagation layer using normalized adjacency.

    y = ReLU( A_hat @ x @ W + b ) followed by dropout.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xw = self.lin(x)
        ax = sparse_mm(adj_norm, xw)
        h = F.relu(ax)
        h = self.dropout(h)
        return h


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over the time axis for each node.

    Applies attention in mini-batches over the node dimension to control GPU memory.

    Input:  Z of shape (N, T, D)
    Output: H of shape (N, T, D)
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0, chunk_size: int = 1024) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Number of nodes processed per attention call; tune to fit GPU memory
        self.chunk_size = int(max(1, chunk_size))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, T, D)
        N = z.shape[0]
        outs = []
        # Process nodes in chunks to reduce attention memory footprint
        for i in range(0, N, self.chunk_size):
            zi = z[i:i + self.chunk_size]
            attn_out, _ = self.mha(zi, zi, zi, need_weights=False)
            h1 = self.norm1(zi + self.dropout(attn_out))
            ff_out = self.ff(h1)
            h2 = self.norm2(h1 + self.dropout(ff_out))
            outs.append(h2)
        return torch.cat(outs, dim=0)


@dataclass
class DySATConfig:
    in_dim: int
    struct_hidden: int = 64
    out_dim: int = 64
    struct_layers: int = 1
    temporal_heads: int = 4
    dropout: float = 0.1


class DySAT(nn.Module):
    """Simplified DySAT-like encoder.

    Forward signature
    - adj_list: list of sparse normalized adjacencies, length T.
    - x: static node features, shape (N, F_in).
    Returns embeddings H of shape (N, T, D_out).
    """
    def __init__(self, cfg: DySATConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Structural encoder (shared weights across time snapshots)
        layers: List[nn.Module] = []
        in_dim = cfg.in_dim
        for i in range(cfg.struct_layers):
            out_dim = cfg.struct_hidden if i < cfg.struct_layers - 1 else cfg.out_dim
            layers.append(GraphProp(in_dim, out_dim, dropout=cfg.dropout))
            in_dim = out_dim
        self.struct = nn.ModuleList(layers)

        # Temporal encoder operating on per-node time sequences
        # Chunked temporal attention keeps memory bounded for large N
        self.temporal = TemporalSelfAttention(dim=cfg.out_dim, num_heads=cfg.temporal_heads, dropout=cfg.dropout, chunk_size=1024)

    def structural_encode(self, adj_list: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Apply structural encoder independently per time snapshot.

        Returns Z of shape (N, T, D_struct) where D_struct = cfg.out_dim.
        """
        zs = []
        for adj in adj_list:
            h = x
            for layer in self.struct:
                h = layer(adj, h)
            zs.append(h)
        # Stack along time dimension
        Z = torch.stack(zs, dim=1)  # (N, T, D)
        return Z

    def forward(self, adj_list: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        Z = self.structural_encode(adj_list, x)
        H = self.temporal(Z)
        return H  # (N, T, D_out)
