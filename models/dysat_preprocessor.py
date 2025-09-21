"""
Preprocessing utilities to convert NetworkX graphs and node features into
PyTorch tensors suitable for the DySAT model.

Responsibilities
- Build normalized adjacency (A+I) with symmetric normalization.
- Convert to torch.sparse_coo tensors for efficient propagation.
- Provide static node feature matrix X (standardized) as torch.float.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """Symmetrically normalize adjacency: D^(-1/2) * (A + I) * D^(-1/2)."""
    adj = adj.tocsr()
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format="csr")
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    d_inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def to_torch_sparse(mat: sp.spmatrix) -> torch.Tensor:
    mat = mat.tocoo()
    indices = np.vstack([mat.row, mat.col])
    indices = torch.from_numpy(indices).long()
    values = torch.from_numpy(mat.data).float()
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class DySATPreprocessor:
    def __init__(self) -> None:
        pass

    def build_adjs(
        self,
        graphs: Sequence[nx.DiGraph],
        nodes: Sequence[int],
        use_weight: bool = True,
    ) -> List[torch.Tensor]:
        """Create normalized sparse adjacency tensors per snapshot.

        Edges use 'weight' attribute when present; otherwise default 1.0.
        """
        n = len(nodes)
        node_index = {nid: i for i, nid in enumerate(nodes)}
        adjs: List[torch.Tensor] = []
        for G in graphs:
            # Build weighted directed adjacency; then symmetrize for propagation
            rows = []
            cols = []
            data = []
            for u, v, d in G.edges(data=True):
                w = float(d.get("weight", 1.0)) if use_weight else 1.0
                rows.append(node_index[u])
                cols.append(node_index[v])
                data.append(w)
            # Convert to sparse and symmetrize (A + A^T)
            A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
            A = (A + A.T).tocsr()
            A_norm = normalize_adj(A).tocsr()
            adjs.append(to_torch_sparse(A_norm))
        return adjs

    def build_features(self, feats: dict) -> torch.Tensor:
        """Return standardized node features as torch.tensor (N, F)."""
        X = feats["X"]
        X = np.asarray(X, dtype=np.float32)
        return torch.from_numpy(X)

