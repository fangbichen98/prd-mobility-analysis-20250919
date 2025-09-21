"""
Two-stage STGMM pattern recognizer.

Implements the workflow described in codex.md (阶段三):
1) For each node v, fit a local GaussianMixture on its temporal embeddings
   H_v (shape T x D_out) to extract a rhythm signature phi_v by
   concatenating parameters (means_, covariances_, weights_).
2) Fit a global GMM on the signature matrix Phi to discover K_global
   global patterns using BIC model selection over a candidate range.
   Produce node-level global memberships U_global (|V| x K_global).
3) Reconstruct temporal memberships by applying a K_global-component GMM
   on the instantaneous embeddings (|V|*T x D_out) and reshaping to
   U_final (|V| x T x K_global).

Notes
- Step (3) requires a GMM trained in the embedding space (D_out). Since
  step (2) trains on signature space (D_rhythm), we select K_global via
  signatures, and then fit a second GMM on the flattened embeddings with
  that fixed K, to compute U_final. We return this embedding-space GMM as
  the `global_gmm`.

Input
- H: np.ndarray or torch.Tensor of shape (N, T, D_out)

Output
- U_final: np.ndarray (N, T, K_global)
- U_global: np.ndarray (N, K_global)   [from signature-space GMM]
- global_gmm: sklearn.mixture.GaussianMixture (trained on embedding space)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is optional for this module
    torch = None


@dataclass
class TwoStageConfig:
    k_temporal: int = 4                   # local GMM components per node
    k_global_range: Sequence[int] = (4, 5, 6, 7, 8)
    covariance_type_local: str = "diag"   # for local rhythm GMMs
    covariance_type_global: str = "diag"  # for global GMMs
    n_init: int = 2
    max_iter: int = 200
    reg_covar: float = 1e-6
    random_state: int = 0
    n_jobs: int = 1                       # parallelism over nodes (stage 1)
    scale_signatures: bool = True         # standardize Phi before global GMM
    scale_embeddings: bool = True         # standardize H before temporal reconstruction


class TwoStagePatternRecognizer:
    def __init__(self, cfg: TwoStageConfig) -> None:
        self.cfg = cfg
        self._sig_scaler: StandardScaler | None = None
        self._emb_scaler: StandardScaler | None = None
        self._gmm_phi: GaussianMixture | None = None  # global GMM in signature space (for U_global)

    # ---------- Public API ----------
    def run(self, H) -> Tuple[np.ndarray, np.ndarray, GaussianMixture]:
        """Run the two-stage STGMM pipeline.

        Parameters
        - H: (N, T, D) numpy array or torch tensor.

        Returns
        - U_final: (N, T, K) temporal memberships (embedding-space GMM)
        - U_global: (N, K) global memberships (signature-space GMM)
        - global_gmm: GaussianMixture trained on embedding space (K comps)
        """
        H_np = self._to_numpy(H)
        Phi = self._extract_rhythm_signatures(H_np)
        U_global, gmm_phi, K = self._aggregate_global_patterns(Phi)
        U_final, gmm_embed = self._reconstruct_temporal_memberships(H_np, K)
        # Keep for introspection
        self._gmm_phi = gmm_phi
        return U_final, U_global, gmm_embed

    # ---------- Stage 1: Local rhythm signatures ----------
    def _extract_rhythm_signatures(self, H: np.ndarray) -> np.ndarray:
        """Fit per-node local GMMs on H_v (T x D) and build signature matrix.

        Signature for node v concatenates: means_, covariances_, weights_.
        For covariance_type='diag', phi_v length = k*(2*D) + k.
        """
        assert H.ndim == 3, "H must be (N, T, D)"
        N, T, D = H.shape
        k = max(1, min(self.cfg.k_temporal, T))

        from joblib import Parallel, delayed

        def fit_one(v: int) -> np.ndarray:
            X = H[v]  # (T, D)
            # Guard against degenerate data: add small noise if necessary
            X = np.asarray(X, dtype=np.float64)
            if np.allclose(X.std(axis=0, ddof=1), 0.0):
                X = X + 1e-6 * np.random.RandomState(self.cfg.random_state + v).normal(size=X.shape)
            gm = GaussianMixture(
                n_components=k,
                covariance_type=self.cfg.covariance_type_local,
                n_init=self.cfg.n_init,
                max_iter=self.cfg.max_iter,
                reg_covar=self.cfg.reg_covar,
                random_state=self.cfg.random_state,
            )
            gm.fit(X)
            means = gm.means_.reshape(-1)
            if self.cfg.covariance_type_local == "diag":
                covs = gm.covariances_.reshape(-1)
            else:
                # Flatten full covariances; may be large but preserves info
                covs = gm.covariances_.reshape(-1)
            weights = gm.weights_.reshape(-1)
            phi = np.concatenate([means, covs, weights])
            return phi.astype(np.float32)

        Phi = Parallel(n_jobs=self.cfg.n_jobs, prefer="threads")(delayed(fit_one)(v) for v in range(N))
        Phi = np.stack(Phi, axis=0)  # (N, D_rhythm)
        return Phi

    # ---------- Stage 2: Global patterns on signatures ----------
    def _aggregate_global_patterns(self, Phi: np.ndarray) -> Tuple[np.ndarray, GaussianMixture, int]:
        """Fit global GMM on Phi using BIC for model selection.

        Returns
        - U_global: (N, K_best) signature-space memberships
        - gmm_phi: trained GMM on Phi with K_best components
        - K_best: int
        """
        X = Phi
        if self.cfg.scale_signatures:
            self._sig_scaler = StandardScaler()
            X = self._sig_scaler.fit_transform(X)

        best_k = None
        best_bic = np.inf
        best_model = None
        for K in self.cfg.k_global_range:
            gm = GaussianMixture(
                n_components=K,
                covariance_type=self.cfg.covariance_type_global,
                n_init=self.cfg.n_init,
                max_iter=self.cfg.max_iter,
                reg_covar=self.cfg.reg_covar,
                random_state=self.cfg.random_state,
            )
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_k = K
                best_model = gm

        assert best_model is not None and best_k is not None
        U_global = best_model.predict_proba(X)
        return U_global.astype(np.float32), best_model, int(best_k)

    # ---------- Stage 3: Temporal memberships on embeddings ----------
    def _reconstruct_temporal_memberships(self, H: np.ndarray, K: int) -> Tuple[np.ndarray, GaussianMixture]:
        """Fit a K-component GMM on flattened embeddings and return U_final.

        H_flat: (N*T, D). If scale_embeddings=True, standardize using a
        StandardScaler fitted on H_flat before GMM.
        """
        N, T, D = H.shape
        H_flat = H.reshape(N * T, D)
        X = H_flat
        if self.cfg.scale_embeddings:
            self._emb_scaler = StandardScaler()
            X = self._emb_scaler.fit_transform(X)

        gm = GaussianMixture(
            n_components=K,
            covariance_type=self.cfg.covariance_type_global,
            n_init=self.cfg.n_init,
            max_iter=self.cfg.max_iter,
            reg_covar=self.cfg.reg_covar,
            random_state=self.cfg.random_state,
        )
        gm.fit(X)
        U = gm.predict_proba(X)  # (N*T, K)
        U_final = U.reshape(N, T, K)
        return U_final.astype(np.float32), gm

    # ---------- Utils ----------
    @staticmethod
    def _to_numpy(H) -> np.ndarray:
        if isinstance(H, np.ndarray):
            return H
        if torch is not None and isinstance(H, torch.Tensor):
            return H.detach().cpu().numpy()
        raise TypeError("H must be a numpy array or torch tensor")

