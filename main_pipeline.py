"""
Main pipeline to run Phase 1 (data load + features) and Phase 2 (DySAT
training + embeddings export) as described in codex.md.

This script:
1) Loads graphs and features (uses cache if enabled and present).
2) Preprocesses to tensors for the DySAT model.
3) Trains the DySAT model in a self-supervised manner.
4) Exports embeddings to outputs/experiment_{tag}/embeddings/

Note: This script assumes PyTorch is installed. If not, please install
PyTorch in your environment before running.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np

from config import CONFIG
from data.graph_cache import GraphCache
from data.temporal_graph_loader import TemporalGraphLoader
from data.graph_statistics import GraphStatistics

# Lazy import torch-related modules to allow Phase 1 use without torch
def _torch_imports():
    import torch
    from models.dysat import DySAT, DySATConfig
    from models.dysat_preprocessor import DySATPreprocessor
    from models.dysat_trainer import DySATTrainer, TrainConfig
    return torch, DySAT, DySATConfig, DySATPreprocessor, DySATTrainer, TrainConfig


def ensure_phase1(cache: GraphCache, mobility_edges_dir_override: str | None = None, use_cache: bool | None = None):
    cfg = CONFIG
    if use_cache is None:
        use_cache = cfg.cache.enable_cache
    graphs = nodes = time_keys = None
    if use_cache and cache.graphs_exist(cfg.cache.graphs_key):
        graphs, nodes, time_keys = cache.load_graphs(cfg.cache.graphs_key)
    else:
        loader = TemporalGraphLoader(
            mobility_edges_dir=mobility_edges_dir_override or cfg.paths.mobility_edges_dir,
            date_col=cfg.loader.date_col,
            hour_col=cfg.loader.hour_col,
            src_col=cfg.loader.src_col,
            dst_col=cfg.loader.dst_col,
            weight_col=cfg.loader.weight_col,  # cu_freq per requirement
            include_metadata_nodes=cfg.loader.include_metadata_nodes,
            grid_metadata_csv=cfg.paths.grid_metadata_csv,
            use_chunks=cfg.loader.use_chunks,
            chunk_rows=cfg.loader.chunk_rows,
            enforce_int_node_ids=cfg.loader.enforce_int_node_ids,
            top_n_nodes_by_flow=cfg.loader.top_n_nodes_by_flow,
        )
        graphs, nodes, time_keys = loader.load()
        if use_cache:
            cache.save_graphs(cfg.cache.graphs_key, graphs, nodes, time_keys)

    feats = None
    if use_cache and cache.features_exist(cfg.cache.feats_key):
        feats = cache.load_features(cfg.cache.feats_key)
    else:
        stats = GraphStatistics(
            pagerank=cfg.feats.compute_pagerank,
            pagerank_alpha=cfg.feats.pagerank_alpha,
            epsilon_smoothing=cfg.feats.epsilon_smoothing,
        )
        feats = stats.compute_static_node_features(graphs, nodes, time_keys)
        if use_cache:
            cache.save_features(cfg.cache.feats_key, feats)

    return graphs, nodes, time_keys, feats


def run_phase2(graphs, nodes, time_keys, feats, experiment_tag: str, epochs: int, device: str, run_id: str):
    torch, DySAT, DySATConfig, DySATPreprocessor, DySATTrainer, TrainConfig = _torch_imports()

    # Resolve device with optional auto-detection.
    # - 'auto': prefer CUDA if available; else attempt MPS; else CPU.
    # - explicit 'cuda': fall back to CPU if CUDA unavailable.
    def resolve_device(req: str) -> str:
        req = (req or "").lower()
        if req == "auto":
            if torch.cuda.is_available():
                return "cuda"
            # MPS for Apple Silicon (optional)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if req == "cuda" and not torch.cuda.is_available():
            print("[main_pipeline] CUDA requested but not available; falling back to CPU")
            return "cpu"
        return req or "cpu"

    device = resolve_device(device)
    print(f"[main_pipeline] Using device: {device}")

    # Preprocess
    prep = DySATPreprocessor()
    adj_list = prep.build_adjs(graphs, nodes)
    x = prep.build_features(feats)

    # Model
    in_dim = x.shape[1]
    model_cfg = DySATConfig(in_dim=in_dim, struct_hidden=64, out_dim=64, struct_layers=1, temporal_heads=4, dropout=0.1)
    model = DySAT(model_cfg)

    # Trainer
    trainer = DySATTrainer(model=model, adj_list=adj_list, x=x, graphs_nx=graphs, device=device)
    tcfg = TrainConfig(epochs=epochs, lr=1e-3, weight_decay=1e-5, batch_neg_ratio=1.0, max_pos_per_snap=50000, device=device)
    trainer.train(tcfg, nodes)

    # Save checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"dysat_{experiment_tag}_{run_id}.pt")
    trainer.save_checkpoint(ckpt_path)

    # Embeddings
    H = trainer.get_embeddings().cpu().numpy()  # (N, T, D)

    # Export (new layout): per-run root with embeddings under run_{run_id}/embeddings
    run_root = os.path.join(os.path.dirname(__file__), "outputs", f"experiment_{experiment_tag}", f"run_{run_id}")
    out_dir = os.path.join(run_root, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, f"embeddings_{run_id}.npz")
    np.savez_compressed(
        emb_path,
        H=H,
        node_ids=np.array(nodes, dtype=np.int64),
        time_keys=np.array(time_keys),
    )
    # Persist richer metadata for traceability
    meta = {
        "experiment_tag": experiment_tag,
        "run_id": run_id,
        "shape": {"N": int(H.shape[0]), "T": int(H.shape[1]), "D": int(H.shape[2])},
        "checkpoint": ckpt_path,
        "embeddings": emb_path,
        "train": {
            "epochs": int(epochs),
            "device": device,
            "lr": float(1e-3),
            "weight_decay": float(1e-5),
            "batch_neg_ratio": float(1.0),
            "max_pos_per_snap": int(50000),
        },
        "model": {
            "in_dim": int(in_dim),
            "struct_hidden": int(model_cfg.struct_hidden),
            "out_dim": int(model_cfg.out_dim),
            "struct_layers": int(model_cfg.struct_layers),
            "temporal_heads": int(model_cfg.temporal_heads),
            "dropout": float(model_cfg.dropout),
        },
        "loader": {
            "top_n_nodes_by_flow": getattr(CONFIG.loader, 'top_n_nodes_by_flow', None),
            "include_metadata_nodes": CONFIG.loader.include_metadata_nodes,
            "use_chunks": CONFIG.loader.use_chunks,
            "chunk_rows": CONFIG.loader.chunk_rows,
            "grid_metadata_csv": CONFIG.paths.grid_metadata_csv,
        },
    }
    with open(os.path.join(out_dir, f"meta_{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"), help="experiment tag")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="auto", help="'auto'|'cpu'|'cuda' (falls back to cpu if unavailable)|'mps'")
    ap.add_argument("--data-dir", default=None, help="override mobility edges directory for Phase 1 (small test)")
    ap.add_argument("--no-cache", action="store_true", help="disable cache read/write for Phase 1")
    ap.add_argument("--run-id", default=None, help="unique run id to suffix outputs; default=timestamp")
    args = ap.parse_args()

    cache = GraphCache(CONFIG.paths.cache_dir)
    graphs, nodes, time_keys, feats = ensure_phase1(cache, mobility_edges_dir_override=args.data_dir, use_cache=(not args.no_cache))

    # Phase 2 (DySAT)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_phase2(graphs, nodes, time_keys, feats, args.tag, args.epochs, args.device, run_id)


if __name__ == "__main__":
    main()
