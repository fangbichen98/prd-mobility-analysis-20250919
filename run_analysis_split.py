"""
Run Phases 3–5 separately for weekday and weekend subsets using existing
embeddings from Phase 2. Produces two sets of reports and figures under
run-specific directories to avoid overwrite.

Usage example:
  python run_analysis_split.py \
    --tag fullgpu \
    --run-id legacy \
    --k-temporal 4 \
    --k-range 4,5,6,7,8 \
    --n-jobs 2

Outputs:
  outputs/experiment_{tag}/report/run_{run_id}_weekday/ ...
  outputs/experiment_{tag}/report/run_{run_id}_weekend/ ...
  outputs/experiment_{tag}/figures/run_{run_id}_weekday/ ...
  outputs/experiment_{tag}/figures/run_{run_id}_weekend/ ...
"""
from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import CONFIG
from data.graph_cache import GraphCache
from analysis.pattern_recognizer import TwoStagePatternRecognizer, TwoStageConfig
from analysis.report_generator import ReportGenerator, ReportPaths
from visualization.geographic_visualizer import GeographicVisualizer
from visualization.cluster_visualizer import ClusterVisualizer


def _find_embeddings_file(emb_dir: str, run_id: str | None) -> tuple[str, str]:
    import glob
    if run_id:
        cand = os.path.join(emb_dir, f"embeddings_{run_id}.npz")
        if not os.path.exists(cand):
            raise FileNotFoundError(f"Embeddings for run_id '{run_id}' not found: {cand}")
        return cand, run_id
    files = glob.glob(os.path.join(emb_dir, "embeddings_*.npz"))
    if files:
        latest = max(files, key=os.path.getmtime)
        rid = os.path.splitext(os.path.basename(latest))[0].split("_")[-1]
        return latest, rid
    legacy = os.path.join(emb_dir, "embeddings.npz")
    if os.path.exists(legacy):
        return legacy, "legacy"
    raise FileNotFoundError(f"No embeddings found under {emb_dir}")


def ensure_phase1_from_cache(mobility_edges_dir_override: str | None = None, use_cache: bool = True):
    """Load graphs/nodes/time_keys using Phase-1 cache; fallback to rebuild.

    Reuses the helper from main_pipeline.ensure_phase1 with optional overrides
    so that graphs align with the embeddings' node set/data source.
    """
    from main_pipeline import ensure_phase1
    cache = GraphCache(CONFIG.paths.cache_dir)
    graphs, nodes, time_keys, _ = ensure_phase1(
        cache,
        mobility_edges_dir_override=mobility_edges_dir_override,
        use_cache=use_cache,
    )
    return graphs, nodes, time_keys


def _weekday_weekend_indices(time_keys: Sequence[str]) -> Tuple[List[int], List[int]]:
    import datetime as dt
    wk_idx, we_idx = [], []
    for i, tk in enumerate(time_keys):
        date_s, _ = str(tk).split("-")
        y, m, d = int(date_s[:4]), int(date_s[4:6]), int(date_s[6:8])
        wd = dt.date(y, m, d).weekday()  # 0=Mon..6=Sun
        (wk_idx if wd < 5 else we_idx).append(i)
    return wk_idx, we_idx


def run_one_split(split_name: str, idx: List[int], H: np.ndarray, nodes: Sequence[int], time_keys: Sequence[str], graphs_all, args) -> None:
    if not idx:
        print(f"[split:{split_name}] no snapshots; skipping")
        return
    # Subset embeddings and time
    Hs = H[:, idx, :]
    tks = [time_keys[i] for i in idx]
    graphs = [graphs_all[i] for i in idx]

    # Phase 3: STGMM
    cfg = TwoStageConfig(
        k_temporal=args.k_temporal,
        k_global_range=[int(x) for x in args.k_range.split(',') if x.strip()],
        n_jobs=args.n_jobs,
    )
    pr = TwoStagePatternRecognizer(cfg)
    U_final, U_global, global_gmm = pr.run(Hs)

    # Output dirs
    root = os.path.dirname(__file__)
    run_root_split = os.path.join(root, "outputs", f"experiment_{args.tag}", f"run_{args.run_id}_{split_name}")
    base_rep = os.path.join(run_root_split, "report")
    base_fig = os.path.join(run_root_split, "figures")
    os.makedirs(base_rep, exist_ok=True)
    os.makedirs(base_fig, exist_ok=True)
    # Ensure embeddings are available under split run root (symlink or copy)
    try:
        emb_split_dir = os.path.join(run_root_split, 'embeddings')
        os.makedirs(emb_split_dir, exist_ok=True)
        dst_npz = os.path.join(emb_split_dir, f'embeddings_{args.run_id}.npz')
        if not os.path.exists(dst_npz):
            try:
                os.symlink(os.path.abspath(emb_npz), dst_npz)
            except Exception:
                import shutil; shutil.copy2(emb_npz, dst_npz)
        src_meta = os.path.join(os.path.dirname(emb_npz), f'meta_{args.run_id}.json')
        if os.path.exists(src_meta):
            dst_meta = os.path.join(emb_split_dir, f'meta_{args.run_id}.json')
            if not os.path.exists(dst_meta):
                try:
                    os.symlink(os.path.abspath(src_meta), dst_meta)
                except Exception:
                    import shutil; shutil.copy2(src_meta, dst_meta)
    except Exception as e:
        print(f"[split:{split_name}] warn: failed to place embeddings in split dir: {e}")

    # Save memberships
    np.savez_compressed(os.path.join(base_rep, "memberships.npz"), U_final=U_final, U_global=U_global)

    # Phase 4: reports
    rg = ReportGenerator(ReportPaths(out_dir=base_rep))
    inflow, outflow = rg.compute_hourly_flows_from_graphs(graphs, nodes)
    profiles = rg.generate_pattern_profiles(U_final, inflow, outflow, tks)
    tie_df = rg.calculate_tie_stability(U_final)
    dist_df = rg.analyze_global_pattern_distribution(U_global)
    rg.export_node_summary(U_global, node_ids=nodes, grid_metadata_csv=CONFIG.paths.grid_metadata_csv, tie_df=tie_df)

    # Persist run configuration for this split
    try:
        import json as _json
        # Locate embeddings meta for training config (epochs, device)
        root_dir = os.path.dirname(__file__)
        emb_dir_new = os.path.join(root_dir, "outputs", f"experiment_{args.tag}", f"run_{args.run_id}", "embeddings")
        emb_dir_legacy = os.path.join(root_dir, "outputs", f"experiment_{args.tag}", "embeddings")
        meta_json = os.path.join(emb_dir_new, f"meta_{args.run_id}.json") if os.path.isdir(emb_dir_new) else os.path.join(emb_dir_legacy, f"meta_{args.run_id}.json")
        emb_meta = {}
        if os.path.exists(meta_json):
            with open(meta_json, 'r', encoding='utf-8') as f:
                emb_meta = _json.load(f)
        run_cfg = {
            "experiment_tag": args.tag,
            "run_id": args.run_id,
            "split": split_name,
            "embedding_shape": {"N": int(H.shape[0]), "T_split": int(Hs.shape[1]), "D": int(H.shape[2])},
            "embeddings_meta": emb_meta,
            "loader": {
                "data_dir": getattr(args, 'data_dir', None) or CONFIG.paths.mobility_edges_dir,
                "grid_metadata_csv": CONFIG.paths.grid_metadata_csv,
                "include_metadata_nodes": CONFIG.loader.include_metadata_nodes,
                "use_chunks": CONFIG.loader.use_chunks,
                "chunk_rows": CONFIG.loader.chunk_rows,
                "top_n_nodes_by_flow": getattr(CONFIG.loader, 'top_n_nodes_by_flow', None),
            },
            "analysis": {
                "k_temporal": args.k_temporal,
                "k_range": [int(x) for x in args.k_range.split(',') if x.strip()],
                "n_jobs": args.n_jobs,
            },
            "time_keys": tks,
        }
        with open(os.path.join(base_rep, "run_config.json"), 'w', encoding='utf-8') as f:
            _json.dump(run_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[split:{split_name}] warn: failed to write run_config.json: {e}")

    # Phase 5: figures
    gv = GeographicVisualizer(
        figure_dir=base_fig,
        grid_metadata_csv=CONFIG.paths.grid_metadata_csv,
        nodes=nodes,
        time_keys=tks,
        pattern_names=None,
    )
    names_json = profiles.get("pattern_names_json")
    if names_json and os.path.exists(names_json):
        import json
        with open(names_json, "r", encoding="utf-8") as f:
            pn = json.load(f)
        gv.pattern_names = {int(k): v for k, v in pn.items()}

    gv.plot_global_dominant_pattern_map(U_global)
    for h in (8, 18):
        gv.plot_hourly_dominant_pattern_map(U_final, hour=h)
    gv.plot_metric_map(tie_df, metric_col="tie_total_variation")

    # Cluster visuals
    cv = ClusterVisualizer(base_fig)
    cv.plot_pattern_signature_curves(profiles["hourly24_curves_csv"], names_json, metric="outflow")
    cv.plot_global_distribution_pie(distribution_csv=os.path.join(base_rep, "global_distribution.csv"))

    print(f"[split:{split_name}] done. report={base_rep} figures={base_fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run-id", default=None, help="embeddings run id (default=latest or legacy)")
    ap.add_argument("--k-temporal", type=int, default=4)
    ap.add_argument("--k-range", default="4,5,6,7,8")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--data-dir", default=None, help="override mobility edges directory for Phase 1 reload; set to PRD path to align with embeddings")
    ap.add_argument("--no-cache", action="store_true", help="disable cache read/write for Phase 1 reload")
    args = ap.parse_args()

    root = os.path.dirname(__file__)
    emb_dir_legacy = os.path.join(root, "outputs", f"experiment_{args.tag}", "embeddings")
    if args.run_id:
        emb_dir_new = os.path.join(root, "outputs", f"experiment_{args.tag}", f"run_{args.run_id}", "embeddings")
        emb_dir = emb_dir_new if os.path.isdir(emb_dir_new) else emb_dir_legacy
    else:
        emb_dir = emb_dir_legacy
    emb_npz, run_id = _find_embeddings_file(emb_dir, args.run_id)
    args.run_id = run_id

    npz = np.load(emb_npz)
    H = npz["H"]
    nodes = npz["node_ids"].tolist()
    time_keys = [t.decode() if isinstance(t, bytes) else str(t) for t in npz["time_keys"]]

    graphs, nodes_cached, time_keys_cached = ensure_phase1_from_cache(mobility_edges_dir_override=args.data_dir, use_cache=(not args.no_cache))
    # Sanity: node set should align; we assume same order by pipeline design

    wk_idx, we_idx = _weekday_weekend_indices(time_keys)
    print(f"weekday snapshots: {len(wk_idx)}, weekend snapshots: {len(we_idx)}")

    run_one_split("weekday", wk_idx, H, nodes, time_keys, graphs, args)
    run_one_split("weekend", we_idx, H, nodes, time_keys, graphs, args)


if __name__ == "__main__":
    main()
