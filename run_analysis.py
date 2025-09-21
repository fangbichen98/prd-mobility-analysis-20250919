"""
One-click runner for Phases 3–5 (pattern recognition, reporting, visualization).

Prerequisites
- Run Phase 1 & 2 to generate embeddings (via main_pipeline.py).
  This also populates caches for graphs and features.

This script will:
1) Load embeddings from outputs/experiment_{tag}/embeddings/embeddings_{run_id}.npz
2) Ensure graphs/nodes/time_keys are available via cache (Phase 1)
3) Run TwoStagePatternRecognizer to produce U_final and U_global
4) Generate reports (CSV/JSON) to outputs/experiment_{tag}/report/run_{run_id}/
5) Generate figures to outputs/experiment_{tag}/figures/run_{run_id}/
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np

from config import CONFIG
from data.graph_cache import GraphCache
from analysis.pattern_recognizer import TwoStagePatternRecognizer, TwoStageConfig
from analysis.report_generator import ReportGenerator, ReportPaths
from visualization.geographic_visualizer import GeographicVisualizer
from visualization.cluster_visualizer import ClusterVisualizer


def ensure_phase1_from_cache(mobility_edges_dir_override: str | None = None, use_cache: bool = True):
    """Load graphs/nodes/time_keys using Phase-1 cache; fallback to rebuild.

    Reuses the helper from main_pipeline.ensure_phase1 to avoid duplication.
    """
    from main_pipeline import ensure_phase1  # lazy import to avoid torch dep

    cache = GraphCache(CONFIG.paths.cache_dir)
    graphs, nodes, time_keys, feats = ensure_phase1(cache, mobility_edges_dir_override=mobility_edges_dir_override, use_cache=use_cache)
    return graphs, nodes, time_keys


def parse_k_range(s: str) -> Sequence[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _find_embeddings_file(emb_dir: str, run_id: str | None) -> tuple[str, str]:
    """Return (embeddings_path, resolved_run_id).

    If run_id provided, look for embeddings_{run_id}.npz.
    Else pick the latest embeddings_*.npz by mtime; fallback to embeddings.npz.
    """
    import glob, os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="experiment tag used in outputs/experiment_{tag}")
    ap.add_argument("--k-temporal", type=int, default=4)
    ap.add_argument("--k-range", default="4,5,6,7,8", help="candidate K_global list, comma-separated")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel jobs for local GMM fits")
    ap.add_argument("--hours", default="8,18", help="comma-separated hours-of-day to map (optional)")
    ap.add_argument("--data-dir", default=None, help="override mobility edges directory (must match embeddings)")
    ap.add_argument("--no-cache", action="store_true", help="disable cache read/write for Phase 1 re-load (use when testing)")
    ap.add_argument("--run-id", default=None, help="run id of embeddings to consume; default=latest")
    args = ap.parse_args()

    # Embeddings path
    root = os.path.dirname(__file__)
    # Prefer per-run embeddings directory when run_id provided (new layout); fallback to legacy
    emb_dir_legacy = os.path.join(root, "outputs", f"experiment_{args.tag}", "embeddings")
    if args.run_id:
        emb_dir_new = os.path.join(root, "outputs", f"experiment_{args.tag}", f"run_{args.run_id}", "embeddings")
        emb_dir = emb_dir_new if os.path.isdir(emb_dir_new) else emb_dir_legacy
    else:
        emb_dir = emb_dir_legacy
    emb_npz, run_id = _find_embeddings_file(emb_dir, args.run_id)
    npz = np.load(emb_npz)
    H = npz["H"]        # (N,T,D)
    node_ids = npz["node_ids"]
    time_keys = [str(x) for x in npz["time_keys"]]

    # Phase 1 artifacts (graphs)
    graphs, nodes_cached, time_keys_cached = ensure_phase1_from_cache(mobility_edges_dir_override=args.data_dir, use_cache=(not args.no_cache))
    # Sanity: node ordering for plots/report uses node_ids from embeddings
    # (Phase 2 exports embeddings in the same node order as Phase 1)

    # Phase 3: STGMM
    cfg = TwoStageConfig(
        k_temporal=args.k_temporal,
        k_global_range=parse_k_range(args.k_range),
        n_jobs=args.n_jobs,
    )
    pr = TwoStagePatternRecognizer(cfg)
    U_final, U_global, global_gmm = pr.run(H)

    # Save memberships for reuse
    run_root_dir = os.path.join(root, "outputs", f"experiment_{args.tag}", f"run_{run_id}")
    report_dir = os.path.join(run_root_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(report_dir, "memberships.npz"),
        U_final=U_final,
        U_global=U_global,
    )

    # Phase 4: Reports
    rg = ReportGenerator(ReportPaths(out_dir=report_dir))
    from analysis.report_generator import ReportGenerator as RG  # for type hints only
    inflow, outflow = rg.compute_hourly_flows_from_graphs(graphs, node_ids)
    profiles_paths = rg.generate_pattern_profiles(U_final, inflow, outflow, time_keys)
    tie_df = rg.calculate_tie_stability(U_final)
    dist_df = rg.analyze_global_pattern_distribution(U_global)
    # Node summary CSV (grid_id, lon, lat, dominant_pattern, tie metrics)
    rg.export_node_summary(U_global, node_ids=node_ids, grid_metadata_csv=CONFIG.paths.grid_metadata_csv, tie_df=tie_df)

    # Save run configuration into report dir for traceability
    try:
        meta_json = os.path.join(os.path.dirname(emb_npz), f"meta_{run_id}.json")
        meta = {}
        if os.path.exists(meta_json):
            import json as _json
            with open(meta_json, 'r', encoding='utf-8') as f:
                meta = _json.load(f)
        run_cfg = {
            "experiment_tag": args.tag,
            "run_id": run_id,
            "embeddings_npz": emb_npz,
            "embeddings_meta": meta,
            "embedding_shape": {"N": int(H.shape[0]), "T": int(H.shape[1]), "D": int(H.shape[2])},
            "loader": {
                "data_dir": args.data_dir or CONFIG.paths.mobility_edges_dir,
                "grid_metadata_csv": CONFIG.paths.grid_metadata_csv,
                "include_metadata_nodes": CONFIG.loader.include_metadata_nodes,
                "use_chunks": CONFIG.loader.use_chunks,
                "chunk_rows": CONFIG.loader.chunk_rows,
                "top_n_nodes_by_flow": getattr(CONFIG.loader, 'top_n_nodes_by_flow', None),
            },
            "analysis": {
                "k_temporal": args.k_temporal,
                "k_range": parse_k_range(args.k_range),
                "n_jobs": args.n_jobs,
                "hours": args.hours,
            },
        }
        with open(os.path.join(report_dir, "run_config.json"), 'w', encoding='utf-8') as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[run_analysis] warn: failed to write run_config.json: {e}")

    # Phase 5: Figures
    fig_dir = os.path.join(run_root_dir, "figures")
    gv = GeographicVisualizer(
        figure_dir=fig_dir,
        grid_metadata_csv=CONFIG.paths.grid_metadata_csv,
        nodes=node_ids,
        time_keys=time_keys,
        pattern_names=None,
    )
    # pattern names (Optional)
    names_json = profiles_paths.get("pattern_names_json")
    if names_json and os.path.exists(names_json):
        with open(names_json, "r", encoding="utf-8") as f:
            pattern_names = json.load(f)
        # keys are strings; convert to int for visualizer lookup convenience
        pattern_names_int = {int(k): v for k, v in pattern_names.items()}
        gv.pattern_names = pattern_names_int

    # Global dominant pattern map
    gv.plot_global_dominant_pattern_map(U_global)

    # Hourly maps
    try:
        hours = [int(x) for x in args.hours.split(",") if x.strip()]
    except Exception:
        hours = []
    for h in hours:
        gv.plot_hourly_dominant_pattern_map(U_final, hour=h)

    # Metric map (TIE total variation)
    gv.plot_metric_map(tie_df, metric_col="tie_total_variation")

    # Cluster visualizations
    cv = ClusterVisualizer(fig_dir)
    cv.plot_pattern_signature_curves(profiles_paths["hourly24_curves_csv"], names_json, metric="outflow")
    cv.plot_global_distribution_pie(distribution_csv=os.path.join(report_dir, "global_distribution.csv"))

    print("Analysis complete. Report and figures written to:")
    print(f"  report:  {report_dir}")
    print(f"  figures: {fig_dir}")


if __name__ == "__main__":
    main()
