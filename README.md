# PRD Mobility Analysis (20250919)

基于 DySAT 动态嵌入与两阶段 GMM（STGMM）的珠三角（PRD）人群流动模式识别。支持全周与工作日/周末分拆，输出模式分布、24h 曲线、稳定性（TIE）与地图/图表，并提供城市/区县剖面与城际联系网络。


## 环境依赖
- Python 3.10+
- 主要包：numpy, pandas, networkx, scikit-learn, matplotlib, joblib, torch (GPU 可选)
- 建议：CUDA GPU（如 NVIDIA A40）；CPU 亦可但训练耗时更长。

## 数据准备
- OD 边列表（逐天 CSV，逐小时）：放入 `data/PRD_mobility_edges/`
  - 列要求：`date_dt,time_,o_grid,d_grid,cu_freq`（使用 `cu_freq` 作为边权）
- 格网元数据：`data/PRD_grid_metadata.csv`
  - 列要求：`grid_id,lon,lat`（可选：`area_name,city_name` 便于城市/区县统计）

示例头：
```
# OD
date_dt,time_,o_grid,d_grid,cu_freq,total_freq
20240513,0,95244,100275,1,3

# 元数据
grid_id,lon,lat,area_name,city_name
149425,113.8119202,22.3973503,南山区,深圳市
```

## 目录结构（代码与产物）
```
20250919-analysis/
├─ data/                     # 仅代码与结构，数据 CSV/edges 不纳入 git
│  ├─ PRD_grid_metadata.csv
│  ├─ PRD_mobility_edges/
│  ├─ cache/                 # 运行缓存（已忽略）
│  ├─ *.py (loader/stat/cache)
├─ models/                   # DySAT 模型/预处理/训练（checkpoints 已忽略）
├─ analysis/                 # STGMM 聚类与报告生成
├─ visualization/            # 地图/图表绘制
├─ outputs/                  # 实验产物（已忽略）
│  └─ experiment_{tag}/
│     ├─ run_{run_id}/
│     │  ├─ embeddings/      # embeddings_{run_id}.npz, meta_{run_id}.json
│     │  ├─ report/          # CSV/JSON（含 run_config.json）
│     │  └─ figures/         # PNG 图件
│     ├─ run_{run_id}_weekday/{embeddings,report,figures}
│     └─ run_{run_id}_weekend/{embeddings,report,figures}
├─ config.py                 # 路径与 loader/缓存/特征配置
├─ main_pipeline.py          # Phase 1–2: 加载+特征+DySAT 训练/导出
├─ run_analysis.py           # Phase 3–5: 全周聚类+报告+图件
├─ run_analysis_split.py     # Phase 3–5: 工作日/周末分拆
└─ codex.md                  # 项目背景与模块职责
```

## 快速开始
1) 训练（Phase 1–2）
```
python -u main_pipeline.py \
  --tag PRD \
  --epochs 5 \
  --device cuda \
  --data-dir data/PRD_mobility_edges \
  --no-cache \
  --run-id N30kE5
```
- 产物：`outputs/experiment_PRD/run_N30kE5/embeddings/embeddings_N30kE5.npz`（含 `H,node_ids,time_keys`）与 `meta_N30kE5.json`（记录 train/model/loader 快照）

2) 全周分析（Phase 3–5）
```
python -u run_analysis.py \
  --tag PRD \
  --run-id N30kE5 \
  --k-temporal 4 \
  --k-range 4,5,6,7,8 \
  --n-jobs 4 \
  --hours 8,18 \
  --data-dir data/PRD_mobility_edges \
  --no-cache
```
- 产物：`outputs/experiment_PRD/run_N30kE5/{report,figures}`；`report/run_config.json` 记录本轮参数。

3) 工作日/周末分拆
```
python -u run_analysis_split.py \
  --tag PRD \
  --run-id N30kE5 \
  --k-temporal 4 \
  --k-range 4,5,6,7,8 \
  --n-jobs 4 \
  --data-dir data/PRD_mobility_edges \
  --no-cache
```
- 产物：`outputs/experiment_PRD/run_N30kE5_weekday/{embeddings,report,figures}` 与 `..._weekend/...`

## 配置要点（config.py）
- `paths.grid_metadata_csv`: 默认 `data/PRD_grid_metadata.csv`
- `loader.weight_col`: `cu_freq`
- `loader.use_chunks, chunk_rows`: 大 CSV 分块读取（默认启用）
- `loader.include_metadata_nodes`: 是否把元数据全部 grid 纳入节点（PRD 规模默认 False）
- `loader.top_n_nodes_by_flow`: 按周总入+出流量选 Top‑N 节点（默认 30000，平衡效果与显存）

## 方法概要
- DySAT（结构传播 + 时间注意力）学习时序嵌入 H（N×T×D）；
- 两阶段 GMM（STGMM）：
  1) 局部：每节点对 H_v（T×D）拟合 GMM→节律签名 Φ_v；
  2) 全局：Φ 上用 BIC 选 K 聚类→U_global；
  3) 嵌入时序重建：对 (N×T) 嵌入以 K 拟合 GMM，获得 U_final（N×T×K）。
- 指标：
  - 全局分布：`global_distribution.csv`（count/share/soft_mean）
  - 稳定性（TIE）：`tie_metrics.csv`（switch_rate、total_variation）与 `metric_map_*.png`
  - 模式剖面：`pattern_profiles_timekey.csv`、`pattern_profiles_hourly24.csv`、`pattern_names.json`
  - 地图：`global_dominant_pattern_map.png`、`hourly_dominant_pattern_map_hour{08,18}.png`
  - 节点汇总：`node_summary.csv`（grid_id, lon, lat, dominant_pattern, TIE）
- 扩展产物（N30kE5 已生成）：
  - 城市/区县 × 模式占比：`city_pattern_share.csv`、`area_pattern_share_*.csv` 与相应热力图
  - 城市×城市 OD：`city2city_flow_week{day,end}.csv` 与网络图 `city2city_network_*.png`

## 结果复现（PRD 一周，N=30k, K≈8）
- 推荐 run：`N30kE5`（5 epoch）；对比 run：`N30kE3`（3 epoch）
- 分析与图件见 `outputs/experiment_PRD/run_N30kE5_*/*`

## 常见问题（FAQ）
- CUDA OOM：
  - 降低 `loader.top_n_nodes_by_flow`（如 20000/10000）
  -（若仍紧张）可在 `models/dysat.py` 中将时间注意力 `chunk_size` 从 1024 调小
- Matplotlib 中文缺字：
  - 安装中文字体（如 SimHei），或在绘图前设置 `plt.rcParams['font.sans-serif']=['SimHei']`；本仓库默认英文字体可能告警但不影响数据。

## 许可
- 建议选择 MIT；目前未附 LICENSE，请根据项目需要添加。

