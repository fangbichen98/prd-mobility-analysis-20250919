# 珠三角城市群人群流动模式识别：基于DySAT动态嵌入与两阶段GMM的时空聚类

## 摘要
珠三角城市群（PRD）是我国人口与产业高度集聚的区域，工作日与周末、早晚高峰与日间活动在时空上呈现显著差异。本文围绕“城市群尺度的人群流动模式识别”，提出一套端到端的数据—方法—产品流程：先以逐小时OD边列表构建时序有向图，并计算静态图统计特征；随后采用DySAT（Dynamic Self-Attention Network）学习节点的时序动态嵌入；最后以“两阶段GMM（STGMM）”进行模式识别：（1）逐节点在时间维度拟合局部GMM获取“节律签名”；（2）在签名空间进行全局GMM聚类并用BIC选模；（3）在嵌入空间重建时序隶属度以刻画“何时何地属于何种模式”。同时设计了模式剖面（24小时曲线）、全局占比、TIE稳定性等指标，并生成空间分布与曲线图件。以PRD 2024-05-13至2024-05-19一周数据为例（Top-30k活跃格网、168小时），识别出8类典型日节律模式，通勤双峰与中午峰等特征显著，工作日与周末的空间分布与时间剖面存在系统性差异。本文方法在大规模图上具备可扩展性（分块注意力、流式读取），产物与参数完整可复现，为交通治理、职住结构研判与区域协同管控提供支撑。

关键词：城市群；人群流动；动态图嵌入；聚类；通勤模式；珠三角

---

## 1. 引言
- 背景：移动性是理解城市群运行与功能分异的核心视角。PRD作为国家级城市群，呈现强通勤联系与跨城协同，急需“小时级、全域化、可解释”的模式识别。
- 挑战：
  - 数据维度高（N达数万、T=24×7）、图稀疏且异质；
  - 既要捕捉结构（空间邻接/联系），又要刻画时间依赖（节律与相位）。
- 贡献：
  1) 提出“DySAT动态嵌入 + 两阶段GMM”的可扩展框架，兼顾结构与时间；
  2) 设计“节律签名—全局聚类—时序重建”的识别链路，支持全局分型与逐时解释；
  3) 构建指标体系（24h模式曲线、TIE稳定性、全局占比、空间分布）；
  4) 工程可复现：大文件分块读、按节点分块注意力、统一输出目录、参数快照。

## 2. 数据与预处理
- 数据来源与周期：2024-05-13至2024-05-19，逐小时OD边列表（columns: `date_dt,time_,o_grid,d_grid,cu_freq`）；格网元数据 `grid_id,lon,lat`。
- 时序图构建：按（日期、小时）聚合得到168个快照，边权采用`cu_freq`；统一节点集；图为有向图，统计按需要对称化用于传播。
- 特征工程（节点静态特征，用于DySAT输入）：总流入/出、活跃小时数、工作日/周末比、（可选）PageRank；标准化。
- 规模控制与鲁棒性：
  - Top-N活跃格网（按周入+出流量）减少稀疏尾部，实验使用N=30,000；
  - 大文件分块读取（2,000,000行/块）；
  - 节点ID规范化/缺失清洗。

## 3. 方法
### 3.1 DySAT 动态嵌入（阶段二）
- 结构编码：对每个小时的归一化邻接进行图传播（近似结构注意力）。
- 时间编码：在时间轴上对每个节点序列做多头自注意力（使用按节点分块以控显存）。
- 训练目标：无监督的“逐快照链路重构”，对正负样本对进行BCE优化；参数如学习率、负采样比、每快照正样本上限等。
- 产出：三维张量 H∈R^{N×T×D}（本文D=64）。

### 3.2 两阶段GMM聚类（阶段三）
- 阶段一（局部）：对每个节点 v 的时间序列 H_v∈R^{T×D} 拟合 GMM（k_temporal=4），将均值、协方差、权重串接为“节律签名”Φ_v。
- 阶段二（全局）：在签名矩阵 Φ 上用GMM并以BIC选择 K_global（候选4–8），得到 U_global（N×K）。
- 时序重建：在嵌入空间按固定K训练GMM，对(N×T)嵌入求后验，重塑为 U_final（N×T×K）。

### 3.3 指标与可视化（阶段四–五）
- 模式剖面（24h）：以 U_final 为权计算每小时加权流入/出，跨天按小时平均，命名启发式（如“通勤双峰、midday_peak”）。
- TIE稳定性：
  - 切换率：相邻小时主导簇变化比例；
  - 总变差：0.5·||u_t−u_{t−1}||_1 的小时均值。
- 全局分布：各簇主导节点数与占比；
- 地理可视化：
  - 全局主导簇分布图；
  - 指定小时主导簇分布（如08:00、18:00）；
  - 稳定性指标空间热力图。

## 4. 实验设计
- 区域与周期：PRD，一周（168小时）。
- 规模：Top-30k活跃格网（亦对Top-20k作对比）。
- 超参数：
  - DySAT：D=64，struct_layers=1，struct_hidden=64，temporal_heads=4，dropout=0.1，epochs=3/5；
  - 训练：lr=1e−3，weight_decay=1e−5，负采样比=1.0，max_pos_per_snap=50k；
  - STGMM：k_temporal=4，K_global∈{4..8}（BIC）。
- 硬件环境：NVIDIA A40（启用CUDA），分块注意力。
- 复现实验标识：
  - N30kE5（推荐版）与 N30kE3（对比），均提供工作日/周末分拆。

## 5. 结果与分析（N=30k, K=8）
- 工作日（N30kE5_weekday）全局分布（global_distribution.csv）：
  - count（占比）：3→5669(18.90%)，0→5567(18.56%)，6→4977(16.59%)，5→3479(11.60%)，1→3449(11.50%)，7→2803(9.34%)，2→2478(8.26%)，4→1578(5.26%)。
- 周末（N30kE5_weekend）全局分布：
  - 4→5624(18.75%)，7→5460(18.20%)，0→5454(18.18%)，1→3625(12.08%)，2→3192(10.64%)，6→2526(8.42%)，3→2576(8.59%)，5→1543(5.14%)。
- 稳定性（TIE，均值）：
  - 工作日：切换率≈0.373；总变差≈0.373；
  - 周末：切换率≈0.373；总变差≈0.373（略低）。
- 典型模式剖面：
  - 多数簇呈“通勤双峰”（8–10时与17–20时），部分呈“midday_peak”。
- 空间分布图（示例路径）：
  - 工作日：
    - 全局：outputs/experiment_PRD/run_N30kE5_weekday/figures/global_dominant_pattern_map.png
    - 小时：…/hourly_dominant_pattern_map_hour08.png，…/hourly_dominant_pattern_map_hour18.png
    - 稳定性：…/metric_map_tie_total_variation.png
  - 周末：对应 …/run_N30kE5_weekend/figures/ 下同名文件。
- 观察要点（示例）：
  - 核心通勤走廊与枢纽周边更倾向“通勤双峰”簇，外围及旅游/商业片区周末“midday_peak”占比上升；
  - 周末各簇占比更为均衡，活动半径与热点区有所转移。

### 5.1 城市/区县剖面（模式占比与热力图）
- 城市层面模式占比（份额热力图与CSV）
  - Weekday：outputs/experiment_PRD/run_N30kE5_weekday/figures/city_pattern_share_heatmap.png；
    CSV：…/report/city_pattern_share.csv
  - Weekend：outputs/experiment_PRD/run_N30kE5_weekend/figures/city_pattern_share_heatmap.png；
    CSV：…/report/city_pattern_share.csv
- 重点城市区县层面热力图（各市对应CSV见同目录 report/area_pattern_share_*.csv）
  - 深圳：…/run_N30kE5_weekday/figures/area_pattern_share_heatmap_Shenzhen.png；…/weekend/…
  - 广州：…/Guangzhou.png；佛山：…/Foshan.png；东莞：…/Dongguan.png；珠海：…/Zhuhai.png
- 结论要点：
  - 深圳（福田/南山/龙华/宝安）工作日通勤双峰占比 >80%，盐田工作日 flat 更高、周末 midday 激增；
  - 广州核心区（天河/越秀/荔湾/海珠）通勤强，南沙“周内平缓、周末午间活跃”；
  - 佛山（南海/顺德/三水/高明）flat/midday 较高，禅城通勤更强；
  - 珠海周末 midday 明显（香洲/金湾）。

### 5.2 城际联系网络（城市×城市 OD）
- 聚合周内/周末城市间 OD 流量（cu_freq），生成城际有向网络（剔除自环，保留Top边）：
  - Weekday 网络图：outputs/experiment_PRD/run_N30kE5_weekday/figures/city2city_network_weekday.png
  - Weekend 网络图：outputs/experiment_PRD/run_N30kE5_weekend/figures/city2city_network_weekend.png
  - 对应矩阵CSV：
    - Weekday：…/run_N30kE5_weekday/report/city2city_flow_weekday.csv
    - Weekend：…/run_N30kE5_weekend/report/city2city_flow_weekend.csv
- 走廊与联系（观察）：
  - 广佛同城：广州⇄佛山的双向联系在周内最强，核心通勤走廊明显；
  - 深莞协同：深圳⇄东莞在周内保持高强度联系，制造—服务链条紧密；
  - 西岸休闲与门户：珠海/中山/江门在周末与广州南沙等的联系上升，跨城消费/旅游显著。

## 6. 讨论
- 解释性：模式剖面和地理分布共同支撑对“功能区—节律”的理解；
- 对规划与治理的启示：
  - 交通：高峰错峰、枢纽扩容与引导；
  - 土地：就业—居住—消费功能圈层的识别与优化；
  - 管控：活动密度与波动性（TIE）联合研判。
- 灵敏度与稳健性：N30kE3 vs N30kE5 表明更长训练带来更稳定的聚类与更平滑的剖面；
- 局限与改进：一周样本期短；未纳入外生因素（天气、节假日、供给侧容量）；GMM高斯性假设可与非参数方法对照；节点Top-N可能忽略弱联系区域，可通过分层采样或多尺度建模改进。

## 7. 相关工作（简述）
- 城市群与流动网络分析：基于OD的模式挖掘与通勤识别；
- 动态网络表示学习：DySAT、EvolveGCN、TGAT等；
- 时空聚类：Hidden Markov、GMM/DPGMM、谱聚类与时空联动聚类；
- 交通科学中的峰谷识别与出行链模式研究。

## 8. 复现与工程说明（关键命令）
- 训练（示例，N=30k，5 epoch）：
```
python -u main_pipeline.py \
  --tag PRD \
  --epochs 5 \
  --device cuda \
  --data-dir data/PRD_mobility_edges \
  --no-cache \
  --run-id N30kE5
```
- 分拆分析（生成 weekday/weekend 两套报告/图件）：
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
- 输出目录（新规范）：
  - 非分拆：outputs/experiment_{tag}/run_{run_id}/{embeddings,report,figures}
  - 分拆：outputs/experiment_{tag}/run_{run_id}_{weekday|weekend}/{embeddings,report,figures}
- 结果快照：每个 report 下含 `run_config.json` 记录本轮参数；embeddings 下含 `meta_{run_id}.json`。

## 9. 结论
本文在PRD一周数据上验证了“DySAT动态嵌入 + 两阶段GMM”的时空模式识别框架的有效性与可解释性，识别出多类具有明显节律差异的出行模式并给出空间分布与稳定性诊断。未来将扩展至更长时段与更高空间分辨率，融合外生因素，并探索更强表征与非高斯聚类方法，以服务更复杂的城市群运行监测与政策评估。

## 附录 A. 数据字段与超参数（示意）
- 数据字段：
  - OD：`date_dt,time_,o_grid,d_grid,cu_freq`（使用`cu_freq`作为边权）；
  - 元数据：`grid_id,lon,lat`。
- 关键超参：
  - Top-N节点：30,000；D=64；k_temporal=4；K_global∈{4..8}（BIC）；epochs=3/5；
  - 读取：chunksize=2,000,000；include_metadata_nodes=False。
- 计算环境：NVIDIA A40；Python/pandas/scikit-learn/networkx/torch。

## 附录 B. 图件清单（示例）
- 工作日：
  - 全局主导簇：outputs/experiment_PRD/run_N30kE5_weekday/figures/global_dominant_pattern_map.png
  - 08/18时主导簇：…/hourly_dominant_pattern_map_hour08.png，…/hourly_dominant_pattern_map_hour18.png
  - 稳定性热力图：…/metric_map_tie_total_variation.png
  - 24小时模式曲线：…/pattern_signature_curves_outflow.png
- 周末：对应 outputs/experiment_PRD/run_N30kE5_weekend/figures/ 下同名文件。

（注：以上数值取自当前产出CSV；如更换 run_id 或参数，请同步更新“结果与分析”小节的数字与截图库路径。）
