# 实验索引

这份文件只回答“哪些结果可以放在一起看、哪些不能放在一起看”。具体数字和图从当前统一报告与图索引进入，历史细节在 `docs/archive/`。

## 阅读顺序

| 顺序 | 入口 | 用途 |
|---|---|---|
| 1 | `docs/CURRENT_STATUS.md` | 当前进展、结论和剩余风险 |
| 2 | `docs/analysis/UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md` | 性能、错误转移、表示、边界和监督差异 |
| 3 | `docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md` | OOS 主图合同、三类主图入口、补充证据路径和阅读顺序 |

## 结果分层

2026-09-06 最新实验入口：[自适应中心与 Gate/pipeline 指标对账](analysis/historical_trainable_adaptive_centers_presentation.md)。
该实验使用 KIR=.50、seed=13/42/87、validation OOS F1-only 选择；Known/Accuracy 不再是硬约束。
九单元真实 CUDA cascade 已完成，自适应中心没有关闭 CLINC 论文差距。后续 [checkpoint follow-up](analysis/historical_trainable_checkpoint_selection_presentation.md) 的九个训练单元与三个真实确认也已完成：CLINC `91.64±.27%`，逐 seed `91.37/91.56/92.01%`，仅一个 seed 超过论文；同时报告 Gate/pipeline macro F1 和 Accuracy 的净损失。

新增 [KIR=.25 CLINC full pipeline](analysis/historical_trainable_kir25_full_pipeline_presentation.md)：Trainable K=1 Gate 接入同 KIR 重新训练的 Router/Expert 后，OOS F1 `95.10±.30%`，相对论文 KIR=.25 Ours `95.01%` 为 `+0.09 pp`，2/3 seed 超过；这是目前 CLINC 上最好的 matched full-pipeline 配置，但仍属于 H1 controlled，且 macro F1/Accuracy 必须单独报告。

KIR=.50 的 [geometry/recipe 搜索](analysis/historical_trainable_gate_search_presentation.md) 已完成：geometry 为 `91.79±.43%`，Known-only recipe 为 `91.93±.48%`，仍低于论文 `91.96%`。这批实验没有支持“简单增加中心或继续调阈值即可突破”的结论。

新增[完整主实验与论文设置消融报告](analysis/historical_paper_ablation_report.md)：当前 H1 Trainable 的已完成 full-pipeline 结果按 KIR 汇总；论文消融则完整覆盖 `CLINC150/StackOverflow/BANKING77-OOS × KIR=.25/.50/.75 × Ours/Without Gate/Cascade-MiniLM/Cascade-SmolLM` 的 36 个历史 anchor 单元。历史消融的 CUDA 状态为不可用，因此只作为论文设置参照，不与当前 H1 结果混合排名。

当前 H1 full pipeline 已闭合为 `27/27` 个 CUDA 单元（3 数据集 × 3 KIR × 3 seed）；其中 KIR=.25 和 KIR=.75 是直接 Trainable K=1，KIR=.50 使用当前各数据集已锁定的最优 H1 配置。历史 Ours artifact 与当前 H1 的 Banking 数据键都是 `banking77_oos`；archive `banking77` 仍单独保留，比较时必须标注方法合同差异。

新增 [Banking77-OOS OOS-first 搜索](analysis/historical_trainable_oos_priority_presentation.md)：只按 validation OOS F1 选边界，KIR=.25 达到 `95.95±0.48%`，相对历史 Ours `93.99%` 提升 `+1.96 pp`，Known Recall 降至约 `61.53%`；KIR=.75 提升约 `+2.07 pp`。报告同时汇总了 KIR=.50 的旧均衡结果 `91.53±0.05% OOS F1 / 75.84% Known F1` 与 OOS-first `91.73±0.08% / 71.30%` 的差异。Known F1、Accuracy 和 False Acceptance 均已作为代价报告。

新增 [OOS-SOTA 与 Known/Accuracy 代价权衡](analysis/historical_oos_sota_tradeoff.md)：以论文其他 baseline 的 OOS F1 最好值为验证集硬约束，三组 BANKING77-OOS CUDA full pipeline 已直接确认；StackOverflow KIR=.25 是当前最平衡工作点。

新增 [Frozen/Trainable Gate 表示消融](analysis/historical_gate_ablation_report.md)：固定 Router、Expert、KIR、K=1 和边界规则，只切换 Gate 是否训练；共完成 54 个 CUDA full-pipeline 单元。

新增 [论文设置 Trainable Gate 扩展](analysis/historical_paper_style_trainable_gate_ablation_report.md)：seed42、K=2、论文 lambda 和 threshold=1 的 9 个 CUDA 单元；下游使用当前 H1 fixed Router/Expert，作为 paper-style extension 单独报告。

新增 [论文几何四变体消融](analysis/trainable_paper_four_variants_paper_geometry.md)：按论文布局重新运行 Ours、Without Gate、Cascade-MiniLM、Cascade-SmolLM，共 36 个当前 H1 CUDA 单元。

新增 [CCF A/B 审稿意见闭环审计](analysis/REVIEWER_GAP_CLOSURE_V1.md)：逐项标注 MOGB、K 消融、调参合同、指标口径、Related Work 和 deployment claim 的当前证据与剩余缺口。Frozen/Trainable K=1 的 Gate-only 与 full Cascade CUDA benchmark 见 [benchmark 报告](analysis/historical_deployment_benchmark.md)。

| 层 | 可比较对象 | 当前状态 | 结论用途 |
|---|---|---|---|
| same-protocol fair | Trainable K=1、Frozen single-centroid/Frozen K=2（Euclidean MOGB-Fair 组件）、Random K=2、边界交换 | 已闭合的主要层 | 比较表示、中心、半径和 coverage–rejection 工作点 |
| native detector | Trainable 表示上的 Gate、MSP、Energy、kNN、LOF | 已有对照 | 区分表示收益和 detector 规则收益 |
| external backbone | MSP、DOC、ADB、DA-ADB、MOGB 官方兼容单元 | MSP/DOC/ADB 已闭合；DA-ADB 部分完成；MOGB 官方状态表已补齐但仍非严格复现；KNNCL 仍 blocked | 同源数据上的外部合同参考，不进入 MiniLM fair 排名 |
| different supervision | DCLOOS pseudo-OOS / external-OOS | reduced 或 blocked | 监督强度和成本参考，不与 Known-only 直接排名 |
| historical system-level | `fulltex.tex` 历史 Cascade、旧版 TextOIR 表 | 历史资料 | 说明系统层级差异，不作为当前 S2C 的同协议结果 |
| historical H1 parameterized cascade | Trainable MiniLM Gate 的 K/λ/threshold/acceptance work-points + 固定 H1 Router/Expert | 已闭合，2268 个测试候选、9 个 selected 直接 pipeline 核验 | 以 validation 选参、test confirmation；与论文 Ours 做 OOS F1 参照，不声称严格 H0 复现 |
| blocked / incomplete | KNNCL、OpenMax、DeepUnk 等未闭合路线 | 不补伪指标 | 记录缺口，等待同 split、同 Known list、同评估器的 final metrics；DOC 已有 native compatibility 结果 |

### 后续协议与历史协议的边界

后续新实验统一使用 `historical_v19_paper_main` 的 H1 controlled 可执行快照：
`../assets/datasets/s2c/prepared/data/multidataset/v19/{clinc150,stackoverflow,banking77_oos}`，
按旧 `KNOWN_INTENTS.json`、seed=42 和论文的 KIR 设置构造。论文主 Cascade 使用旧的
`K_y=2`、对角 Mahalanobis、CLINC lambda=.5/其余 lambda=1 和历史 Router/Expert；Frozen/Trainable
K=1 只作为同一旧数据上的 Gate-only 控制。

当前仍处于实验验证阶段：结果、协议审计和 OOS 可视化可以继续更新，但 `fulltex.tex` 和论文正文暂不修改。

`protocol_v2_textoir_v1` 是已经完成的冻结参考，不再作为新实验默认。旧 fulltex/v19 本身也
不是一个完全无缺陷的单一协议：`fulltex.tex` 主表把论文 reference 写作 Banking77，而当前 H1
可执行主线使用 `banking77_oos`；
Banking77 原始 split 也不是严格的 6:1:3；CLINC 和 Banking 原始样本存在跨 split 重复。
这些问题要在结果中说明，不能用新协议悄悄替换。

因此，本文档固定使用两个明确数据键：`banking77_oos` 代表原论文/历史 H1 v19 主线；
`banking77` 代表 archive full pipeline 与当前 protocol_v2 参考线。两者不能在一张无标注表或图中合并。

此外，H0 归档锚点在不同 KIR 行之间还改变了 CLINC 数据根和语义 Gate 模式，因此不能把
整张论文主表压缩成一个未经核验的“旧协议”名称；需要精确到数据集×KIR×锚点。H1 controlled
v19 才是后续 Frozen/Trainable 同协议比较的固定基准。

严格 H0 主 Cascade 另行处理：只有 `data/v19`、历史 Gate detector、匹配的 Router/Expert
和必要的语义输入同时存在时才允许运行；当前不以 H1 快照替代它。

本阶段主视觉收缩到 OOS detection：OOS F1/precision/recall、AUROC/AUPR、false acceptance
和样本状态转移。Known Recall/false reject 仅作 coverage guard，不再扩展 Known intent 分类图。

TEXTOIR baseline 的已有 native 结果属于冻结 protocol_v2 参考。以后若与历史 S2C 主表并列，
必须先用旧 v19 的 Known 列表和 OOS 测试集生成同一输入；不能把当前 protocol_v2 的
compatibility 行直接填入历史表，也不能用中间 prediction 或残留 `results.csv` 填补。

## 当前主结果的位置

- current fair 轻量结果：`results/analysis/cross_protocol_tradeoff_v1/`
- ADB 外部工作点：`results/analysis/adb_kir_sensitivity_v2/`
- ADB/DA-ADB 样本级错误机制：`results/analysis/da_adb_current_protocol_summary_v1/`；对应图为 `figures/da_adb_current_protocol_summary_v1/`
- MOGB-Fair 图和报告：`figures/s2c_baseline_mogb_overview_v1/`、`docs/analysis/UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`
- 深层空间机制图包：`figures/deep_geometry_mechanism_v1/`、`results/analysis/deep_geometry_mechanism_v1/`
- 历史 v19 主线 OOS 机制实验与图组：`figures/historical_protocol_oos_v3/`、`results/analysis/historical_protocol_v3/`
- archive 数据协议下的 Gate→Router→Expert 结果：`results/analysis/historical_archive_full_pipeline/`、`docs/analysis/historical_pipeline_experiment_presentation.md`
- 面向论文 OOS F1 的 archive tuned 候选：`results/analysis/historical_archive_tuned_full_pipeline/`、`docs/analysis/historical_pipeline_experiment_presentation.md`
- 历史 H1 Trainable Gate 参数化 full pipeline：`results/analysis/historical_trainable_parameter_full_pipeline/`、`docs/analysis/historical_trainable_parameter_full_pipeline_presentation.md`
- 冻结 `protocol_v2` 的 OOS 低阅读成本参考图：`figures/oos_readability_figures_v1/`、`results/analysis/oos_readability_figures_v1/`
- score/错误机制图包：`figures/mechanism_explanation_v1/`、`results/analysis/mechanism_explanation_v1/`
- S2C–MOGB 机制面板：`results/analysis/s2c_vs_mogb_mechanism_dashboard_v1/`、`figures/s2c_vs_mogb_mechanism_dashboard_v1/`
- 跨方法 OOS 比较与机制汇报：`docs/analysis/cross_method_oos_mechanism_presentation.md`；图与派生表位于 `figures/cross_method_oos_mechanism/`、`results/analysis/cross_method_oos_mechanism/`，按 fair / TextOIR-BERT / 外部监督分层。
- 历史 H1 Fig.1 的 3D 局部白化单位球版本：`figures/historical_oos_visual_explanation/paper_style_local_boundary_geometry_3d.png`；真实 384 维接受/拒绝计数见 `results/analysis/historical_oos_visual_explanation/local_boundary_geometry_3d_summary.csv`。
- 统一合同审计：`results/analysis/unified_prediction_contract_v1/`
- 机器可读资产登记：`configs/experiment_registry.yaml`

## 规则

1. `Known Recall`、false acceptance/rejection、OOS F1、F1-All、AUROC 和 AUPR 只有在数据划分、Known list、seed、评估器和阈值选择合同一致时才进入 fair 对照。
2. 测试 OOS 不能参与 checkpoint、阈值、K 或结构选择；测试集曲线只用于事后解释。
3. Gate-only、完整 Cascade、MOGB 官方 BERT、MOGB-MiniLM-Fair 和 DCLOOS reduced 必须分栏。
4. 新分析先登记 `analysis_bundles`，再生成结果和图；不要在 `docs/analysis/` 创建一次性报告。
5. 已完成但不再作为当前入口的报告放在 `docs/archive/analysis/`，不删除原始 artifact。

审计命令：

```bash
python tools/maintenance/audit_asset_catalog.py
python tools/maintenance/check_research_state.py
```
