# 当前研究状态

2026-09-11 严格 Known-only 三 seed 消融已完成：`trainable_full_pipeline_ablation_known_only` 完成 135/135 CUDA 变体单元。Cascade-MiniLM 已修正为独立 Frozen MiniLM Gate + Known-only 阈值 + Frozen MiniLM 下游头；Ours 在九个 dataset×KIR 设置的 OOS F1 和 Accuracy 均高于四个消融。漏检计数回归测试 4 项通过：误放行 OOS 计为 FN，下游仅更换 Known 标签不改变该 FN；再次拒识才改变最终 OOS F1。详见[消融核对](analysis/trainable_full_pipeline_ablation_known_only.md)。

2026-09-11 Known-only 搜索已得到数值上的 **9/9 OOS-F1 优势**：CLINC150、StackOverflow、BANKING77-OOS 在 KIR=.25/.50/.75 均超过当前最强外部参照。CLINC150 三格为 `95.19±0.48/92.13±0.81/86.80±1.55`，相对差值 `+1.63/+2.03/+0.80 pp`。汇总见[Known-only 9/9 结果](analysis/historical_known_best_9of9.md)。所有选择仅使用 Known 训练/验证，未使用 pseudo-OOS；但该汇总合并了两轮最终评估，StackOverflow 固定合同确定前已有探索性 test artifact 被查看，因此仍需一次预注册全部九格锁定后的 fresh rerun，才能作为干净的 untouched-holdout 论文主表结果。

2026-09-11 前置搜索记录：`historical_known_geometry` 完成 27/27 个 Known-only 验证单元，`historical_known_training` 完成 11 配方筛选及三 seed 扩展；随后由锁定配置完成九格 test 评估并形成上述汇总。

2026-09-11 严格 Known-only 搜索记录：`historical_known_geometry` 覆盖三数据集 × 三 KIR × 三 seed，禁止真实及 pseudo-OOS 选参，仅以 Known 正确接收率减去四倍错误接收率选择 K、距离、半径、acceptance 与 margin fusion；该阶段不读取 test。

2026-09-10 Known-only 实验收口：CLINC 校准九单元和新表示训练六单元均已完成 CUDA。新表示 OOS F1 在 KIR=.50/.75 为 **89.64±0.77 / 81.37±0.54**，距离表中外部最好值仍差 **0.46 / 4.63 pp**，9/9 目标未达成。两种 Known-only 校准规则均未改善旧固定阈值。详见[实验结果](analysis/historical_known_validation.md)。本轮为 Gate-only，尚无 full-pipeline 确认。

2026-09-10 Known-only validation follow-up：用户当前要求改为仅 Known validation 选择，优先 CLINC。已启动[CLINC 校准实验](analysis/historical_known_validation.md)：三 KIR × 三 seed，固定现有 Known-only checkpoint、K=1 和距离，以 Known macro F1 为主选择规则，95% Known 覆盖率为预定对照。全部阈值锁定后评估 test，当前尚无完成结果。下方 validation OOS F1 only 描述属于历史阶段。

2026-09-09 paper-style Trainable Gate extension：按论文的 K=2、CLINC λ=.5、其余 λ=1、threshold=1、normalized boundary 和 seed42 完成 9 个 CUDA 单元。该扩展使用当前 H1 fixed Router/Expert，因为论文原始部分 checkpoint 路径无法完整恢复；结果单独见[论文设置 Trainable Gate 扩展](analysis/historical_paper_style_trainable_gate_ablation_report.md)，不能写成严格论文 H0 消融。

2026-09-09 current Trainable Gate four-variant ablation：按论文表格结构完成 Ours、Without Gate、Cascade-MiniLM、Cascade-SmolLM 的 36/36 CUDA 单元（3 数据集 × 3 KIR × seed42）。Ours 使用当前 Trainable Gate；结果见[Trainable Gate 四变体消融](analysis/trainable_paper_four_variants.md)。该表是当前 H1 matched ablation，历史论文四变体 artifact 继续单独保留。

2026-09-09 paper-geometry four-variant ablation：将 Ours 固定为论文 K=2、CLINC λ=.5、其余 λ=1、threshold=1、normalized boundary 后，重新完成 36/36 CUDA 单元；结果见[论文几何四变体消融](analysis/trainable_paper_four_variants_paper_geometry.md)。这是当前 H1 的 paper-style extension，不能冒充原论文 H0。

2026-09-10 reviewer-gap audit：已将附件中的 CCF A/B 风险逐项映射到当前证据，见[审稿意见闭环审计](analysis/REVIEWER_GAP_CLOSURE_V1.md)。MOGB 完整官方复现仍未闭合；Frozen/Trainable K=1 的 Gate-only 与 full Cascade CUDA deployment benchmark 已完成，完整四变体/CPU 部署仍不作声明。

2026-09-08 指标纠正：此前 Known F1 对比使用了仅真实 Known 子集的 legacy known_macro_f1，不能支持“三个条件全面 SOTA”。按全测试集 Known-class macro F1 重算后，CLINC .50 / StackOverflow .25 / Banking77-OOS .25 分别为 82.45 / 80.07 / 60.42，均未超过对应外部 baseline Known F1。OOS F1、Accuracy 不变。已筛查 3636 个已有候选来源记录，并按验证集选择 30 个折中工作点，见[口径纠正与配置筛选](analysis/historical_known_oos_tradeoff.md)。下文 legacy Known 数字需按此修正理解。

2026-09-08 OOS-SOTA tradeoff：将“验证集 OOS F1 达到论文其他 baseline 最好值”作为硬约束，再按验证集 Accuracy、Known Recall 选择配置；BANKING77-OOS KIR=.25/.50/.75 已完成 9 个 CUDA 直接 full-pipeline 确认。结果为 `95.89/65.45/92.37`、`91.53/67.46/85.59`、`88.25/70.53/81.36`（Known/OOS/Acc）。跨数据集最平衡的是 StackOverflow KIR=.25：`80.07/95.38/91.58`，相对其他 baseline 最好值的差值为 `−0.80/+2.73/+2.51 pp`。详见[OOS-SOTA与代价权衡报告](analysis/historical_oos_sota_tradeoff.md)。

2026-09-08 matched Gate ablation：在固定同一 Router/Expert、K=1、对角 Mahalanobis、mean+1×std、threshold=1 和同一数据/seed下，Frozen Gate 与 Trainable Gate 的 54 个 CUDA full-pipeline 单元已完成。Trainable 相对 Frozen 的 OOS F1 增益为 CLINC `+0.80/+1.47/+1.84 pp`、StackOverflow `+1.52/+9.76/+12.03 pp`、BANKING77-OOS `+1.00/+3.65/+3.05 pp`（KIR=.25/.50/.75）；对应 Accuracy 全部上升，False Acceptance 全部下降。见[Gate 表示消融报告](analysis/historical_gate_ablation_report.md)。

更新时间：2026-09-07
新实验默认协议：`historical_v19_paper_main`
冻结参考协议：`protocol_v2_textoir_v1`
当前主方法：**Known-only Trainable MiniLM + K=1 单中心 Gate**
当前阶段：**实验验证阶段**；`fulltex.tex` 暂不维护，论文写作不在本阶段范围内。

## Banking 数据键确认：历史 `93.99` Ours artifact 实际使用 `banking77_oos`

- 当前 H1 主线和历史 Ours artifact 的实际数据键都是 `banking77_oos`；`fulltex.tex` 只把该列展示为 `Banking77`。
- `artifacts/s2c/outputs/paper_results/banking77_oos/kir25_seed42/full_anchor/eval_results.json` 实际记录 OOS F1 `93.9852%`、Accuracy `89.0686%`，即论文表中的 `93.99/89.07`。
- `banking77`（19 Known/58 OOS/3080 test at KIR=.25）是另一条 archive/protocol_v2 数据线，不能替代当前 H1 的 `banking77_oos`。
- 详见[完整主实验与论文设置消融报告](analysis/historical_paper_ablation_report.md)。

## 最新阶段：KIR=.25 CLINC full pipeline 已超过论文参照

- 用户最新选择目标是 **validation OOS F1 only**；Known F1、Accuracy、Known Recall 仅作诊断，不再使用旧实验的 1 pp guard。下方早期 guard 描述只适用于对应历史实验。
- 自适应中心九个 dataset×seed 单元已完成 CUDA full pipeline。overall 结果为 CLINC150 `90.99±0.78`、StackOverflow `89.96±1.50`、BANKING77-OOS `91.89±0.19`；历史 Ours artifact 的实际数据键也是 `banking77_oos`，相对差值为 `−0.97/+0.25/+3.66 pp`；std 使用 ddof=0。
- [自适应中心汇报](analysis/historical_trainable_adaptive_centers_presentation.md)及其结果目录保留 fixed/adaptive/overall 选择、逐 seed 配置、ranking oracle 和 intent 错误。27 条 selected 行（9 单元×3 选择范围）的 direct pipeline 字段已核对，CSV 布尔汇总回归测试通过。
- **OOS 判决一致不代表 pipeline 整体无损。** 原始 K=1 九单元逐样本 CUDA 重算已完成，OOS 判决全部一致。CLINC/SO/Banking 的 macro F1 平均净下降 `3.96/2.43/3.77 pp`，Accuracy 平均净下降 `1.45/1.20/0.74 pp`。CSV 分列 Router/Expert 错误、修复/新增错误数、Known 接受率和 OOS 误接收。当前重放与历史 macro 聚合最大差异 `0.183 pp`，两者分别保留，不声称精确复现。
- CLINC 自适应已选 score 的三 seed test threshold oracle 均值为 `91.22%`，触发已完成的 checkpoint follow-up。九个训练单元（81 条 CUDA epoch 记录）及三个真实全链路确认均完成；复用既有 last2_long/seed13，未重训它。验证集锁定 seed13 last2/epoch7、seed42 last2/epoch9、seed87 LoRA/epoch9 后才读取 test。
- [Checkpoint 与 full-pipeline 汇报](analysis/historical_trainable_checkpoint_selection_presentation.md)：CLINC OOS F1 为 **91.64±0.27%**，相对论文 `−0.32 pp`；逐 seed `91.37/91.56/92.01%`，仅 1/3 超过。新固定 score test oracle 均值 `91.78%`，阈值余量仅 `.13 pp`，仍不足以关闭差距。当前表示训练仍不足，不再扩大同一阈值网格。
- 新 checkpoint 的 Gate/pipeline macro F1 均值为 `87.49/81.92%`，Accuracy 为 `89.77/87.58%`。下游净损失分别为 `5.57/2.19 pp`；OOS 判决逐样本一致。LoRA 可训练参数 `535,936`，last2 为 `3,746,944`，projection-only 为 `198,016`；LoRA/projection 的原始骨干可训练参数均为 0。
- 新增 KIR=.25 matched full pipeline 验证：使用既有 CLINC Trainable K=1 Gate，并重新训练同一 KIR=.25 的 3 个 Router + 30 个 Expert；所有组件审计 ready、CUDA 训练和 CUDA 推理均有日志。三 seed OOS F1 为 `95.40/95.21/94.68%`，均值 `95.10±0.30%`，相对 fulltex.tex KIR=.25 Ours `95.01%` 为 `+0.09 pp`，2/3 seed 超过。结果见 [KIR=.25 full pipeline 汇报](analysis/historical_trainable_kir25_full_pipeline_presentation.md)。
- KIR=.25 full pipeline 的 macro F1 为 `75.43±1.68%`、Accuracy 为 `91.20±0.26%`；OOS F1 的提升不能概括为所有指标全面超过。Router error `2.05%`、Expert error `3.76%`，Gate 与 pipeline 的 OOS 判决逐单元一致。
- KIR=.50 新增 geometry search 将 full-pipeline OOS F1 提升到 `91.79±0.43%`；新增 Known-only recipe search（12 个 CUDA 训练单元）进一步达到 `91.93±0.48%`，相对论文 `91.96%` 仍差 `0.03 pp`。当前最优 seed 为 `91.55/91.64/92.61%`，没有稳定超过论文。
- 本轮结果见 [Gate 搜索汇报](analysis/historical_trainable_gate_search_presentation.md)。旧 threshold 网格不再扩展；下一阶段若继续，必须改变 support score 或训练表示目标，并预先定义独立验证规则。
- OOS-first boundary search 已完成：validation 只按 OOS F1 选参，允许 Known 指标下降。BANKING77-OOS KIR=.25 达到 `95.95±0.48%`，相对历史 Ours `93.99%` 为 `+1.96 pp`；Known Recall 降至约 `61.53%`。KIR=.50 为 `91.73±0.08%`，KIR=.75 为 `88.56±0.48%`。KIR=.50 的旧“均衡”结果为 `91.53±0.05%` OOS F1、`75.84%` Known F1、`85.59%` Accuracy；OOS-first 只增加约 `+0.20 pp` OOS F1，但 Known F1 下降约 `4.54 pp`。结果见 [OOS-first 搜索汇报](analysis/historical_trainable_oos_priority_presentation.md) 及其 [均衡对照汇总](../results/analysis/historical_trainable_oos_priority_search/balanced_vs_oos_priority_kir50.csv)。
- 当前 H1 Trainable full pipeline 的 KIR=.25 与 KIR=.75 行已补齐为 `3 数据集 × 3 seed`：KIR=.25 的 OOS F1 为 CLINC `95.10±0.30`、StackOverflow `95.38±1.02`、BANKING77-OOS `92.58±0.89`；KIR=.75 为 CLINC `80.80±0.66`、StackOverflow `75.16±2.36`、BANKING77-OOS `86.60±0.42`。完整表见[主实验与论文消融报告](analysis/historical_paper_ablation_report.md)，Banking 行与历史 Ours 使用同一 `banking77_oos` 数据键，但方法合同不同。
- 新增[完整主实验与论文设置消融报告](analysis/historical_paper_ablation_report.md)：论文设置的 `3 数据集 × 3 KIR × 4 变体 = 36/36` 个历史 anchor 单元已完成审计；当前 H1 Trainable full pipeline 也按 KIR 汇总。两类结果分开标记，历史消融不冒充本轮 CUDA H1 重跑。
- 所有上述结果属于 `historical_v19_paper_main` 的 **H1 controlled**；`fulltex.tex` Ours 是历史数值参考，不是同模型链严格 H0 对照。

## Banking 数据集命名锁定

| 实验线 | 实际数据键 | 报告名称 | 用途 |
|---|---|---|---|
| 当前 H1 / 历史 Ours artifact | `banking77_oos` | `BANKING77-OOS` 或论文表头 `Banking77` | 当前 Trainable/Frozen、历史 Ours、full pipeline 与机制图 |
| archive / protocol_v2 参考 | `banking77` | `Banking77` | archive Gate→Router→Expert、MOGB-Fair 和 current fair matrix |
| archive full pipeline / 当前 protocol_v2 参考 | `banking77` | `Banking77` | 本轮 archive Gate→Router→Expert、MOGB-Fair 和 current fair matrix |

后续文档和图注必须使用实际数据键；不能仅写裸 `Banking77` 来跨两条实验线比较。

## 先看哪里

1. [近期机制分析汇报](analysis/RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md)：直接用于组会或阶段汇报，含图片粘贴位置和读图说明。
2. [统一对比与机制报告](analysis/UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md)：查完整结论、合同和证据边界。
3. [旧协议与历史结果对账](analysis/HISTORICAL_PROTOCOL_RECONCILIATION_V1.md)：查清 H0/H1 历史数据链、78.64% 来源、旧协议 Trainable 结果和 TextOIR 基线闭合度。
4. [关键图索引](analysis/VISUAL_ANALYSIS_INDEX_V1.md)：按问题定位图和源表。
5. 只有需要查实现、实验合同或复现命令时，才看 [METHOD.md](METHOD.md)、[EXPERIMENTS.md](EXPERIMENTS.md) 和 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

`docs/archive/` 只保存历史、失败、外部兼容和被合并的报告，不是当前事实入口。

## 目前进展

- `protocol_v2_textoir_v1` 的 fair 主矩阵已闭合：3 个数据集 × 3 个 KIR × 5 个 seed；该结果冻结为参考，不作为新实验默认。
- Frozen 有两个必须分开的基线：`Frozen single-centroid（MOGB-Fair；Euclidean）` 的 `78.64%` 是组件行；`Frozen E2 K=1` 的对角 Mahalanobis/`mean+std` 均值为 `81.07%`，后者才用于判断 Trainable 的同几何训练收益。
- 统一逐样本 prediction contract 已建立，并完成 sample、split、Known list、hash 和监督层级审计。
- 机制图包当前以历史 H1 OOS 解释 bundle 为阅读入口：局部边界、score 分布、配对 crossing、distance/radius 分解、跨数据集密度、margin 难度、KIR/seed 稳定性和 OOS subtype 残余风险共八张图；Known Recall/false reject 只作 coverage guard，Known intent 分类图仍不进入主线。
- ADB/DA-ADB 已有可审计外部参照和样本级错误机制包；本轮又补齐了 ADB 原生 BERT 表示、最近中心距离、学习半径和边界风险图，但它们仍属于 BERT/TextOIR external 合同。DCLOOS 仍按 pseudo-OOS/external-OOS 单独报告。
- KNNCL、OpenMax、DeepUnk 等未闭合路线仍是 blocked/historical，不使用中间预测补齐排名；DOC 的 native compatibility 已有完整 3×3×3 结果。
- 文档入口已收缩：分析层只保留统一报告和图索引；原来的性能总览全文已移至 `docs/archive/analysis/`。
- 数据协议决定已更新：后续新实验统一回到 `assets/datasets/s2c/prepared/data/multidataset/v19` 的历史主协议；`protocol_v2_textoir_v1` 只保留已有参考结果。
- 历史对账已确认：论文后部受控 K 表已复现；H1 旧协议上的 Trainable K=1 在 CLINC、StackOverflow、BANKING77-OOS 的 KIR=.50 seed=42 均超过 Frozen K=1。严格 H0 的 `data/v19` 原始目录和部分旧主 Cascade 输入仍未逐字恢复，不能把 H1 重建快照称为 H0 重跑。
- 补充判因实验已完成：H1 v19 的 3 个数据集×3 个 KIR×3 个 seed 共 27 个 Frozen/Trainable 配对单元，9 个 OOS F1 差值均为正；KIR=.50 的均值提升为 CLINC +1.47 pp、StackOverflow +9.76 pp、BANKING77-OOS +3.65 pp。另完成 KIR=.50、3 seed 的旧 v19 Frozen 与当前 protocol_v2 Frozen 输入协议对照。
- OOS 可视化机制 bundle 已完成：原有 OOS 机制图加 3D Fig.1 均由既有 H1 artifacts 的聚合 post-hoc 分析生成；重编码后的 StackOverflow seed42 score 与历史 prediction 的决策和最近中心均 0 mismatch，精确分解闭合误差小于 5.6e-17，Python 图源预检 14/14 PASS，视觉 verdict 91/100。
- Fig.1 已补充 3D 主图：使用 Frozen/Trainable 共同 PCA、半透明 score=1 椭球，并用真实 384 维判决区分 OOS rejected 与 false acceptance。StackOverflow/KIR=.50/seed=42 的局部 cohort 中，Frozen 为 `361 accepted / 2445 rejected`，Trainable 为 `27 accepted / 2779 rejected`，修复 `349` 个 Frozen 误接收；二维图只保留为补充，不能按投影位置读取判决。
- 跨方法 OOS 比较 bundle 已生成：标准 `banking77`/CLINC150/StackOverflow 的 MiniLM fair、TextOIR native BERT compatibility、ADB/DA-ADB 外部工作点、MOGB ball 风险和 DCLOOS 证据状态共 9 张图；入口为 `docs/analysis/cross_method_oos_mechanism_presentation.md`，不把不同合同合成一个 SOTA 排名。
- 当前 H1 Trainable full pipeline 已补齐：使用既有 `last2_minilm_plus_projection` checkpoint 和 `cascade_full/gpu_kir50` 的 Router/Expert，在 CLINC150、StackOverflow、BANKING77-OOS 各 3 个 seed 上完成 9 个显式 CUDA 推理单元。相对 Frozen K=1，full-pipeline macro F1 均值分别提升 +2.08、+7.89、+3.47 pp；OOS false acceptance 分别下降 2.94、16.43、6.73 pp。结果见 `results/analysis/historical_trainable_full_pipeline/`，本次未重新训练 MiniLM。
- Trainable Gate 的参数化 full pipeline 搜索已完成：在历史 H1 KIR=.50、3 个 seed 下评估 K∈{1,2,3}、6 个 λ、7 个 threshold 和两种 acceptance mode，共 2268 个测试候选；验证集选择只使用 OOS 标签和 Known F1/Accuracy 1 pp guard。selected 结果为 CLINC150 `91.04±0.81`、StackOverflow `89.15±1.54`、BANKING77-OOS `91.53±0.05` OOS F1；其中 BANKING77-OOS 比 `fulltex.tex` KIR=.50 Ours `88.23` 高 `+3.30 pp`，三个 seed 均超过。结果与汇报见 `results/analysis/historical_trainable_parameter_full_pipeline/` 和 `analysis/historical_trainable_parameter_full_pipeline_presentation.md`。
- archive 全链路对照已完成：在 `../archives/submissions/s2c-submission/data` 的标准 CLINC150、StackOverflow、Banking77 上，以 `seed=42`、`KIR=.50`、`K=1`、对角 Mahalanobis、`mean+std` 和关闭 semantic gate 固定协议，完成 Frozen/Partial/LoRA 三种 Gate 加 archive Router/Expert 的 9 个 full-pipeline 单元。Partial 相对 Frozen 的 OOS F1 提升为 `+1.40/+7.14/+4.01 pp`；LoRA 相对 Partial 为 `−0.80/−0.58/+0.51 pp`。结果见 `results/analysis/historical_archive_full_pipeline/` 和[完整 pipeline 阶段汇报](analysis/historical_pipeline_experiment_presentation.md)。
- archive 下游重训已确认使用 CUDA：CLINC150 的 10 个 Expert、StackOverflow 的 6 个 Expert、标准 Banking77 的 38 类 Expert 以及三个 Router 均有 `Device: cuda` 日志；三套 full pipeline manifest 均记录 `device=cuda`。LoRA Gate 的原始 MiniLM 参数 `22,713,216` 个全部冻结，可训练参数为 `535,936`。
- OOS 优先的 workpoint 搜索已完成：在验证集选择边界候选后再做测试确认，CLINC150 Partial `K=3, λ=2, normalized_union` 的 full-pipeline OOS F1 为 `91.36%`，StackOverflow Partial `K=1, λ=2, threshold=.85` 为 `91.85%`，标准 Banking77 Partial `K=2, λ=1, threshold=.95` 为 `85.62%`。对应 full-pipeline Gate replay 最大差异均为 `0`；标准 Banking77 的剩余论文差距来自任务协议不同，匹配 `banking77_oos` 的已有 H1 Trainable 三 seed 均值为 `88.47%`。
- 旧协议代码与数据已定位：`scripts/data/active/rebuild_multi_dataset_v19.py`、`scripts/data/active/rebuild_v19_2_strict.py`、`../archives/submissions/s2c-submission/`；论文 H0 结果锚点位于 `../artifacts/s2c/outputs/paper_results/`，其 CLINC 预测已用于重建并挂载 `data/v19`（75 个 Known、5499 条 Gate test）。该目录明确标注为锚点重建，精确 H0 原始快照与模型链仍缺失。旧快照的结构约束正确，但原始数据存在跨 split 重复，论文的 Banking77 命名和 6:1:3 表述也与实际输入不完全一致。
- 复现强度已分层：raw source 与 TEXTOIR commit 字节一致，可用于上游 `DataManager` 严格复现；MSP/DOC/ADB native compatibility 已完成 3×3×3，DA-ADB 目前为 StackOverflow 三 seed，KNNCL 仍 blocked。不能把 compatibility 行写成 byte-identical 论文复现。

## 当前结论

1. `S2C-Trainable-K1` 是当前 Known-only MiniLM Gate 层最平衡的自有工作点：fair 矩阵均值 OOS F1 为 85.75%，AUROC 为 93.12%，OOS AUPR 为 89.70%，false acceptance 为 8.73%。
2. Trainable 的主要收益是表示适配后的 score separation/order 改善，不是单纯移动阈值。
3. 相对 MOGB，S2C 的主要 OOS 工作点收益是更低的 false acceptance 与更高的 coverage 平衡；MOGB 的低 false acceptance 伴随明显 Known under-coverage。
4. K=2 的接受区域并集会放大 Near-OOS 风险，且方向依赖数据集；不能从表示分离改善推出多中心边界一定更安全。
5. 在 StackOverflow/KIR=.50/seed42 的 Euclidean MOGB-Fair 组件空间回放中，K=2 比 K=1 额外接受 31.60% 的 OOS；MOGB 相比该 Frozen single 组件移除 55.67% 的 Known 接受样本。这分别直接对应“并集过覆盖”和“ball 覆盖不足”。
6. 在相同 Trainable MiniLM 表示上，MSP/Energy 的 OOS score rank 与 S2C 的 Spearman 相关只有 0.54/0.57，S2C-only 正确比例为 54.9%/59.5%；kNN/LOF 排序更接近，但仍有 29.8%/38.9% 的 S2C-only OOS 正确样本。差异来自 detector 的排序与边界规则，不只是 backbone。
7. 当前不能声称超过完整 MOGB、DCLOOS、ADB 或历史 fulltex SOTA；这些结果的 backbone、监督、数据合同或系统层级不同。
8. 在 StackOverflow/KIR=.50 的三 seed 外部错误构成中，S2C 的 Known Recall 为 `83.68%`、OOS false acceptance 为 `8.18%`；ADB 为 `80.78%`、`7.52%`，表现为略保守但覆盖更低；DA-ADB 为 `75.97%`、`29.07%`，主要代价是 Known 拒绝和 OOS 误接收同时增加。该图层解释了“差在哪里”，但不能把差异归因成单一 backbone 或损失。
9. 新 H1 OOS 机制图显示：KIR=.50 三 seed 的 `2,891` 个 OOS 由 Frozen 误接收转为 Trainable 正确拒识，反向退化 `506` 个；修复样本的 score 增益主要来自距离 step（CLINC +0.144、StackOverflow +0.426、BANKING77-OOS +0.073），而不是统一扩大半径。StackOverflow 的 `cocoa`/`hibernate` 仍是残余 subtype 风险。
10. H1 full pipeline 的当前结论是：Trainable Gate 的 OOS 改善能够传递到最终 macro F1 和 Accuracy，但下游仍有明确上限；Trainable 端 Expert error 均值约为 CLINC 3.45%、StackOverflow 5.77%、BANKING77-OOS 11.51%。此前历史 Trainable checkpoint 的 legacy manifest 未记录实际训练 device，因此其 GPU 训练 provenance 仍不完整；当前 full pipeline 推理的 device 已明确记录为 CUDA。
11. archive seed42 的原始 K=1 对照确认：训练后的 MiniLM 做 Gate 的方向成立；未调边界时 Partial 是 CLINC150/StackOverflow 最佳，LoRA 只在标准 Banking77 小幅领先。LoRA 不满足三数据集均不下降条件，不作为统一默认。
12. archive 标准 `banking77` 线与历史 H1 `banking77_oos` 不能合并；历史 `93.99` Ours artifact 已在 `paper_results` 中确认。当前没有证据表明 Gate→pipeline 存在 OOS 指标传递 bug。
13. 最新 H1 参数搜索的 test confirmation 表明：CLINC150 和 StackOverflow 当前网格没有超过论文 Ours 的三-seed 均值；BANKING77-OOS 有 `110/252` 个候选超过论文，测试后验最高为 `91.73%`，验证集选中的主结果为 `91.53%`。两者均保留，但后验最高点不作为选参结论。

## 数据协议与 Frozen/Trainable 判断

- `Frozen E2 K=1` 是冻结参考协议下的 Gate-only 基线，不是旧论文完整 `Ours`；旧方法还包含 `K_y=2` 的多中心 Gate、SmolLM Router/Expert 和 Cascade。78.64% 的 `Frozen single-centroid` 是另一条 Euclidean MOGB-Fair 组件行。
- 后续新实验统一使用 `historical_v19_paper_main`：CLINC150、StackOverflow、BANKING77-OOS 的旧数据根、旧 Known 列表和旧 OOS 构造；当前 `protocol_v2_textoir_v1` 不再新开实验。
- 本轮用户指定的 archive 控制线单独使用 `historical_v19_archive`：标准 `banking77` 与 archive 的 Known/OOS 构造保持一致，不与 `banking77_oos` 混用；它用于验证 Gate/下游链路，不替代论文主表。
- 在 canonical E2 Frozen K=1 的同几何配对中，45 个 seed 单元有 44 个 OOS F1 提升，9 个 dataset×KIR 组均值全部为正：Trainable/Frozen 为 `85.75% vs 81.07%`；主收益是 OOS score separation/order，Known Recall 仅作为 coverage guard。
- OOS-only 主视觉保留 score/rank、OOS precision/recall/F1、AUROC/AUPR、false acceptance 和 OOS 状态转移；Known intent 分类图不再进入主汇报。

## 关键机制图（按协议分层）

后续历史 v19 主线首先使用这张同协议对账图：

1. **历史协议 H1 OOS 可视化机制 bundle**：[`RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md`](analysis/RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md) 直接嵌入九张图，分别回答局部边界、score 分布、样本 crossing、distance/radius 分解、跨数据集密度、margin 难度、KIR/seed 稳定性、残余 subtype 风险和表示位移；机器清单为 [`MANIFEST.json`](../results/analysis/historical_oos_visual_explanation/MANIFEST.json)。

2. **历史协议 v3 supporting 图**：[`main_oos_performance.png`](../figures/historical_protocol_oos_v3/main_oos_performance.png) 和 [`main_oos_mechanism.png`](../figures/historical_protocol_oos_v3/main_oos_mechanism.png)，
   保留作为上一版结果与机制汇总，不再作为当前主汇报入口。

3. **历史协议补充图组**：[`supp_training_dynamics.png`](../figures/historical_protocol_oos_v3/supp_training_dynamics.png)、[`supp_distance_radius_score.png`](../figures/historical_protocol_oos_v3/supp_distance_radius_score.png)、[`supp_oos_geometry.png`](../figures/historical_protocol_oos_v3/supp_oos_geometry.png)。

冻结的 `protocol_v2_textoir_v1` 参考图仍保留三类 OOS 低阅读成本版式，版式合同见
[关键图索引](analysis/VISUAL_ANALYSIS_INDEX_V1.md)：

3. **OOS score 分布（protocol_v2 冻结参考）**：[`oos_score_distribution_kir050.png`](../figures/oos_readability_figures_v1/oos_score_distribution_kir050.png)，
   1×3 小面板、OOS 强调色、Known 淡色参照和明确 `score=1` 阈值线。
4. **OOS 工作点（protocol_v2 冻结参考）**：[`oos_operating_point_kir.png`](../figures/oos_readability_figures_v1/oos_operating_point_kir.png)，
   只展示 OOS F1 与 false acceptance，Known Recall 仍作为单一 coverage guard。
5. **OOS 机制（protocol_v2 冻结参考）**：[`oos_mechanism_transitions_kir050.png`](../figures/oos_readability_figures_v1/oos_mechanism_transitions_kir050.png)，
   展示 OOS 状态转移率与 `mean Δscore`；distance/radius 细分放在补充证据中。

上面三张 v2 图的 `Banking77` 是标准 `banking77` 键，不是历史主线的 `banking77_oos`，不能用于
解释论文主表或与历史 v19 图直接拼接。它们的源表和可读性合同见
`results/analysis/oos_readability_figures_v1/MANIFEST.json`；旧的
`figures/archive/analysis/minilm_boundary_diagnostics_v1/` 仅作为版式参考和历史证据。

`figures/deep_geometry_mechanism_v1/`、`figures/mechanism_explanation_v1/`、
`figures/adb_deep_mechanism_v1/` 和 `figures/da_adb_current_protocol_summary_v1/` 的其余图件
全部保留为补充证据库：只有在回答一个明确的 OOS 追问时才单独调用，不再作为主汇报的连续图墙。

## 当前限制

- 深层图在运行时重建 Trainable embedding，raw array 不写入轻量结果；重建依据、逐行回放和 K=2/native detector 精确复现记录在 `results/analysis/deep_geometry_mechanism_v1/reconstruction_audit.json`。
- PCA 边界图是解释性投影，不替代原始 384 维判决；所有 OOS 机制图均是事后解释，不是新的判决器。
- ball 风险图和错误转移图使用测试标签做事后归因，不参与训练、阈值或结构选择；它们解释的是错误发生在哪里，不是新的性能工作点。
- 外部 baseline 的预测与当前 protocol_v2 测试文本/标签顺序已逐 seed 核对。ADB 现在有单独的 native distance/radius 分析，但 ADB/DA-ADB 运行 manifest 仍没有完整声明 test-selection 字段；DA-ADB 没有稳定可复用的 native score/embedding。因此外部图只按合同解释 OOS 错误构成，不声称 MiniLM 与 BERT 的差异来自单一内部因果因素。
- 外部 baseline 仍按合同分层；没有 final metrics 的方法不进入主表。历史 H0/H1 结果见[旧协议对账报告](analysis/HISTORICAL_PROTOCOL_RECONCILIATION_V1.md)，不回填当前 fair 表。

## 资产入口

- 注册关系：`configs/experiment_registry.yaml`
- 当前轻量结果：`results/analysis/`
- archive full pipeline 结果：`results/analysis/historical_archive_full_pipeline/`；汇报：`docs/analysis/historical_pipeline_experiment_presentation.md`
- archive OOS 调参对论文对比：`results/analysis/historical_archive_tuned_full_pipeline/`；候选 artifact：`../artifacts/s2c/runs/historical_archive_tuned_full_pipeline_seed42/`
- 历史 H1 Trainable Gate 参数化 full pipeline：`results/analysis/historical_trainable_parameter_full_pipeline/`；汇报：`analysis/historical_trainable_parameter_full_pipeline_presentation.md`；图：`figures/historical_trainable_parameter_full_pipeline/`
- 当前图：`figures/`
- 历史结果和报告：`results/analysis/archive/analysis/`、`figures/archive/analysis/`、`docs/archive/`
- 原始运行和逐样本 artifact：`../artifacts/s2c/`，不进入公开轻量结果

## 当前唯一下一步

当前最有价值的 CLINC 结果是 KIR=.25 的 matched full pipeline；KIR=.50 经过 geometry 和 Known-only recipe 搜索后为 `91.93±0.48%`，仍低 `0.03 pp`。后续应把 KIR=.25/.50/.75 作为明确实验条件汇报；若继续创新，应改变 support score 或表示目标并定义独立验证方案。archive `banking77` 与历史 H1 `banking77_oos` 分开，不改写 `fulltex.tex`。
