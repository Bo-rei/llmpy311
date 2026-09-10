# S2C 统一对比与机制报告

> 当前唯一的综合分析入口。它只使用已落盘、可追溯的结果；不同骨干、监督条件、数据划分或系统层级不混成一个排名。
>
> 直接汇报请使用：[近期机制分析汇报](RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md)。

更新时间：2026-08-12
分析对象：`protocol_v2_textoir_v1`（冻结参考，不是后续新实验默认）
当前主方法：**Known-only Trainable MiniLM + K=1 单中心 Gate**

> 旧论文协议与当前协议的逐项对账、归档 Gate 复跑以及旧协议 Frozen/Trainable 控制结果，现以
> [`HISTORICAL_PROTOCOL_RECONCILIATION_V1.md`](HISTORICAL_PROTOCOL_RECONCILIATION_V1.md) 为准。
> 本文保留当前协议的机制分析；其中的 `78.64%` 只表示 Euclidean 单中心组件，不表示旧论文完整方法。

## 1. 先看结论

在统一 Known-only MiniLM Gate 合同下，`S2C-Trainable-K1` 是当前自有方法中最平衡的工作点：它同时改善了 Known coverage 和 OOS score ordering；固定多中心的主要风险是接受区域并集过覆盖；MOGB-MiniLM 的低误接收主要以严重 Known 拒绝为代价。

这不是“所有指标都超过所有 baseline”的结论：ADB/DA-ADB 是 BERT/TextOIR 外部合同，DCLOOS 使用 pseudo-OOS/external-OOS，历史 Cascade 是完整系统，不能与当前 Gate-only fair 行直接排名。

## 2. 结果合同与比较层

统一 prediction contract 位于 `results/analysis/unified_prediction_contract_v1/`，本地逐样本压缩 JSONL 位于 `../artifacts/s2c/analysis/unified_prediction_contract_v1/`。它包含 `2,394,360` 行、`486` 个运行组；对齐审计中 `441` 个方法-cell 完全匹配参考样本序列，`45` 个外部或失败对齐单元被保留为非 fair 证据。

| 层级 | 可比较对象 | 用途 |
|---|---|---|
| same-protocol fair | Trainable K=1、Frozen E2 K=1、Euclidean MOGB-Fair 组件、Random K=2、边界交换 | 当前主结论：表示、中心、半径和 coverage–rejection |
| native detector | 同一 Trainable MiniLM 上的 Gate、MSP、Energy、kNN、LOF | 区分表示收益和检测规则收益 |
| external backbone | ADB、DA-ADB、官方 MOGB 兼容单元 | 同源数据参照，不进入 MiniLM fair 排名 |
| different supervision | DCLOOS pseudo-OOS / external-OOS | 监督强度与成本参考 |
| historical system-level | 历史完整 Cascade 与 TextOIR 表 | 背景，不替代当前 Gate-only 结果 |
| blocked/incomplete | KNNCL、OpenMax、DOC、DeepUnk 等未闭合路线 | 记录缺口，不填补中间预测 |

### 2.1 当前 fair 主矩阵

来源：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`，3 个数据集 × 3 个 KIR × 5 个 seed。

| 方法 | OOS F1 | AUROC | AUPR (OOS) | False acceptance | Known Recall (guard) | 机制含义 |
|---|---:|---:|---:|---:|---:|---|
| **S2C Trainable K=1** | **85.75%** | **93.12%** | **89.70%** | **8.73%** | 80.22% | 当前自有矩阵最平衡 |
| Frozen single-centroid（MOGB-Fair component；Euclidean） | 78.64% | 89.37% | 84.31% | 23.01% | 83.50% | 冻结表示组件基线 |
| Random K=2 | 78.39% | 89.53% | 84.55% | 23.86% | 84.25% | 多中心但无语义适配 |
| Frozen K=2（MOGB-Fair component；Euclidean） | 75.51% | 87.36% | 81.39% | 25.81% | 81.44% | 接受区域并集风险 |
| MOGB partition + S2C boundary | 77.74% | 87.35% | 80.32% | 2.65% | 52.21% | 误接收较低但 coverage guard 较低 |
| S2C partition + MOGB boundary | 75.90% | 82.98% | 76.86% | 7.35% | 49.98% | MOGB 边界工作点偏保守 |
| MOGB-MiniLM | 73.39% | 85.19% | 77.69% | 0.90% | 31.21% | 低误接收换取低 coverage guard |

这里固定命名：`Frozen single-centroid（MOGB-Fair component；Euclidean）` 的内部 method id 是
`single_centroid`，不能简称为 `Frozen K=1`；后者只保留给 canonical E2 的对角
Mahalanobis/`mean+std` Gate。两者都使用冻结 MiniLM，但距离、边界和来源运行不同，78.64% 与
canonical E2 Frozen K=1 的 81.07% 不是同一行。

这些数字只支持“当前 fair Gate 层的工作点比较”，不支持超过完整 MOGB、DCLOOS 或历史 SOTA 的声明。

### 2.3 旧 fulltex 协议与当前协议不是同一个 benchmark

旧论文写的是 6:1:3 数据划分和 22500/20000/13083 的数据规模
（[`fulltex.tex:277-303`](../../fulltex.tex:277)），但旧 v19 manifest 已经显示出另一套实际运行合同：
CLINC150 的 Gate 为 `7500/3100/5500`，StackOverflow 为 `5995/1998/5990`，
而 `BANKING77-OOS` 为 `2933/2236/4080`。当前活动快照则固定为 TEXTOIR commit
`dffe2b1b848a069a6808f8089b4cb9bd16e2062b`，原始划分为 CLINC150 `15000/3000/5700`、
Banking77 `9003/1000/3080`、StackOverflow `12000/2000/6000`
（[`protocol_v2_textoir_v1.yaml:1-41`](../../configs/data/protocol_v2_textoir_v1.yaml:1)，
[`data manifests`](../../data/manifests/protocol_v2_textoir_v1/)）。

| 维度 | 旧 fulltex/v19 | 当前 `protocol_v2_textoir_v1` | 影响 |
|---|---|---|---|
| Known list | CLINC 用 `domain_balanced_largest_remainder`；StackOverflow 用旧 title 快照的 seeded random；Banking 使用 50 个 in-domain intent 的 `BANKING77-OOS` | 统一按固定 TEXTOIR label order、`RandomState(seed).choice`、`round(label_count*KIR)`；Banking 是完整 77 intent | 同一个 KIR 不代表同一组 Known 类 |
| KIR 与 seed | 历史主表使用 KIR `.25/.50/.75`，旧 Gate/重建流程的 seed 与选择器不统一 | 正式 registry 固定 seed `13/42/87/100/123`，正式 KIR 为 `.10/.20/.25/.30/.40/.50/.60/.70/.75/.80/.90`；当前主矩阵报告 `.25/.50/.75` | 结果单元的随机性和 Known 类集合不同 |
| OOS 构成 | CLINC 为 held-out + native OOS；StackOverflow 为旧快照的 held-out；Banking 为 held-out + `id_oos` + `ood_oos` | 所有数据集先做 held-out intent OOS；CLINC 另保留 native `oos`，Banking/StackOverflow 无 native OOS | OOS 难度、数量和边界分布都变了 |
| 文本/样本 | StackOverflow 使用旧 deduplicated title 快照；Banking 旧论文表对应 OOS 扩展路线 | 使用固定 TEXTOIR 导出的当前样本和 sample-id/hash | 逐样本不能直接对齐 |
| 系统层级 | 论文 `Ours` 是冻结 MiniLM Gate + SmolLM Router/Expert 的完整 Cascade；Gate 默认 `K_y=2` | 当前主方法是 Known-only Trainable MiniLM + K=1 Gate；Frozen 行是 Gate-only control | 不能把 Gate-only 行当作旧系统结果 |
| 选择/校准 | 论文记载 CLINC `lambda=.5`、其余 `lambda=1`，并明确写过用 OOS 样本帮助学习 `lambda` | 只用 Known train/calibration；不使用 test OOS 选结构、阈值或 checkpoint | 旧结果存在监督和选择合同差异 |
| 评价口径 | Known F1、OOS F1、Acc；旧表是完整 Cascade 的系统级结果 | OOS F1、AUROC、AUPR、false acceptance 为主；Known Recall/false reject 只作 coverage guard；主 fair 行是 Gate-only | 指标名称相同也不等于评价对象相同 |

因此，旧结果无法复现不是单一预处理错误，而是数据源、Known list、OOS 构成、系统层级和校准监督同时改变。旧论文的具体定义见
[`fulltex.tex:223-275`](../../fulltex.tex:223)、[`fulltex.tex:329`](../../fulltex.tex:329) 和
[`fulltex.tex:426-455`](../../fulltex.tex:426)；当前运行规则见
[`docs/REPRODUCIBILITY.md:3-17`](../REPRODUCIBILITY.md:3)。

#### 2.3.1 当前快照能复现什么

当前三套 raw `train/dev/test.tsv` 与 TEXTOIR commit
`dffe2b1b848a069a6808f8089b4cb9bd16e2062b` 已完成字节级核对；上游方法注册和运行合同也已
审计到 MSP、DOC、ADB、OpenMax、KNNCL、DA-ADB。因而当前数据源是合理的 TEXTOIR 同源输入。
但现有 `textoir/adb/da_adb` exports 是为了统一当前 Known registry 而构造的有效视图：
train/dev 只保留 Known，test 保留相同样本集合但按 Known→OOS 分组。已有 ADB 等结果因此是
“当前协议下的 BERT/TextOIR compatibility”，不是 byte-identical 的上游论文复现；adapter 的
统一 OOS 字面量 `oos` 还会在非 CLINC 上被运行器映射为上游的 `<UNK>` unknown id。

推荐的使用方式是：用已核验的 `../textoir/data` + 上游 `DataManager` 做严格方法复现；用同一
registry 的 method export 做当前 S2C fair 对照。canonical source 的目录名已标准化为
`clinc150/banking77/stackoverflow`，不能未经映射直接作为上游 runner 的 `oos/banking/`
目录。若单独调试一个最小 cell，选 StackOverflow/KIR=.50；
正式表仍回到三数据集和固定正式 seeds。没有 final metrics 的 KNNCL、OpenMax、DOC 等不应
因为输入快照已经对齐就被视为“已复现”。

### 2.4 78.64% 的 Frozen single-centroid 与旧方法的关系

先直接回答：**不是旧论文的完整方法。** 用户表中的 78.64% 是当前
`cross_protocol_tradeoff_v1` 中的 `single_centroid` 行，即冻结 MiniLM、Euclidean、单中心、
`mean+std` 边界的 MOGB-Fair 组件；它不是旧论文完整的 `Ours`，也不是 canonical E2 Frozen K=1。

旧论文真正的 `Ours` 还包含每个 intent 的多中心 Gate、SmolLM Router/Expert 和完整 Cascade。
旧论文后面受控补充表确实报告过一个“冻结 MiniLM、Gate-only、K=1”的结果，但那一行使用旧
data/split、旧 Known list、旧 seed/校准合同（例如 StackOverflow 对角 Mahalanobis K=1 为
`79.02±4.63`），不能把它与当前 78.64% 或 E2 81.07% 互换。

因此，当前报告固定使用三种名字：

- **Frozen E2 K=1**：当前协议、对角 Mahalanobis、`mean+std`，用于和 Trainable 做同几何配对；9 个 dataset×KIR 单元均值为 81.07%。
- **Frozen single-centroid（MOGB-Fair component；Euclidean）**：当前公平矩阵的组件行，均值为 78.64%。
- **历史完整 Ours**：旧 fulltex/v19 的 Gate→Router→Expert Cascade，只进入 historical lane。

即使把当前 `Frozen K=2` 作为更接近的 Gate 组件，也不能恢复旧数据、旧 Known list、旧
Banking77-OOS 以及旧校准合同。

如果目标是复现旧表，应把旧 v19 数据和完整 Cascade 作为独立的 historical reproduction lane；如果目标是复现 TEXTOIR 中的其他方法并形成公平比较，继续使用当前固定 TEXTOIR 快照，不要把两个 lane 合并成一个主表。

### 2.5 Frozen 与 Trainable：先回答“训练是否有效”

在同一当前协议、同一 `dataset×KIR×seed`、同一 K=1 Gate 和同一对角 Mahalanobis/`mean+std` 边界下，canonical E2 Frozen K=1 与 Trainable 的 45 个 seed 配对中有 44 个 OOS F1 提升；按 9 个 dataset×KIR 组聚合后，9/9 组均为正。五 seed 的 9 组均值为：OOS F1 `81.07% → 85.75%`（`+4.69pp`）、AUROC `89.98% → 93.12%`、false acceptance `17.62% → 8.73%`；Known Recall 为 `81.13% → 80.22%`。因此训练收益不是来自换成更宽松或更激进的检测器；此前表中的 `78.64%` 是 Euclidean MOGB-Fair 组件行，不能当作 Frozen E2 K=1。这里的“同一”指 protocol、样本序列、K、距离、边界和评价口径一致；运行 manifest 没有独立记录 evaluator 代码 hash，所以不延伸为实现级完全同一的声明。源表为 `results/analysis/archive/analysis/minilm_trainable_5seed_fair_v1/frozen_e2_per_seed.csv`、`trainable_per_seed.csv` 和 `trainable_vs_frozen_paired.csv`。

现有 OOS 机制证据支持以下解释：StackOverflow/KIR=.50/seed42 中，OOS 平均 normalized score 由 Frozen `1.022` 升到 Trainable `1.198`，OOS correct rate 由 `80.70%` 升到 `90.70%`；同一样本的 `Δdistance` 与 `Δscore` 相关为 `0.99`，`Δradius` 与 `Δscore` 仅为 `0.15`。因此当前证据更支持“Known-only 训练改变了表示空间中的距离和样本排序”，而不是“只把半径或阈值调大”。这仍是机制证据，不是因果证明；对应源表在 `results/analysis/deep_geometry_mechanism_v1/`。

### 2.6 当前主图只保留 OOS 问题

后续主视觉只回答 OOS：OOS score/rank ordering、OOS precision/recall/F1、AUROC/AUPR、false acceptance 和同样本 OOS 状态转移。Known Recall/false reject 只作为防止“全拒绝”误读的一个 coverage guard，不再展开 Known intent 分类、Router/Expert 或 intent-level 主图。既有 intent-level artifact 保留作历史追溯，但不进入 OOS-only 主汇报。

主图遵循低阅读成本合同：参考 `oos_intent论文.pdf` 后部 Figure 3–4 的单问题、短图例和明确阈值线；主文最多使用三类图（score 分布、OOS 工作点、OOS 机制），每张最多 3 个数据集面板。per-ball、detector rank transfer、五状态全矩阵和外部 backbone 图只作为按需补充证据。

### 2.2 外部 baseline 的正确写法

- ADB 与 Trainable 的数据来源接近，StackOverflow/KIR=.50 的 OOS F1 接近；但 ADB 是 BERT/TextOIR 合同，差异同时包含 backbone、训练目标和边界。
- DA-ADB 当前可审计的 StackOverflow 三 seed 结果仍是 BERT 外部参照，不并入 MiniLM fair 表；其样本级错误图位于 `figures/da_adb_current_protocol_summary_v1/`。
- DCLOOS 的 reduced 结果使用 pseudo-OOS 与外部 SQuAD OOS；它只能写成“更强监督下的参考结果”，不能直接声称 S2C 超过或落后于完整 DCLOOS。
- KNNCL、OpenMax、DOC、DeepUnk 当前没有同 split、同 Known list、同评估器下的 final metrics，因此登记为 blocked/historical，不用中间预测补表。

在 StackOverflow/KIR=.50 的三 seed 外部错误构成中，S2C 的 Known Recall/false acceptance 为 `83.68%/8.18%`，ADB 为 `80.78%/7.52%`，DA-ADB 为 `75.97%/29.07%`。因此 ADB 更像是“略保守、覆盖较低”的工作点，DA-ADB 在当前运行中同时损失 Known coverage 并增加 OOS 误接收；这比单看 OOS F1 更能说明 baseline 的代价，但仍不能把差异归因成单一 backbone 或损失。

## 3. 机制图包：从样本翻转到边界风险

本轮新生成的 bundle：

- 结果与源表：`results/analysis/mechanism_explanation_v1/`
- 图：`figures/mechanism_explanation_v1/`
- 机器清单：`results/analysis/mechanism_explanation_v1/MANIFEST.json`
- 构建脚本：`tools/analysis/build_mechanism_explanation_pack_v1.py`

分析固定在 KIR=.50，读取统一逐样本合同、既有表示几何表和 MOGB ball 风险表；测试标签只用于事后分层，`selection_used_test_oos=false`，不用于阈值、checkpoint 或 K 的选择。

### 3.1 OOS 机制 bundle

当前活跃视觉只保留 OOS score/rank、拒识、接受区域和 OOS 样本转移；Known Recall/false reject 仅作为 coverage guard。Known-intent 分类、intent-level 风险和 Router/Expert 图不再进入后续论文或视觉入口，既有对应 artifact 保留在原目录供历史追溯。

真实高维 bundle：`results/analysis/deep_geometry_mechanism_v1/`；图源清单为
`results/analysis/deep_geometry_mechanism_v1/MANIFEST.json`。它复用了已冻结的 Trainable/Frozen/MOGB/native
artifact，逐样本 replay 通过 sample-id 顺序审计，不是新的训练结果。

当前保留的 OOS 图为：

1. `real_space_boundary_overlay.png`：在原始高维空间解释不同边界的 OOS 接受区域。
2. `frozen_trainable_movement.png`：OOS 样本由 Frozen 到 Trainable 的位置与拒识状态变化。
3. `local_neighborhood_failure_map.png`：OOS 局部邻域相似性与最终拒识 score 的关系。
4. `boundary_component_decomposition.png`：OOS score 变化由最近距离还是半径变化驱动。
5. `acceptance_overlap_risk.png`：K=2 新增 OOS 接受与 MOGB OOS 拒识的区域差异。
6. `acceptance_surface_exposure.png`：高维局部 mode 中 K=1/K=2/MOGB 的 OOS 接受覆盖。
7. `mogb_ball_support_risk_map.png`：局部 ball 的 OOS 污染；coverage 只作 guard。
8. `detector_rank_transfer.png`：同一 Trainable MiniLM 表示上各 detector 的 OOS 排序分叉。
9. `multi_method_error_transitions.png`：同一 OOS 样本在各方法之间的拒绝/误接收转移。

补充的同样本 score/错误图位于 `figures/mechanism_explanation_v1/`，只保留
`paired_score_reordering.png`、`boundary_margin_fingerprint.png`、`mogb_ball_risk_surface.png`、
`geometry_boundary_coupling.png` 和 `native_detector_error_fingerprint.png` 的 OOS 解读。

### 3.2 外部 BERT/TextOIR 的 OOS 机制参照

外部图位于 `figures/da_adb_current_protocol_summary_v1/`，源表、manifest 和对齐审计位于
`results/analysis/da_adb_current_protocol_summary_v1/`。只追踪同一测试样本的 OOS 状态，不把 BERT 与
MiniLM embedding 放在同一坐标系；外部方法仍不进入 MiniLM fair 排名。

- `external_error_transition_heatmaps.png`：S2C、ADB、DA-ADB 的 OOS 状态转移。
- `external_error_signature.png`：OOS correctly rejected 与 false accept 的构成差异。

三 seed 聚合的 OOS false acceptance 为 S2C `8.18%`、ADB `7.52%`、DA-ADB `29.07%`。这只说明当前外部
工作点的 OOS 代价，不承担单一 backbone 或损失的因果归因。

### 3.3 ADB 原生 OOS 边界 probe

`results/analysis/adb_deep_mechanism_v1/` 的 StackOverflow/KIR=.50/seed42 probe 只用于补充 ADB 的
`distance / radius` 与 OOS false accept 解释；它与 canonical ADB 预测有 `113/6000` 个样本差异，不能替换
`adb_kir_sensitivity_v2` 主指标。活跃图为 `adb_native_distance_radius.png`、`adb_centroid_pca_error_map.png`、
`adb_s2c_native_error_transition.png` 和 `adb_s2c_score_rank_transfer.png`。

## 4. 当前可以下的机制结论

1. **Trainable 的首要收益是 OOS score 重排。** paired score 图和 score-mass 图共同支持 OOS separation 改善，而不是只换了一个默认阈值。
2. **S2C 相对 MOGB 的主要收益是更好的 OOS–coverage 平衡。** MOGB 的 OOS 保守性更强，但 Known Recall 只作为 guard；不能用低 false acceptance 单独宣称整体更优。
3. **K=1 是当前证据支持的稳定边界。** K=2 的风险来自接受区域并集，且依赖数据集和局部结构；不能把“表示分离更好”外推成“多中心更安全”。
4. **MOGB 的短板是 coverage–rejection 工作点，不是单一半径参数。** OOS 污染集中在部分 ball，需同时看分区、支持、半径和边界规则；Known coverage 只作 guard。
5. **native detector 对照说明方法收益不是 backbone 单独造成的。** 在同一 Trainable MiniLM 上，检测规则会改变错误类型；因此当前结论应写成“表示适配 + 单中心 Gate 的组合工作点”，不能只归因于 MiniLM。
6. **真实高维回放把“表示改变”落到了 OOS 样本层。** Trainable 的 OOS correct rate 为 `90.70%`，Frozen 为 `80.70%`；相对 MOGB，OOS 中 MOGB-only correct 为 `8.43%`。这支持表示适配改变了 OOS 的位置和拒识状态。
7. **K=2 的失败可以直接归因于接受区域并集。** 在 StackOverflow/seed42，K=2 新增接受的 OOS 比例为 `31.60%`，高于新增 Known 的 `9.43%`；这不是单个阈值点的偶然，而是 union geometry 改变后的空间结果。
8. **MOGB 的 OOS 风险是局部 ball 覆盖不均。** ball 之间的 OOS 污染并不均匀，最高 ball 的 OOS false acceptance 为 `19.05%`；不能只用“半径偏大/偏小”概括 MOGB。
9. **native detector 的劣势是排序不一致。** 同一 Trainable MiniLM 表示上，MSP/Energy 的 OOS rank 与 S2C 的相关只有 `0.54/0.57`；kNN/LOF 虽然更接近 S2C，仍分别有 `29.8%/38.9%` 的 S2C-only OOS 正确样本。这个证据支持“表示适配 + 单中心 Gate”是组合收益。
10. **外部 baseline 的可见差异是 OOS 工作点差异。** StackOverflow/KIR=.50 三 seed 的 OOS false acceptance 为 S2C `8.18%`、ADB `7.52%`、DA-ADB `29.07%`；由于合同仍是 BERT/TextOIR external，不能将其写成纯方法因果结论。
11. **K=2 的局部覆盖增益来自额外 mode，而不是单纯半径放大。** 无标签几何探针显示 K=2 在局部 mode 内打开了 K=1 未覆盖的方向；真实测试集上的 OOS 代价仍由 `acceptance_overlap_risk.png` 单独报告。

## 5. 证据限制与尚未闭合的图

- ADB 现在已有独立的 native distance/radius 机制图；它仍不进入 MiniLM score 重排或 MiniLM embedding 图。DA-ADB 当前只有预测级 OOS 状态转移和错误构成，没有稳定落盘的 native score/embedding，因此不对它补造内部空间结论。
- ADB 机制 probe 与 canonical seed42 主指标存在 `113/6000` 个预测差异；这份 probe 只用于边界解释，不覆盖或改写 `adb_kir_sensitivity_v2` 的主表。
- 深层图使用运行时重建的 Trainable embedding，不把 raw array 提交到轻量结果；若 checkpoint 或本地模型被移除，需按 manifest 中的 hash 重新复原。
- `acceptance_surface_exposure.png` 是固定随机方向的高维几何探针，只用于解释 K=2 的局部覆盖，不是新的测试性能结果；它与 `acceptance_overlap_risk.png` 的分工分别是“高维规则探针”和“真实样本错误转移”。
- PCA 边界图只用于解释，不能读取二维轮廓反推原始高维接受区域；所有 OOS 机制图都不是新的测试性能结果。
- 外部完整 baseline 仍按合同分层；没有 final metrics 的方法不进入主排名。

## 6. OOS 主视觉入口与停止条件

低阅读成本主图已经完成，不再把“图形重绘”当作待办事项，也不新增 baseline、loss、K 或训练实验。
主文只保留三类单问题图，顺序固定为 `OOS score 分布 → OOS 工作点 → OOS 机制`：

| 图 | 直接回答的问题 | 主文件 |
|---|---|---|
| OOS score 分布 | Trainable 是否把 OOS score 推向拒绝侧？ | `figures/oos_readability_figures_v1/oos_score_distribution_kir050.png` |
| OOS 工作点 | OOS F1 与 false acceptance 是否跨 KIR 同时改善？ | `figures/oos_readability_figures_v1/oos_operating_point_kir.png` |
| OOS 机制 | Trainable-only OOS 正确是否伴随更高的拒绝侧 score？ | `figures/oos_readability_figures_v1/oos_mechanism_transitions_kir050.png` |

三张图均为 1×3 数据集小面板，只比较当前协议下的 `Frozen E2 K=1` 与 `Trainable K=1`。OOS 使用强调色，
Known 只作淡色参照或 coverage guard；score 图固定黑色 `score=1` 虚线；机制图只保留 OOS 正确性转移和
`mean Δscore`。输入表、输出表、SHA256、阈值语义、测试 OOS 使用边界和逐图合同见
`results/analysis/oos_readability_figures_v1/MANIFEST.json`。

### 6.1 证据边界

- score 图使用五 seed 的完整分位数摘要，不抽样、不伪造逐样本密度；逐样本机制仍回到
  `results/analysis/deep_geometry_mechanism_v1/movement_transition_summary.csv`。
- 工作点图使用 3 个数据集 × 3 个 KIR × 5 个 seed 的正式配对均值；Known Recall 只作为 coverage guard，
  不展开 Known-intent 分类。
- 机制图使用 KIR=.50 的 OOS 状态转移，测试标签只用于 post-hoc 分组；它不改变任何训练、阈值、K 或 checkpoint。
- `Frozen single-centroid（MOGB-Fair component；Euclidean）` 的 78.64% 不进入这组三图，也不被标成当前
  canonical Frozen K=1；历史 fulltex Cascade、MOGB/BERT、ADB/DA-ADB 继续留在补充或历史层。

### 6.2 明确不再做的图

- 不再把原始点云、五状态全矩阵、per-ball 风险、detector rank transfer 或外部 backbone 图组成连续主图墙。
- 不再用新的 PCA/UMAP 或高维路径 probe 替代当前三张图；只有出现明确、可证伪的新 OOS 机制问题时才单独新增补充图。
- 不为 KNNCL、OpenMax、DOC、DeepUnk 补中间结果，也不为了 baseline 数量扩展实验矩阵。

当前停止条件是：三张主图能直接回答 OOS 结果、工作点和训练机制；若后续图不能提供超出这三张图的新解释，
就停止视觉扩展。

## 7. 机器可读入口

- Fair 汇总：`results/analysis/cross_protocol_tradeoff_v1/`
- 统一合同与对齐：`results/analysis/unified_prediction_contract_v1/`
- 真实 embedding 深层 bundle：`results/analysis/deep_geometry_mechanism_v1/`
- 本轮机制 bundle：`results/analysis/mechanism_explanation_v1/`
- 本轮 OOS 深层图：`figures/deep_geometry_mechanism_v1/`
- 本轮 OOS score/错误图：`figures/mechanism_explanation_v1/`
- 外部 backbone 机制图：`figures/da_adb_current_protocol_summary_v1/`
- ADB 原生边界机制图：`figures/adb_deep_mechanism_v1/`
- ADB 原生边界源表：`results/analysis/adb_deep_mechanism_v1/`
- OOS 低阅读成本主图与源表：`figures/oos_readability_figures_v1/`、`results/analysis/oos_readability_figures_v1/`
- 外部机制源表与对齐：`results/analysis/da_adb_current_protocol_summary_v1/`
- 总图清单：`results/analysis/archive/analysis/unified_comparison_v1/figure_manifest.json`
- 资产注册：`configs/experiment_registry.yaml`
- 历史报告：`docs/archive/analysis/`，不作为当前事实入口
