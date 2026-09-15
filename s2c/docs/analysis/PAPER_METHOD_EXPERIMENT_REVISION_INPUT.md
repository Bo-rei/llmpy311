# Final Paper Method / Experiments Fact Pack

更新时间：2026-09-11。范围：只整理当前仓库已有的代码、results、artifact manifest 和分析报告；不重跑实验，不把事后最高值改写成主结果。

## 0A. 2026-09-15 current audit addendum（优先于下文冲突口径）

本文件下文保留 2026-09-11 的历史事实包。以下 addendum 是对当前仓库最新状态的只读审计；凡与下文冲突，以本段和 `docs/CURRENT_STATUS.md`、`results/final_paper_main/` 为准。

**结论：当前没有一组覆盖三数据集、三 KIR、同一 candidate space、同一 Known-only selection rule、完整 Gate→Router→Expert 且可无条件称为 9/9 SOTA 的主表结果。** 先前的“9/9”是数值上把历史 H1 CLINC150/StackOverflow 与标准 `banking77` 或历史 `banking77_oos` 结果合并后的结论，不能作为统一 protocol 的主表数字。

### Current standard Banking77 evidence

`results/final_paper_main/MANIFEST.json` 的 `36/36` 只包含标准 `banking77` 的 Ours 9 cells 和 official-compatible MOGB 27 cells；它不包含当前 protocol-v2 下 CLINC150/StackOverflow 的完整 Ours 3×3 矩阵。标准 Banking77 Ours 的 locked final bundle 为：

| KIR | OOS F1 | Known F1 | Accuracy | Known Recall | False Acceptance |
|---:|---:|---:|---:|---:|---:|
| .25 | 91.782±0.577 | 76.485±2.187 | 87.359±0.976 | 81.711 | 10.101 |
| .50 | 86.395±0.989 | 81.200±0.892 | 83.766±1.107 | 80.746 | 9.658 |
| .75 | 70.620±1.151 | 82.131±0.186 | 78.680±0.432 | 80.043 | 12.105 |

这些行来自同一批 full-pipeline predictions，selection 为 Known train/dev only，`pseudo_oos_used=false`、`real_oos_used_for_selection=false`、`test_used_for_selection=false`。root manifest 同时记录 `test_previously_observed=true`，所以它是 locked Known-only result，但不能宣称 untouched-holdout。

### Absolute best observed is not a legal merged result

在所有当前标准 Banking77 Ours full-pipeline JSON 中逐 campaign 聚合得到的最高观察值为 `.25=94.023±0.163`（coverage70 campaign）、`.50=86.395±0.989`（final threshold1 bundle）、`.75=73.236±3.425`（fixed-known control）。三者来自不同 selection contract，不能拼接成一行 Ours；coverage70/fixed-known 还记录了既有 test artifact。其余 11-point Banking77 KIR sweep 的结果属于 `coverage_repair`，可用于 sensitivity 诊断，但不是一个新的 untouched-holdout 主表。

历史 H1 full-pipeline 的 CLINC150 `95.185/92.130/86.805` 和 StackOverflow `95.501/89.232/75.944`（均值，KIR=.25/.50/.75）仍可作为历史 H1 结果；它们与标准 Banking77 final 不得混排。历史 `banking77_oos` 也必须单独报告。

### K dependence audit

K 不是普遍最优为 1：历史 CLINC150 `.25/.50` 选择过 `K=2`；标准 Banking77 `coverage_repair` 的 Known-only selector 在多个 KIR 选择过 `K=3/4/5`；StackOverflow 的既有 K-sweep 则显示 `K>1` 退化。由于这些证据来自不同 campaign，当前只能支持“最优 K 具有 dataset/KIR dependence”，不能支持把各 cell 的 test 最高 K 拼成最终 Ours。

### Paper-eligibility resolution

- **条件可写入**：标准 Banking77 `results/final_paper_main/summary.csv` 的 3 个 Ours rows，以及 27-cell MOGB official-compatible matrix；均须保留 protocol、selection 和 MOGB compatibility caveat。
- **历史/诊断**：`historical_known_best_9of9`、`trainable_full_pipeline_ablation_known_only`、`coverage_repair` 和各 adaptive/geometry search；不得冒充当前统一主表。
- **当前缺口**：没有一份同时覆盖 CLINC150、StackOverflow、标准 Banking77 的 current-protocol 3×3 Ours full-pipeline bundle，且所有 cell 共享同一 candidate space、selection rule 和 test-before-lock barrier。因此当前不能输出一个审计意义上的统一 9/9 paper-eligible table。

证据入口：`results/final_paper_main/MANIFEST.json`、`results/final_paper_main/summary.csv`、`results/analysis/trainable_full_pipeline_ablation_known_only/summary.csv`、`results/analysis/kir_sensitivity_known_only/coverage_repair/summary.csv`、`docs/analysis/final_paper_experiment_closeout.md`。

**优先级说明：下文原有的“Final Ours=K=1”是 2026-09-11 历史统一合同建议；它不覆盖本次不预设 K 的逐 cell 审计，也不能把 K=1 事后提升为所有 dataset/KIR 的最终最优配置。**

### Existing-cell audit snapshot（标准 Banking77）

以下是当前仓库已有标准 `banking77` full-pipeline 结果中，每个 KIR 的最高观察值；百分比均来自同一 cell 的同一批三 seed predictions。`coverage_repair` 的选择分数是 Known-dev utility 的跨 seed 均值；其余 contract 没有可与该分数直接比较的统一导出分数。

| KIR | OOS F1 | Known F1 | Acc | Known Recall | FA | selected recipe / geometry | validation evidence | paper-eligible in unified 3-dataset table |
|---:|---:|---:|---:|---:|---:|---|---|:---:|
| .10 | 93.509±1.239 | 63.272±4.628 | 88.755±1.995 | 86.667±3.076 | 10.809±2.125 | last2; K1 Euclidean mean+0; ratio | coverage_repair utility=.905806 | no |
| .20 | 88.149±3.832 | 69.114±4.038 | 82.154±5.132 | 86.944±1.499 | 18.468±6.516 | last4; K1 Euclidean mean+0; inverse-margin | coverage_repair utility=.901877 | no |
| .25 | 94.023±0.163 | 76.893±0.739 | 90.206±0.276 | 71.272±0.926 | 2.931±0.153 | last2; K1 Euclidean mean+0; no fusion; coverage=.70 | geometry score not exported | no; separate coverage70 campaign |
| .30 | 84.607±2.670 | 73.171±2.816 | 79.329±3.047 | 87.826±1.161 | 22.762±4.545 | compact_margin; K4 Euclidean center-quantile .5; max-ratio | coverage_repair utility=.884211 | no |
| .40 | 84.585±0.846 | 78.944±0.762 | 81.212±0.889 | 88.629±0.302 | 21.087±1.350 | last1; K4 Euclidean mean+0; inverse-margin | coverage_repair utility=.876114 | no |
| .50 | 86.395±0.989 | 81.200±0.892 | 83.766±1.107 | 80.746±1.080 | 9.658±2.413 | temperature014; K1 diagonal Mahalanobis mean+1; no fusion; threshold=1 | Known-dev F1 mean=.973573 | conditional; final threshold1 bundle |
| .60 | 75.511±2.820 | 81.497±0.577 | 78.593±1.434 | 89.674±0.672 | 29.919±4.730 | temperature10; K3 Euclidean intent-quantile .99; inverse-margin | coverage_repair utility=.847953 | no |
| .70 | 71.377±3.783 | 83.571±0.533 | 79.578±1.378 | 90.093±0.587 | 31.377±6.431 | lr_high; K5 Euclidean mean+1.5; inverse-margin | coverage_repair utility=.827348 | no |
| .75 | 73.236±3.425 | 84.533±0.770 | 81.353±1.180 | 88.807±0.739 | 22.281±6.436 | last2+projection; K1 diagonal Mahalanobis mean+1; normalized-union | fixed Known-dev coverage=.90 | no; separate fixed-known campaign |
| .80 | 61.305±3.872 | 84.674±0.067 | 79.816±0.866 | 89.879±0.846 | 37.278±4.304 | all6; K1 Euclidean mean+1.5; inverse-margin | coverage_repair utility=.820735 | no |
| .90 | 53.253±2.719 | 86.089±1.079 | 81.807±1.298 | 90.266±1.124 | 33.333±3.118 | lr_high; K2 Euclidean mean+0; max-ratio | coverage_repair utility=.796506 | no |

The source for rows `.10/.20/.30/.40/.60/.70/.80/.90` is `artifacts/s2c/analysis/kir_sensitivity_known_only/coverage_repair/`; `.25` is `.../banking_known_recipe_geometry_search/coverage70/`, `.50` is `.../banking_textoir_aligned/fixed_k1_mahalanobis_threshold1_known_only/`, and `.75` is `.../banking_fixed_known/`. The result manifest for the dense sweep is `partial` (43/495 summary units) and documents prior test artifacts; these rows are therefore not a clean unified paper table even when their selection fields say no test selection.

### Current dataset×KIR coverage matrix

| Dataset / data key | .10 | .20 | .25 | .30 | .40 | .50 | .60 | .70 | .75 | .80 | .90 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CLINC150 | — | — | full pipeline, coverage_repair | — | — | full pipeline, coverage_repair | — | — | full pipeline, coverage_repair | — | — |
| StackOverflow | — | — | Gate-only / historical full pipeline only | — | — | current bridge only; no search lock | — | — | Gate-only / historical full pipeline only | — | — |
| standard Banking77 | full pipeline, coverage_repair | full pipeline, coverage_repair | full pipeline, multiple contracts | full pipeline, coverage_repair | full pipeline, coverage_repair | full pipeline, final bundle | full pipeline, coverage_repair | full pipeline, coverage_repair | full pipeline, multiple contracts | full pipeline, coverage_repair | full pipeline, coverage_repair |

Thus the only current full-pipeline dense-KIR source is standard Banking77; CLINC150 has three anchor cells under the repair campaign, and StackOverflow has no current-protocol full-pipeline search matrix. Missing cells must remain missing rather than receive historical or Gate-only numbers.

## 0. 先给最终口径

**修订稿的 Ours 应定义为：`Gate → Router → Expert` 不变，Gate 采用 Known-only Trainable `all-MiniLM-L6-v2`，只解冻最后两个 Transformer blocks 加 residual projection，最终使用单中心 `K=1`、对角 Mahalanobis、`mean+1 std` 半径和固定 normalized score threshold `1`；Router/Expert 使用对应 H1 v19 的 `SmolLM-135M` LoRA 组件。`K>1` 不属于最终 Ours，而属于可配置扩展及 controlled ablation/analysis。**

当前结果已经实现 **9/9 numerical OOS-F1 wins**：三个数据集 × 三个 KIR 的 Known-only 结果均超过当前注册外部 OOS-F1 参照。需要单独标注的不是“是否胜出”（胜出成立），而是证据层级：该 9/9 bundle 合并了两个最终评估 campaign，manifest 标记 `clean_single_holdout_claim=false`，且它是 Gate-only、没有 full-pipeline verification。因此论文可以如实写“在当前注册参照上 Known-only OOS F1 为 9/9 数值胜出”，但不能把它无标注地写成同一统一配置、干净 untouched-holdout 下的完整 Cascade SOTA。

## 1. Final Ours

### 1.1 方法合同

| 项目 | 最终采用口径 |
|---|---|
| 系统拓扑 | `Gate → Router → Expert`。Gate 先作二分类 OOS 决策；仅 ID 样本进入 Router 和 Expert。 |
| Gate backbone | 本地 `all-MiniLM-L6-v2`，6 个 Transformer layers，384 维；mean pooling 后 L2 normalize。 |
| Gate 表示 | **Trainable**。Embedding、前 4 个 Transformer blocks（0--3）和 pooler 冻结；最后 2 个 blocks（4--5）以及 projection 可训练。 |
| Projection | `LayerNorm(384) → Linear(384,256) → GELU → Linear(256,384)`；以 residual 方式加回 pooled embedding；最后一层零初始化；输出再 L2 normalize。Projection 参数为 198,016；Trainable Gate 共 3,746,944 个可训练参数。 |
| Gate loss | Known train 上 `CE + 0.1 * intra + 0.1 * inter`；CE logits 为负的中心距离除以 temperature；`intra` 为到本类中心的平方距离，`inter=ReLU(own-other+0.20)`。temperature `0.07`，inter-class margin `0.20`，classification weight `1.0`。 |
| Gate training | AdamW；projection warm-up 1 epoch，随后最多 3 个 finetune epochs，patience=1；projection LR `2e-4`，最后两层 backbone LR `2e-5`；Gate batch=64，max length=256。仅使用 Known train。 |
| Center | 最终 `K_y=1`：每个 Known intent 一个由 Known train normalized embeddings 求得并再次 L2 normalize 的类中心。训练阶段每 epoch 刷新中心；推理阶段中心固定在 checkpoint 对应的 train representation 上。 |
| Distance | 对角 Mahalanobis：每个中心按 train residual 的逐维方差加 `1e-6`，`d_y(e)=sqrt(sum_j((e_j-c_{y,j})^2/(var_{y,j}+1e-6)))`。 |
| Radius | `r_y=mean(d_y(train))+lambda*std(d_y(train))`；最终 Ours `lambda=1.0`。 |
| Score | `s(e)=d_{nearest}(e)/r_{nearest}`。K=1 时 `nearest_sphere` 与 union 规则等价；最终保留 `nearest_sphere` 命名。 |
| Decision | `s(e)>1` 判 OOS，`s(e)≤1` 判 ID；threshold=1.0。阈值不是由 test OOS 选择。 |
| Checkpoint selection | Known-only validation：materialized val 中只保留 `label=0` 的 Known rows；以 `f1_k + 0.05 * Known Recall` 选择 checkpoint。真实 OOS、pseudo-OOS 和 test 不参与训练或 checkpoint 选择。 |
| Optional semantic gate | H1 full-pipeline evaluation 的 `semantic_gate_enabled=False`；这关闭的是额外的二阶段语义 verifier，不是上面的主几何 Gate。 |

### 1.2 Router / Expert

| 项目 | 最终采用口径 |
|---|---|
| Backbone | Router 和 Expert 均为约 135M 参数的 `SmolLM-135M`。 |
| Router | CLINC150 使用 10-domain Router；StackOverflow 和 BANKING77-OOS 为单域 constant routing。LoRA rank=32，alpha=64，target modules 为 `q_proj`/`v_proj`；训练上限 10 epochs，batch=32，LR `2e-4`，patience=5，max length=64。 |
| Expert | 每个 domain 一个 Expert；LoRA rank=16，alpha=32，target modules 为 `q_proj`/`v_proj`；训练上限 15 epochs，batch=32，LR `2e-4`，patience=5，max length=64。 |
| Supervision | Router/Expert 只使用 Known labels；OOS 样本不用于其分类训练。 |
| Matched comparison | Frozen-vs-Trainable 报告中 Router/Expert、数据、KIR、seed 和 Gate boundary 均保持一致；该配对专门识别 Gate 表示变化的作用。 |

### 1.3 K>1 的论文定位

最终 Ours 固定 K=1。K>1 只用于：

- `K=1,...,5` 的 Gate-level controlled study；
- fixed/adaptive K、KMeans/random partition、MOGB-style boundary 的结构分析；
- 若正文需要，可称为可配置的 multi-center extension，但不能把每个 dataset/KIR 的事后最优 K 拼成一个统一 Ours。

现有证据是数据集依赖的：CLINC150 的 K=2 收益有限且继续增大 K 后下降；StackOverflow 的 K>1 明显退化并扩大 OOS acceptance；BANKING77-OOS 在部分距离设置下随 K 增大受益。故不能保留“multi-cluster 一定产生更清晰边界”的表述。

## 2. Final Protocol

### 2.1 H0、H1、protocol_v2 的边界

| 线 | 实际作用 | 选择/监督 | 可写结论 |
|---|---|---|---|
| Historical H0 / old paper | `../artifacts/s2c/outputs/paper_results/` 中的历史 full Cascade 锚点；原论文主表意图为多中心 Gate、Router、Expert 和语义组件。 | 旧论文合同不是当前严格 Known-only control；历史文字和运行记录允许/使用 OOS validation 参与部分 `lambda`/boundary 选择。 | 只能作 historical reference。原始 H0 的完整 data/v19、Gate、Router/Expert、语义输入链没有全部逐字恢复，不能称严格 H0 重跑。 |
| H1 controlled v19 | registry 名为 `historical_v19_paper_main`；数据根为 `../assets/datasets/s2c/prepared/data/multidataset/v19`；本事实包主表使用 `historical_gate_ablation` 的 27 个 Trainable full-pipeline rows。 | 主表配对使用 Known-only checkpoint selection；Known-only val 是 materialized val 的 `label=0` 子集；test OOS 只作最终评估。另有 H1 OOS-aware/adaptive-K 结果，必须单独标注。 | 是当前最适合改写 Method/Experiments 的历史数据 H1 controlled line，但不是严格 H0。 |
| `protocol_v2_textoir_v1` | 已冻结的当前 fair/reference line；标准 `banking77`，不是 `banking77_oos`；正式 fair matrix 为 3 datasets × 3 KIR × 5 seeds。 | Known-only fair/reference contract；MOGB-MiniLM、Random K=2、TextOIR native 等均在此线分层。 | 只作独立 fair/reference supplement，不能回填 H1 主表或把 `banking77` 写成 `BANKING77-OOS`。 |

### 2.2 H1 数据集、Known 构造和 split

以下计数来自 `assets/datasets/s2c/prepared/data/multidataset/v19/*/kir*_seed42/MANIFEST.json`；`dev` 与 manifest 的 `val` 同义。表中 Gate val/test 的 OOS 行是 materialized view 中的 OOS 行；严格 Known-only 选择时只读取其中 Known rows。

#### CLINC150

- 实际 key：`clinc150`；报告名 `CLINC150`。
- source roots：`clinc_data_origin/data` 和 `clinc_oos_root`；公共来源在稿件中可引用 CLINC/OOS-Eval。
- 150 个 in-domain intents；Known intent 数按 `domain_balanced_largest_remainder` 确定：KIR=.25/.50/.75 分别为 38/75/112 Known，余下为 held-out intents。
- Gate val 是 Known + native `clinc_oos`；Gate test 是 Known + held-out intent OOS + native `clinc_oos`。Gate train、Router 和 Expert 均 Known-only。

| KIR | Known intents | Gate train | Gate dev: Known/OOS | Gate test: Known/OOS |
|---:|---:|---:|---:|---:|
| .25 | 38 | 3,800 | 760 / 2,340 | 1,140 / 4,360 |
| .50 | 75 | 7,500 | 1,500 / 1,600 | 2,250 / 3,250 |
| .75 | 112 | 11,200 | 2,240 / 860 | 3,360 / 2,140 |

CLINC 的旧 v19 root 审计发现跨 split text duplicates；这是历史数据限制，不在本包中删除或静默修复。

#### StackOverflow

- 实际 key：`stackoverflow`；报告名 `StackOverflow`。
- source root：`stackoverflow_origin`；使用 Kaggle-derived `jacoxu/StackOverflow` 20K title/tag snapshot，文本字段为 `title`，20 intents，单 domain。
- Known intent 数按旧 seeded random KIR 确定：KIR=.25/.50/.75 分别为 5/10/15 Known；其余 held-out intents 构成 OOS。没有 native OOS source。
- Gate val/test 是 Known + held-out intent OOS；Gate train、Router 和 Expert 均 Known-only；Router 为 constant single-domain path。

| KIR | Known intents | Gate train | Gate dev: Known/OOS | Gate test: Known/OOS |
|---:|---:|---:|---:|---:|
| .25 | 5 | 2,996 | 500 / 1,498 | 1,497 / 4,493 |
| .50 | 10 | 5,995 | 998 / 1,000 | 2,993 / 2,997 |
| .75 | 15 | 8,994 | 1,498 / 500 | 4,491 / 1,499 |

StackOverflow 的当前 split audit 通过；不要把它的旧 title snapshot 与 `protocol_v2` 的 raw TSV 行序混用。

#### BANKING77-OOS

- 实际 key：`banking77_oos`；论文显示名称应为 `BANKING77-OOS`，不能裸写成标准 `Banking77`。
- source root：`banking77_oos_origin`；这是包含 OOS 扩展的实际数据线，不是标准 77-intent `banking77`。
- KIR 的 intent universe 是 50 个 in-domain Banking intents；Known 数为 KIR=.25/.50/.75 的 12/25/38。unknown intents 包含 held-out in-domain intents 以及 native/extension OOS labels。
- Gate val/test 包含 held-out in-domain OOS、`id_oos` 和 `ood_oos`；manifest source 为 `id-oos/{valid,test}` 和 `ood-oos/{valid,test}`。Gate train、Router 和 Expert 均 Known-only；Router 为 constant single-domain path。

| KIR | Known intents | Gate train | Gate dev: Known/OOS | Gate test: Known/OOS |
|---:|---:|---:|---:|---:|
| .25 | 12 | 1,467 | 370 / 1,866 | 480 / 3,600 |
| .50 | 25 | 2,933 | 749 / 1,487 | 1,000 / 3,080 |
| .75 | 38 | 4,497 | 1,146 / 1,090 | 1,520 / 2,560 |

`banking77`（标准 77-intent、protocol_v2/archive）是另一条数据线；历史论文 `93.99` artifact 的实际 key 已核实为 `banking77_oos`。

### 2.3 Training / validation / test selection contract

| 结果线 | Training | Validation | K / lambda / threshold | Test OOS |
|---|---|---|---|---|
| H1 matched Trainable K=1（主表候选） | Known train only；center/loss 只用 Known。 | 只用 `label=0` Known calibration；checkpoint score=`f1_k+0.05*Known Recall`。 | K=1、diag Mahalanobis、lambda=1、threshold=1 均为预先固定的 matched contract，不由 test 选择。 | 不参与训练、checkpoint、K、lambda 或 threshold；只在锁定后评估。 |
| H1 `historical_known_geometry` | 已有 Known-only checkpoint；不重训表示或下游。 | Known utility=`correct accepted - 4*wrong accepted`；搜索 K=1..5、距离、radius、rule、threshold、fusion；不读 test。 | 选择锁写入 `results/analysis/historical_known_geometry/`。 | 仅 finalization 后读；该阶段结果是 Gate-only surrogate，不等于 OOS-F1 validation。 |
| H1 `historical_known_training` | Known train；11 recipes，涉及解冻层数、projection、temperature、loss、margin、LR、epoch。 | seed42 筛选，按 Known validation macro F1 选 top recipes，再扩展 seed13/87；不读 test。 | 需与 geometry lock 联合；不存在用 test 选 recipe 的合法记录。 | final bundle 在读取前锁定；但两轮 campaign 的合并仍不构成 clean untouched-holdout。 |
| H1 mixed / OOS-aware | Known train。 | 某些 H1 full-pipeline `.50` recipe/adaptive-center 结果使用 validation OOS F1（或 OOS F1+Known/Accuracy guard）。 | 可有 adaptive K、不同 lambda/threshold；必须单列。 | test 只确认选定候选，不应与 Known-only 表合并。 |
| Historical H0 | 旧 Cascade 训练/验证合同。 | 旧论文合同涉及 OOS-aware lambda/boundary selection。 | 论文式 K=2、CLINC lambda=.5、其他 lambda=1；但 H0 输入链不完整。 | historical anchor/reference。 |

## 3. Final Main Results

### 3.1 推荐主表候选：H1 matched Trainable K=1 full pipeline

这是当前最干净的论文候选：`results/analysis/historical_gate_ablation/per_seed.csv` 与 `summary.csv`，27 个 Trainable full-pipeline units（3 datasets × 3 KIR × 3 seeds），`test_used_for_selection=false`，`oos_used_for_training=false`。所有行均为同一 `Trainable K=1 / last2+projection / diag-Mahalanobis / mean+1std / nearest_sphere / threshold=1` 合同。

下表数值为百分比；括号内 seed 顺序均为 `13/42/87`；`std` 为 population std（ddof=0）。Known F1 是全测试集上 Known classes 的 macro F1，包含 OOS 被误接收为 Known 所造成的 precision cost；`F1-All` 不是本表的 Known F1。FA 为 OOS false acceptance rate。

| Dataset | KIR | OOS F1: seeds; mean±std | Known F1: seeds; mean±std | Acc: seeds; mean±std | Known Recall mean±std | FA mean±std | strongest registered external ref; Δ |
|---|---:|---|---|---|---:|---:|---:|
| CLINC150 | .25 | 95.40 / 95.21 / 94.68; **95.10±0.30** | 75.31 / 73.41 / 77.45; **75.39±1.65** | 91.58 / 91.02 / 90.98; **91.19±0.27** | 73.13±1.98 | 2.97±1.01 | KNNCL 93.56; **+1.54** |
| CLINC150 | .50 | 88.63 / 89.56 / 90.30; **89.50±0.68** | 75.75 / 77.21 / 77.97; **76.98±0.92** | 83.89 / 85.15 / 85.84; **84.96±0.81** | 73.44±0.96 | 4.11±0.91 | DA-ADB 90.10; **−0.60** |
| CLINC150 | .75 | 81.19 / 81.33 / 79.87; **80.80±0.66** | 78.47 / 78.14 / 77.55; **78.05±0.38** | 80.02 / 79.82 / 78.85; **79.56±0.51** | 73.83±0.15 | 4.36±1.28 | DA-ADB 86.00; **−5.20** |
| StackOverflow | .25 | 96.20 / 95.99 / 93.94; **95.38±1.02** | 82.43 / 82.74 / 82.66; **82.61±0.13** | 92.29 / 92.17 / 90.28; **91.58±0.92** | 83.56±0.19 | 3.83±2.00 | DA-ADB 92.65; **+2.73** |
| StackOverflow | .50 | 88.78 / 91.49 / 86.08; **88.78±2.21** | 81.54 / 85.44 / 83.17; **83.38±1.60** | 84.97 / 88.41 / 84.29; **85.89±1.80** | 83.71±0.22 | 7.10±3.98 | DA-ADB 88.86; **−0.08** |
| StackOverflow | .75 | 73.99 / 78.45 / 73.05; **75.16±2.36** | 82.98 / 84.63 / 85.12; **84.24±0.91** | 80.70 / 83.04 / 82.02; **81.92±0.96** | 83.11±0.22 | 9.21±4.80 | DA-ADB 74.55; **+0.61** |
| BANKING77-OOS | .25 | 92.72 / 93.59 / 91.42; **92.58±0.89** | 65.51 / 63.32 / 59.66; **62.83±2.41** | 87.25 / 88.38 / 85.10; **86.91±1.36** | 84.38±3.12 | 12.01±1.63 | DA-ADB 86.57; **+6.01** |
| BANKING77-OOS | .50 | 88.41 / 87.68 / 89.32; **88.47±0.67** | 67.58 / 64.48 / 65.53; **65.86±1.29** | 81.42 / 80.29 / 82.03; **81.25±0.72** | 85.47±1.53 | 16.93±1.09 | DA-ADB 79.93; **+8.54** |
| BANKING77-OOS | .75 | 86.98 / 86.82 / 86.01; **86.60±0.42** | 70.45 / 69.96 / 69.87; **70.09±0.25** | 79.83 / 79.58 / 78.92; **79.44±0.38** | 86.58±0.76 | 17.54±0.49 | DA-ADB 69.37; **+17.23** |

主表可以写出的直接结论是：在该**统一 H1 matched full-pipeline contract**下，Trainable K=1 相对注册外部数字为 6/9 个 OOS-F1 行更高；CLINC150/.50 和尤其 /.75 仍不是最高。这不否定下一小节的 Known-only 9/9 数值胜出；两者回答的是不同问题：本表是统一配置的 full-pipeline 主候选，下一表是逐 dataset×KIR 选择配置的 Gate-only 高 OOS-F1 结果。External references 来自旧论文/current runner 的 registered values，不是同一 H1 data/backbone/selection contract，故 `Δ` 只能作 contextual reference，不能称 fair ranking。

### 3.2 Known-only 9/9 OOS-F1 结果：可报告的高性能候选，但必须标注证据层级

来源：`results/analysis/historical_known_best_9of9/summary.csv`、`per_seed.csv`、`MANIFEST.json`。训练/验证确实记录为 Known-only，`real_oos_used_for_selection=false`，`pseudo_oos_used=false`，`test_used_for_selection=false`；因此 **9/9 相对注册外部 OOS-F1 参照的数值胜出成立**。但该 bundle 合并了 `historical_known_coverage90_v3` 与 `fixed_known_standard_contract_stackoverflow` 两个 campaign，且在 SO contract 固定前已有 exploratory test artifacts。manifest 明确标记 `clean_single_holdout_claim=false`；它也是 Gate-only，`full_pipeline_verified` 未闭合。若主文只展示 Gate OOS-F1，可以作为 `Known-only validation-selected Gate` 表报告，并在表注披露这些限制；若主文表被定义为完整 `Gate→Router→Expert` Cascade，则仍应使用 3.1 的 full-pipeline 表，不能把 Gate-only 数字冒充 Cascade 数字。

| Dataset | KIR | Known-only locked geometry | OOS F1 mean±std | registered ref | Δ |
|---|---:|---|---:|---:|---:|
| CLINC150 | .25 | K=2, Euclidean, mean+0std, nearest, ratio fusion w=1 | 95.19±0.48 | 93.56 | +1.63 |
| CLINC150 | .50 | K=2, Euclidean, intent-quantile .70, nearest, inverse-margin w=.5 | 92.13±0.81 | 90.10 | +2.03 |
| CLINC150 | .75 | K=1, Euclidean, mean+2std, nearest, max-ratio w=2 | 86.80±1.55 | 86.00 | +0.80 |
| StackOverflow | .25 | K=1, Euclidean, mean+1.5std, normalized-union, no fusion, threshold=.90 | 95.50±0.96 | 92.65 | +2.85 |
| StackOverflow | .50 | K=1, Euclidean, mean+1.5std, normalized-union, no fusion, threshold=.90 | 89.24±2.03 | 88.86 | +0.38 |
| StackOverflow | .75 | K=1, Euclidean, mean+1.5std, normalized-union, no fusion, threshold=.90 | 75.95±1.94 | 74.55 | +1.40 |
| BANKING77-OOS | .25 | K=4, Euclidean, mean+0std, nearest, inverse-margin w=1 | 89.09±2.17 | 86.57 | +2.52 |
| BANKING77-OOS | .50 | K=1, Euclidean, mean+0std, nearest, inverse-margin w=1 | 83.55±3.85 | 79.93 | +3.62 |
| BANKING77-OOS | .75 | K=1, Euclidean, mean+0std, nearest, inverse-margin w=1 | 79.91±1.67 | 69.37 | +10.54 |

CLINC/Bank rows in this Gate-only summary do not carry complete Known Recall/FA fields; do not reconstruct them from another run. `historical_known_final/MANIFEST.json` also reports `wins=3` and `full_pipeline_verified=false`, whereas the joined 9/9 manifest reports `wins=9`; this is an internal status conflict about bundle scope/verification, not evidence that the nine OOS-F1 comparisons failed. The adopted resolution is: report the 9/9 numerical Known-only OOS-F1 wins explicitly; label the table Gate-only and campaign-history limited; use the matched 27-row H1 table whenever the paper table claims a uniform full Cascade; do not call either layer an unconditional fair SOTA ranking.

### 3.3 Baseline provenance and comparison eligibility

| Baseline / source | Backbone | Supervision | Current evidence | Use in paper |
|---|---|---|---|---|
| MSP / OpenMax / DOC / DeepUnk / KNNCL / ADB / DA-ADB old table | Historical reported implementations; current workspace does not recover one common H1 implementation/backbone for all rows. ADB/DA-ADB current compatibility is BERT/TextOIR. | Historical reported contract; current TextOIR native MSP/DOC/ADB use Known-only train/eval and test OOS. | Old fulltex numeric table; KNNCL/OpenMax/DeepUnk do not all have same split/Known-list/evaluator final artifacts. | Historical/context reference only; do not label as same-protocol fair. |
| TextOIR native MSP/DOC/ADB | TextOIR native BERT; keys `oos`, `banking`, `stackoverflow` | Known-only train and eval; test includes unseen labels mapped to OOS. Code evidence: `textoir/open_intent_detection/dataloaders/utils.py`, train lines 97--108, eval lines 110--117, test lines 119--129. | Native compatibility, 3 seeds; separate from H1 `banking77_oos`. | BERT/TextOIR external reference, not MiniLM fair ranking. |
| MOGB-style MiniLM component | MiniLM controlled component; same `protocol_v2_textoir_v1` fair line. | Known-only. | `cross_method_oos_mechanism_presentation.md`, `fair_kir50_summary.csv`; 5 seeds, KIR=.50, standard `banking77`/CLINC/SO. | Fair component/coverage analysis only. |
| Official MOGB / `li2025multi` | Original MOGB BERT and adaptive granular-ball/hierarchical contract. | Original data/training contract; not restored in full. | Official/local compatibility status is recorded in `official_mogb_status.csv`; not strict. | Report status separately; no unconditional rank. |
| DCLOOS / `zhan2021dcloos` | BERT reduced reference. | Pseudo-OOS plus external SQuAD OOS supervision. | Only reduced `87.05%` OOS-F1 reference is available; no same-protocol final rank. | Different-supervision reference, never unmarked in the Known-only table. |
| CLAB / `liu2023clab` | No compatible final run in current workspace. | Not applicable. | No registered compatible predictions/metrics. | Mark unavailable; do not insert a number. |

TextOIR native implementation is structurally Known-only for train/eval, but its BERT/data-key/evaluator contract differs. The only direct causal comparison in this package is H1 Frozen-versus-Trainable under the same K=1 Gate contract.

## 4. Reviewer-facing Ablations

### 4.1 Frozen vs Trainable

| Field | Fact |
|---|---|
| Research question | With data, K, distance, radius, threshold and downstream fixed, does adapting the MiniLM Gate representation improve OOS rejection? |
| Controlled variables | H1 v19; 3 datasets × 3 KIR × 3 seeds; same Router/Expert; K=1; diagonal Mahalanobis; mean+1std; threshold=1; only Frozen vs Trainable Gate changes. 54 CUDA full-pipeline units. |
| Result | Trainable−Frozen OOS F1 (KIR=.25/.50/.75): CLINC `+0.80/+1.47/+1.84 pp`; StackOverflow `+1.52/+9.76/+12.03 pp`; BANKING77-OOS `+1.00/+3.65/+3.05 pp`. FA decreases in all 9 cells; Accuracy increases in all 9 cells. |
| Supported claim | Under this H1 contract, Known-only last-two-layer + projection adaptation improves OOS score ordering and full-pipeline OOS/Accuracy relative to matched Frozen K=1. |
| Unsupported claim | It does not prove Trainable universally improves K>1, every backbone, every data protocol, or all individual seeds; it does not restore strict H0. |

### 4.2 Controlled K=1,...,5

| Field | Fact |
|---|---|
| Research question | Is the historical fixed K=2 assumption justified, and does increasing local centers consistently help? |
| Controlled variables | Gate-level fixed-boundary study; same data, seed set and representation within each row; vary per-intent subcenter count K and compare Euclidean/diagonal Mahalanobis. Historical KIR=.50 table uses 3 data seeds and no Router/Expert. A broader registered Gate analysis covers 1,650 units across 11 KIR values and 5 seeds. |
| Result | Historical KIR=.50 diagonal-Mahalanobis OOS F1: CLINC K=1..5 `88.02, 88.07, 87.33, 86.12, 85.39`; StackOverflow `79.02, 72.80, 66.21, 65.02, 66.80`; BANKING77-OOS `84.82, 85.57, 87.72, 89.25, 89.78`. Euclidean rows are also in `tab:multicluster_k_ablation_v19`/the source report. |
| Supported claim | K=2 is not universal: CLINC has only a small K=2 gain, StackOverflow degrades for K>1, and BANKING77-OOS can benefit from more centers under this distance. |
| Unsupported claim | This Gate-only historical study does not identify a universal optimal K, does not show that Trainable K=1 plus K>1 are one single method, and does not transfer standard `banking77` results to `banking77_oos`. |

### 4.3 Fixed K vs adaptive K

| Field | Fact |
|---|---|
| Research question | Does validation-selected per-intent K outperform a fixed K under the same H1 full-pipeline setting? |
| Controlled variables | H1 KIR=.50, 3 seeds, same Trainable pipeline and candidate geometry; compare fixed/adaptive center selection. **Selection used validation OOS F1**, so this is OOS-aware, not strict Known-only. |
| Result | CLINC overall `90.99±0.78`, fixed `90.83±0.51`, adaptive `90.99±0.78`; StackOverflow overall `89.96±1.50`, fixed `89.99±1.47`, adaptive `89.76±1.50`; BANKING77-OOS overall `91.89±0.19`, fixed `91.56±0.17`, adaptive `91.89±0.19`. Adaptive−fixed: `+0.16/−0.23/+0.32 pp` for CLINC/SO/Bank. |
| Supported claim | Adaptive K is data-dependent; it is not uniformly better. |
| Unsupported claim | These OOS-aware values cannot be used as strict Known-only main results or as evidence that adaptive K is the final Ours contract. |

### 4.4 KMeans vs random partition and multi-center risk

| Field | Fact |
|---|---|
| Research question | Does a multi-center gain come from meaningful partitioning, and what OOS risk is introduced by the union of local regions? |
| Controlled variables | Same-protocol fair component layer: MiniLM, Known-only, KIR=.50, 5 seeds; swap fixed/random/MOGB partitions and S2C/MOGB boundaries. Separate H1 seed42 replay gives a direct acceptance-region diagnostic. |
| Result | In standard `protocol_v2_textoir_v1`, aggregate `Random K=2` OOS F1 is `78.39%` with FA `23.86%` and Known Recall `84.25%`; `Frozen K=2` is `75.51%` with FA `25.81%` and Known Recall `81.44%`. In StackOverflow/seed42, K=2 accepts an additional `31.60%` of OOS but only `9.43%` additional Known relative to K=1. |
| Supported claim | Adding centers changes the acceptance region and can preferentially admit near-OOS; partition quality and boundary rule matter. |
| Unsupported claim | Random partition is not a final baseline replacement; this layer is not H1 `banking77_oos` full-pipeline evidence and does not prove a universal causal advantage of KMeans or MOGB. |

### 4.5 Gate removal and cascade variants

| Field | Fact |
|---|---|
| Research question | Is the explicit Gate necessary, and is the hybrid MiniLM-Gate/SmolLM-downstream cascade preferable to all-SmolLM or all-MiniLM variants? |
| Controlled variables | Historical paper layout: 3 datasets × 3 KIR × 4 variants × seed42 = 36 historical eval JSONs. Variants are historical Ours, Without Gate, Cascade-MiniLM and Cascade-SmolLM. These are old CPU/old-environment artifacts, not fresh CUDA H1 retraining. |
| Result | Historical OOS-F1 rows (Ours / Without Gate / Cascade-MiniLM / Cascade-SmolLM): CLINC `.25 95.30/76.88/94.27/90.98`, `.50 91.96/70.24/90.67/78.69`, `.75 82.14/66.50/81.00/71.79`; SO `.25 91.70/88.30/86.85/45.00`, `.50 89.71/82.79/79.41/55.68`, `.75 75.57/65.83/62.50/28.72`; Bank `.25 93.99/91.27/91.41/64.43`, `.50 88.23/82.88/84.63/77.61`, `.75 85.28/84.20/83.69/55.41`. |
| Supported claim | In the historical system artifact, removing the explicit Gate substantially lowers OOS F1; the Gate is a meaningful architectural component. |
| Unsupported claim | It does not isolate Trainable representation, does not prove current H1 byte identity, and does not justify “SOTA across all settings.” |

### 4.6 MOGB comparison

| Field | Fact |
|---|---|
| Research question | Is the observed work point specific to S2C’s representation/boundary combination, and how does it compare with a multi-granularity competitor? |
| Controlled variables | Same-backbone MiniLM component comparison under `protocol_v2_textoir_v1`, Known-only, KIR=.50, 5 seeds; standard `banking77`, CLINC150 and StackOverflow. Original official MOGB is separately BERT/native-contract. |
| Result | OOS F1 / Known Recall / FA: CLINC S2C Trainable K1 `90.44/74.44/3.69` vs MOGB-MiniLM `81.32/31.57/0.90`; SO `87.67/83.89/9.34` vs `72.92/27.09/0.79`; standard Banking77 `83.56/82.21/15.74` vs `74.99/33.41/1.09`. |
| Supported claim | Under the controlled component contract, MOGB’s lower FA is accompanied by strong Known under-coverage; S2C offers a different coverage–rejection operating point. |
| Unsupported claim | This does not prove S2C exceeds the complete official BERT MOGB, and standard `banking77` rows cannot be ranked against H1 `banking77_oos` rows without an explicit protocol bridge. |

## 5. Unified Mechanism Report and Figures

主文最多保留三类图。若主文只使用 H1 主表，前两类优先；第三类必须在图注中明确它属于 protocol_v2/reference line。

| 类别 / exact path | Dataset / KIR / seed / protocol | 回答的问题 | 可支持的一句话 | 不能过度解读 |
|---|---|---|---|---|
| Score separation: `figures/historical_oos_visual_explanation/paper_style_score_distribution.png` | StackOverflow, KIR=.50, seed=42, H1 controlled, Frozen vs Trainable K=1；score threshold=1。 | 固定阈值不变时，Trainable 是否把 OOS score 推向拒识侧？ | 该单元 OOS FA 从 Frozen `14.6%` 降至 Trainable `2.2%`，变化来自 score 分布相对固定边界的位置变化。 | 不是所有数据集/seed 的完美分离；不能从一张分布图推出普适泛化或阈值严格分离。 |
| Boundary / sample transition: `figures/historical_oos_visual_explanation/paired_oos_score_crossing.png` | CLINC150、StackOverflow、BANKING77-OOS，KIR=.50，seeds=13/42/87，H1 controlled；两方法共同 threshold=1。 | 哪些同一样本跨过边界，Trainable 的提升是否只是均值现象？ | Trainable-only OOS crossing rate 为 CLINC `3.5%`、SO `19.4%`、Bank `8.7%`，反向退化仍存在但较少。 | 这是 OOS state transition，不是语义因果证明；不能说所有 OOS 都被修复，也不能把 downstream classification error 当作 OOS detector error。 |
| Multi-center risk: `figures/deep_geometry_mechanism_v1/acceptance_overlap_risk.png` | CLINC150、StackOverflow、标准 `banking77`，KIR=.50，seeds=13/42/87，plot seed=42，`protocol_v2_textoir_v1`；测试标签只用于 post-hoc error analysis。 | K=2 的 union acceptance 是否扩大 near-OOS 风险？ | 在 SO/seed42，K=2 相对 K=1 额外接受 31.60% OOS、9.43% Known，说明多中心扩张对 OOS 的代价不对称。 | 不能把标准 `banking77`/protocol_v2 图解释成 H1 `banking77_oos` 主结果；测试标签不参与选择；PCA/固定方向可视化不是新的 detector。 |

可选的 H1 局部边界补充图为 `figures/historical_oos_visual_explanation/paper_style_local_boundary_geometry_3d.png`（StackOverflow/.50/seed42）；它使用真实 384D score 决策标记和共同 PCA，但二维方向/点间距离仍不能当作原始高维几何。不要保留旧稿 “threshold strictly separates the two distributions” 的句子。

## 6. Efficiency

benchmark：`docs/analysis/historical_deployment_benchmark.md`、`results/analysis/historical_deployment_benchmark/summary.csv`。单一 NVIDIA RTX 5070 12GB / CUDA；H1 v19，KIR=.50，seed=42，Frozen/Trainable K=1；warm-up 5 次，测量 20 次；batch=1 和 32。吞吐为按样本换算值，显存为 CUDA peak allocated。

### 6.1 Gate-only

| Dataset | Gate | Total / trainable params | B1 latency / throughput / memory | B32 latency / throughput / memory |
|---|---|---:|---|---|
| CLINC150 | Frozen | 22.71M / 0 | 3.61 ms / 277.0 s⁻¹ / 360.8 MB | 0.49 ms / 2050.7 s⁻¹ / 369.4 MB |
| CLINC150 | Trainable | 22.91M / 3.75M | 3.02 ms / 330.8 s⁻¹ / 449.1 MB | 0.43 ms / 2305.5 s⁻¹ / 457.7 MB |
| StackOverflow | Frozen | 22.71M / 0 | 3.41 ms / 293.3 s⁻¹ / 96.1 MB | 0.32 ms / 3103.4 s⁻¹ / 111.4 MB |
| StackOverflow | Trainable | 22.91M / 3.75M | 3.07 ms / 325.8 s⁻¹ / 183.5 MB | 0.29 ms / 3407.6 s⁻¹ / 199.5 MB |
| BANKING77-OOS | Frozen | 22.71M / 0 | 3.26 ms / 307.2 s⁻¹ / 183.2 MB | 0.33 ms / 2993.2 s⁻¹ / 197.8 MB |
| BANKING77-OOS | Trainable | 22.91M / 3.75M | 2.68 ms / 372.9 s⁻¹ / 270.6 MB | 0.29 ms / 3434.5 s⁻¹ / 285.2 MB |

### 6.2 Full Cascade

Full Cascade benchmark没有在同一表中重复报告总参数量；可写的参数事实仍是 Gate 的 22.71M/0 或 22.91M/3.75M。其 measured CUDA latency/throughput/peak memory 为：

| Dataset | Gate | B1 latency / throughput / memory | B32 latency / throughput / memory |
|---|---|---|---|
| CLINC150 | Frozen | 3.44 ms / 290.5 s⁻¹ / 360.8 MB | 2.74 ms / 365.5 s⁻¹ / 629.6 MB |
| CLINC150 | Trainable | 2.89 ms / 346.6 s⁻¹ / 449.1 MB | 27.04 ms / 37.0 s⁻¹ / 1413.0 MB |
| StackOverflow | Frozen | 3.48 ms / 287.3 s⁻¹ / 96.1 MB | 1.35 ms / 741.6 s⁻¹ / 389.7 MB |
| StackOverflow | Trainable | 3.83 ms / 261.2 s⁻¹ / 183.5 MB | 0.31 ms / 3266.9 s⁻¹ / 199.5 MB |
| BANKING77-OOS | Frozen | 3.15 ms / 317.4 s⁻¹ / 183.2 MB | 1.44 ms / 695.7 s⁻¹ / 482.3 MB |
| BANKING77-OOS | Trainable | 2.73 ms / 366.5 s⁻¹ / 270.6 MB | 1.27 ms / 786.9 s⁻¹ / 568.9 MB |

安全 claim：这些结果只证明已测 H1 K=1 CUDA Gate/full-Cascade 路径具有可量化的参数、延迟、吞吐和显存开销；不支持“比所有 baseline 更快/更省”、CPU 优势、所有 batch size 优势或完整四变体部署优势。CLINC Trainable full-Cascade batch=32 的 27.04 ms 是明显异常点，正文不应回避或泛化。

## 7. Reviewer-gap closure 与 Remaining Gaps

### 已有闭环

- Frozen vs Trainable：54 个 CUDA full-pipeline units；Trainable K=1 的 27 行可作为 H1 matched candidate。
- K=1,...,5、adaptive K、random/KMeans、near-OOS 和 acceptance-risk：Gate-level evidence 已存在，但协议层级不同，需分栏。
- Known-only vs OOS-aware tuning：历史 H0、H1 balanced、H1 OOS-first、严格 Known-only 已在报告中拆开；不可合并成一条选择曲线。
- corrected Known F1、Known Recall、FA、F1-All、Accuracy、AUROC/AUPR 均已有可追溯字段。Known F1 必须使用 full-test-set Known-class macro F1，不再使用 legacy known-subset-only `known_macro_f1`。
- TextOIR native MSP/DOC/ADB 的 train/eval Known-only 逻辑已审计；其 BERT/native data key 仍与 H1 不同。
- H1 K=1 Gate-only/full-Cascade CUDA efficiency benchmark 已完成。

### 未闭环或不能写成无条件结论

1. **Official MOGB strict reproduction**：原作者旧 environment、原始数据/采样合同和最终 granular-ball state 没有全部恢复；已有 local exact-compatible/corrected-loss/compatibility rows 不能称 official strict reproduction。
2. **DCLOOS**：现有数值是 BERT + pseudo-OOS + external SQuAD supervision 的 reduced reference；没有 same-protocol Known-only final metrics，不进入无标注排名。
3. **H0 strict Cascade**：历史 full-anchor 和 current H1 不等同；历史 `fulltex` 主表的 Banking77 展示名与实际 `banking77_oos` artifact、不同 KIR 的数据/语义路径和 split duplicates 均需披露。
4. **Protocol mismatch**：`banking77_oos` H1、标准 `banking77` protocol_v2/archive、TextOIR native `banking`/`oos` 不能放入一张无标签 SOTA 表。
5. **SOTA wording**：Known-only joined bundle 对 registered external OOS references 为 **9/9 numerical wins**；matched uniform Trainable K=1 full-pipeline table 为 6/9 higher，CLINC/.50、CLINC/.75、SO/.50 不高于对应 reference。9/9 结果可以作为已完成的 Gate-only numerical result 报告，但不能无标注地升级为 clean untouched-holdout、同一统一配置或完整 Cascade 的 unconditional fair SOTA。
6. **Full-pipeline 9/9 selected geometry**：当前 9/9 bundle 是 Gate-only，且 `clean_single_holdout_claim=false`；不能用它覆盖 matched K=1 main contract，也不能把不同 K/distance/rule 当作一个模型。
7. **Deployment scope**：没有完整四变体、CPU 或更广 batch-size benchmark；不写全局部署优势。
8. **Additional robustness**：没有新增 hard/adversarial OOS、incremental intent-addition 或完整 sentence-encoder factorial 结果；只能写为 limitation/future work。

## 8. LaTeX Context

### 8.1 当前稿件路径和同步状态

- canonical `s2c/fulltex.tex`：当前 worktree 中为 **deleted**，不存在可直接编辑的 canonical fulltex。
- 保留的历史 fulltex copy：`s2c/paper/old_fulltex.tex`。它包含旧 Experiments、K=1,...,5 表和旧四变体表，但仍有 stale claim（如 K=2 默认、OOS validation wording、9/9 SOTA 和 strict separation）。
- 当前工作 draft：`s2c/paper/new_polish.tex`。它包含 `\section{Related Work new}` 和较完整的最新 Related Work，但 Method/Experiments 仍沿用旧的 Frozen/no-training、K=2 和 SOTA 叙述，不能视为已同步的最终稿。
- Related Work 修订材料：`s2c/docs/analysis/RELATED_WORK_CLAIM_CORRECTIONS_V1.md`；对应 patch 为 `s2c/docs/analysis/FULLTEX_REVIEW_REVISION_PATCH_V1.md` / `FULLTEX_REVIEW_REVISION_V1.patch`。**结论：最新 Related Work 草稿已存在，但尚未同步到 canonical fulltex，也尚未与最终 Method/Experiments 对齐。**

### 8.2 Method/Experiments 改写时必须保留

#### Notation

- Cascade：`x`, `G`, `R`, `E_d`, `z∈{ID,OOS}`, `d∈𝒟`, `y∈𝒴_d⊆𝒴_ID`。
- Representation：`e=f_θ(x)`, `\bar e=e/||e||_2`, `h=384`。
- Geometry：`𝒞_y={c_{y,k}}`, `K_y`, `r_{y,k}`, diagonal Mahalanobis `d(\bar e,c)`, normalized score `s(\bar e)=min_k d/r`，Gate rule `s≤1 ID`, `s>1 OOS`。
- 改写后的正文必须说明 final Ours 为 K=1；`K_y` 保留为扩展/分析变量，不要把旧公式直接解释成最终固定 K=2。

#### Existing labels

保留已有 label（其中空格是当前源稿中的实际字符，若不做全局 label migration 不要随意改名）：

`fig:motivation`, `fig:model_architecture`, `fig:gate_embedding`, `fig:gate_score_distribution`, `eq: embed`, `eq: ky`, `eq: lambda`, `eq: ryk`, `tab:dataset_stats`, `tab:main_results_all`, `tab:multicluster_k_ablation_v19`, `tab:ablation_all_kir`, `sec: cascade`。

#### Citation keys

当前 `new_polish.tex`/历史稿中与 Method/Experiments/Related Work 相关的 key：

`larson2019evaluation`, `hendrycks2017baseline`, `bendale2016towards`, `shu2017doc`, `lin2019deep`, `zhou2022knn`, `zhang2021adaptive`, `zhang2023learning`, `zeng2021modeling`, `lee2018simple`, `guo2017calibration`, `wang2021texsmart`, `zawbaa2024improved`, `hu2022lora`, `li2025multi`, `zhang2024new`, `wang2024beyond`, `arora2024intent`, `zaera2025efficient`, `xu2026multi`, `tur2010what`, `louvan2020recent`, `wolflein2025agents`, `hoffman2024inferring`, `muzahid2024survey`, `gorin1997how`, `wang2002combination`, `haffner2003optimizing`, `sarikaya2011deep`, `liu2016attention`, `goo2018slot`, `chen2019bert`, `liu2023clab`, `zhan2021dcloos`。历史 Method 还出现 `vaswani2017attention`, `kim2014convolutional`, `schuurmans2020intent`, `chen2025collaborating`；合并源稿时保留实际仍被引用的 key。

### 8.3 术语锁定

统一使用：`historical_v19_paper_main`（H1 controlled）、`protocol_v2_textoir_v1`（冻结 reference）、`banking77_oos` / `BANKING77-OOS`、`Trainable K=1`、`Frozen K=1`、`Gate→Router→Expert`、`Known-only`、`OOS F1`、corrected `Known F1`、`Known Recall`、`False Acceptance`、`F1-All`、`AUROC/AUPR`、`KIR`、`MOGB-style`（same-backbone component）与 `official MOGB`（non-strict status）。

不要继续使用：`state-of-the-art across all settings`、`multi-cluster always helps`、`threshold strictly separates Known/OOS`、`Trainable makes fixed multi-center geometry universally effective`、`faster than all baselines`。

## 9. Traceability ledger

主要数字和判断的机器可读入口：

- 状态与协议层：`docs/CURRENT_STATUS.md`、`docs/EXPERIMENTS.md`、`docs/EXPERIMENT_LEDGER.csv`、`docs/analysis/REVIEWER_GAP_CLOSURE_V1.md`、`docs/analysis/HISTORICAL_PROTOCOL_RECONCILIATION_V1.md`。
- Final Ours implementation：`src/protocol_v2/experiments/racal_v1/representation.py`、`src/protocol_v2/experiments/racal_v1/boundary.py`、`src/protocol_v2/experiments/racal_v1/runner.py`、`tools/legacy/analysis_v19/run_trainable_minilm_historical_v1.py`。
- Main candidate results：`results/analysis/historical_gate_ablation/MANIFEST.json`、`per_seed.csv`、`summary.csv`、`docs/analysis/historical_gate_ablation_report.md`。
- 9/9 exploratory result：`results/analysis/historical_known_best_9of9/MANIFEST.json`、`summary.csv`、`per_seed.csv`、`docs/analysis/historical_known_best_9of9.md`。
- Historical four variants / K table：`docs/analysis/historical_paper_ablation_report.md`、`paper/old_fulltex.tex`、`../artifacts/s2c/outputs/paper_results/`。
- Adaptive and Known-only searches：`docs/analysis/historical_trainable_adaptive_centers_presentation.md`、`docs/analysis/historical_known_geometry.md`、`results/analysis/historical_known_geometry/`、`results/analysis/historical_known_training/`。
- MOGB/TextOIR/DCLOOS：`docs/analysis/cross_method_oos_mechanism_presentation.md`、`results/analysis/cross_method_oos_mechanism/`、`textoir/open_intent_detection/dataloaders/utils.py`。
- Figures：`results/analysis/historical_oos_visual_explanation/MANIFEST.json`、`docs/analysis/RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md`、`results/analysis/deep_geometry_mechanism_v1/MANIFEST.json`。
- Efficiency：`docs/analysis/historical_deployment_benchmark.md`、`results/analysis/historical_deployment_benchmark/summary.csv`、`MANIFEST.json`。
