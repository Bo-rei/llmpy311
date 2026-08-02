# s2c 项目阶段性研究进展总报告

更新时间：2026-07-20  
项目路径：`/home/bo/bo01/llmpy311/s2c`

> **快照说明**：本文保留原有 Gate-only 研究进展的历史细节，但不再是当前运行契约。当前应先读仓库内的 [`docs/CURRENT_STATUS.md`](../../CURRENT_STATUS.md) 和 [`docs/EXPERIMENTS.md`](../../EXPERIMENTS.md)。截至本次更新，完整 KIR50 Cascade 已完成 36/36 个单元；结果以 `cascade_full/gpu_kir50/evaluations/matrix_manifest.json` 和同级汇总 CSV 为准。

> 本文中 v20 的 `blocked_missing_inputs` 是当时的历史预检状态，不覆盖后续修复后的完整 Cascade 结果；当前事实以 `s2c/results/` 和对应 artifact manifest 为准。

## 1. 总体进展

s2c 当前已经完成从“原有 OOS 结果”到“多簇结构、OOS 可分性和 MiniLM 语义前提”的三阶段实验补强：

```text
v19：完整 K 消融、Gate-only Baseline、聚类质量和 near-OOS 基线
  ↓
v20：排除替代解释，验证随机分簇、selected-K 泛化和直接多边界 Baseline
  ↓
v21：验证 MiniLM 语义空间、语义子簇、冻结表示和 near-OOS 表示碰撞
```

核心方法结构保持不变：

```text
Gate → Router → Expert
```

本轮没有提出新的 Cascade 方法，没有覆盖历史结果，也没有把 TextOIR 外部协议数字混入 s2c 主表。

当前最重要的科学结论是：

> 多簇不是跨数据集普遍有效的结构；其收益取决于 MiniLM 空间中的真实局部语义结构、数据集的类内几何和 OOS 边界目标是否一致。near-OOS 的主要瓶颈目前更接近 MiniLM 表示层碰撞，而不是单纯局部半径过大。

完整 Cascade 的新证据补上了系统闭环：修复后的 Banking77-OOS/StackOverflow Expert 不再使用旧低预算 checkpoint，三数据集的 KIR50、三 seed、四 Gate 组合共 36 个单元均成功完成。这个结果只能支持“Gate 差异可以在固定下游中被观测”，不能支持某个 Gate 在所有数据集上全面最优。

## 2. 研究问题与实验阶段对应关系

| 研究问题 | 主要实验 | 当前状态 |
|---|---|---|
| Known intent 是否存在多个局部簇？ | K=1..5、聚类质量、稳定性、casebook | 已完成 |
| 多簇是否改善 OOS 可分性？ | fixed/tuned K 消融、near-OOS、overlap | 已完成 |
| 多簇收益是否来自语义聚类？ | KMeans vs size-matched random partition | 已完成 |
| validation-selected K 是否泛化？ | validation/test ranking、selection regret | 已完成 |
| MiniLM 是否形成可读出的意图空间？ | Purity、类内/类间距离、linear probe、kNN | 已完成 |
| near-OOS 错误来自表示还是边界？ | representation collision / boundary overcoverage | 已完成首轮 |
| 冻结 MiniLM 是否合理？ | Frozen vs CE vs SupCon | seed42 首轮完成 |
| Gate 收益是否传递到完整系统？ | Gate 替换 Router/Expert | 36/36 完成 |

## 3. v19：核心多簇/OOS 实验

### 3.1 实验矩阵

实验根目录：

[cluster_separability_v19](</home/bo/bo01/llmpy311/artifacts/s2c/outputs/experiments/cluster_separability_v19>)

| 实验 | 配置 | 完成情况 |
|---|---|---:|
| Fixed-boundary K 消融 | 3 数据集 × 3 KIR × K=1..5 × 2 距离 × 3 seed | 270/270 |
| Validation-tuned K 消融 | 同上 | 270/270 |
| K=8 stress control | 3 数据集 × KIR50 × seed42 × 2 距离 | 6/6 |
| MiniLM Gate-only Baseline | 5 方法 × 3 数据集 × 3 KIR × 3 seed | 135/135 |
| 聚类稳定性 | 3 数据集 × KIR50 × K=2..5 × 3 seed | 36/36 |
| near/medium/far OOS | validation 分桶 + 全部 Known test | 144 行 |
| empirical overlap | Known/OOS 多覆盖率 | 18 行 |
| hard-intent 诊断 | 多模态程度与 K=1→K=2 收益 | 660 行 |
| 聚类质量 | intent-level 诊断 | 9,900 行 |
| MiniLM 表征控制 | L2、Mahalanobis、去 PC1、norm-only | 12 行 |

固定协议包括：

- OOS 为正类；
- 分数越大越倾向 OOS；
- K、半径参数和阈值只使用 validation；
- diagonal Mahalanobis 使用每个局部簇独立估计的方差；
- 可视化按 ground truth 分组；
- Gate-only 与端到端指标分开报告。

### 3.2 K 消融结果

#### CLINC150

- Fixed Euclidean 中，K=2 相对 K=1 的 OOS F1 增量为 `+0.0058`，95% bootstrap CI 为 `[+0.0025,+0.0087]`；
- K=5 的增量变为 `-0.0110`；
- 说明少量多簇有帮助，但继续碎片化会损害边界质量。

#### BANKING77-OOS

- Diagonal Mahalanobis 中，K=5 相对 K=1 增量为 `+0.0332`；
- 95% bootstrap CI 为 `[+0.0211,+0.0447]`；
- 9/9 配对单元均获益，是当前最支持多簇边界的任务。

#### StackOverflow

- 所有 K>1 均退化；
- Euclidean K=2 相对 K=1 增量为 `-0.1335`；
- 95% bootstrap CI 为 `[-0.2008,-0.0860]`；
- 说明多簇不能被描述为普遍有效的方法。

### 3.3 聚类稳定性与碎片化

K=2 的 bootstrap ARI 中位数：

| Dataset | K=2 ARI median | K=5 ARI median |
|---|---:|---:|
| BANKING77-OOS | 0.9027 | 0.6975 |
| CLINC150 | 0.9201 | 0.6191 |
| StackOverflow | 0.8865 | 0.5350 |

按“最小簇比例 < 0.10 或 effective-K 小于 requested-K”定义碎片化：

- BANKING77-OOS：K=2 为 `0.078`，K=5 增至 `0.785`；
- CLINC150：K=2 为 `0.000`，K=5 增至 `0.621`；
- StackOverflow：K=2 碎片化仍为 0，但 OOS F1 已明显下降。

因此 StackOverflow 的失败不能只归因于小簇样本不足，还包含语义切分和 OOS 边界不匹配。

### 3.4 Baseline 结果

Gate-only 对比覆盖：MiniLM-MSP、Energy、Entropy、kNN、LOF、单中心 Euclidean、单中心 diagonal Mahalanobis、MultiSphere K=2 和 selected-K。

三个数据集的最佳受控 Baseline 不同：

| Dataset | 最佳 Baseline | 平均 OOS F1 |
|---|---|---:|
| CLINC150 | Entropy | 0.9042 |
| BANKING77-OOS | kNN | 0.8750 |
| StackOverflow | Energy | 0.7961 |

这说明论文不能只与单中心方法比较，也不能假设某一种置信度方法跨数据集最优。

### 3.5 near-OOS 结论

near-OOS 平均 F1（BANKING77-OOS / CLINC150 / StackOverflow）：

```text
near: 0.3399 / 0.4864 / 0.3531
far : 0.9492 / 0.8252 / 0.6999
```

near-OOS 是当前真正的开放世界瓶颈，应优先研究 Known 与语义相邻 OOS 的局部可分性，而不是继续优化容易识别的 far-OOS。

## 4. v20：排除替代解释

实验根目录：

[cluster_separability_v20](</home/bo/bo01/llmpy311/artifacts/s2c/outputs/experiments/cluster_separability_v20>)

### 4.1 selected-K 泛化

使用 v19 tuned 网格中的 validation/test 指标，共 54 个单元：

| Dataset | Spearman(validation, test) | Selection accuracy | Mean test regret | Selected-K 不低于 K=1 |
|---|---:|---:|---:|---:|
| BANKING77-OOS | 0.8667 | 0.7222 | 0.0021 | 0.8889 |
| CLINC150 | 0.7056 | 0.3889 | 0.0031 | 0.7778 |
| StackOverflow | 0.9278 | 0.9444 | 0.0004 | 0.9444 |

结论：selected-K 不能只报告选择频次。CLINC150 的 validation/test 选择一致性较低，必须同时报告 oracle regret、相对 K=1 安全性和 ID Recall 变化。

### 4.2 KMeans vs 随机分簇

随机分簇保持 KMeans 的每个 intent 子簇大小多重集合，只随机改变样本归属。共完成 105 个随机单元，每个单元 5 次重复。

定义：

```text
delta_semantic = F1_KMeans - mean(F1_RandomPartition)
```

| Dataset | Euclidean | Diagonal Mahalanobis | 解释 |
|---|---:|---:|---|
| BANKING77-OOS | +0.0360 | +0.0204 | 支持局部语义归属有贡献 |
| CLINC150 | +0.0076 | -0.0008 | 语义贡献小或不稳定 |
| StackOverflow | -0.1167 | -0.0607 | KMeans 切分与 OOS 目标不匹配 |

这是目前最关键的机制证据：多簇收益不是“球体数量增加就一定有效”。BANKING77-OOS 支持 KMeans 局部语义结构；StackOverflow 则表明随机容量匹配的边界反而更好。

### 4.3 自适应局部边界 Baseline

完成 18 个单元。每个 intent 只使用 Known train silhouette 在 K=1..5 中选择局部 K，不使用 OOS、validation 或 test 选择。

相对固定 K=2 的 test OOS F1：

- BANKING77-OOS：Euclidean `+0.0161`，Mahalanobis `+0.0126`；
- CLINC150：Euclidean `-0.0081`，Mahalanobis `-0.0178`；
- StackOverflow：Euclidean `-0.0251`，Mahalanobis `-0.0196`。

该 Baseline 不能普遍替代固定 K，也没有被包装为新方法。

### 4.4 端到端传递状态

已生成 36 个 Gate 替换代表单元的 preflight 和可恢复命令清单：

```text
3 datasets × 3 KIR × {K=1, K=2, selected-K, validation-best Baseline}
```

当前 workspace 缺少 Router/Expert checkpoint 和历史 SmolLM 权重，因此 36/36 标记为 `blocked_missing_inputs`。没有复制历史 eval 数字，也没有伪造 Gate→Router→Expert 传递结论。

## 5. v21：MiniLM 作为方法前提的直接验证

实验根目录：

[cluster_separability_v21](</home/bo/bo01/llmpy311/artifacts/s2c/outputs/experiments/cluster_separability_v21>)

### 5.1 语义空间探针

覆盖 3 数据集 × KIR50 × 3 seed，共 9 个单元：

- Purity@5/10/20；
- 跨意图近邻混淆；
- 类内/类间 cosine 距离；
- 最近意图 margin；
- Logistic Regression 与 kNN probe；
- near/medium/far 表示碰撞—边界过覆盖分解。

| Dataset | Known test Purity@10 | Relative separation | Linear probe Known macro-F1 |
|---|---:|---:|---:|
| CLINC150 | 0.8993 | 0.4709 | 0.9674 |
| BANKING77-OOS | 0.8506 | 0.3987 | 0.8874 |
| StackOverflow | 0.8168 | 0.2227 | 0.9075 |

这证明最终 MiniLM embedding 中存在可被监督探针读出的意图结构，但 StackOverflow 的空间分离显著较弱。

### 5.2 near-OOS 表示碰撞与边界过覆盖

K=2 的 near-OOS 表示碰撞率：

| Dataset | Representation collision | False-accept 中 boundary overcoverage |
|---|---:|---:|
| CLINC150 | 0.6820 | 约 0.0504 |
| BANKING77-OOS | 0.9751 | 约 0.0095 |
| StackOverflow | 0.8881 | 约 0.0546 |

操作性结论是：near-OOS false accept 中绝大部分样本已经落入最近 Known intent 的语义支持范围，单纯缩小半径不能解决主要问题。

### 5.3 子簇语义 casebook

已生成：

- `minilm_cluster_casebook.csv`：1,645 条子簇记录；
- `minilm_cluster_annotation_template.csv`：735 条人工审计模板；
- `minilm_cluster_casebook_manifest.json`：完整 provenance。

每个子簇包含中心附近代表句、TF-IDF 关键词、簇大小、中心间隔和词汇 JSD。自动关键词只用于准备人工审计，不能替代人工语义结论。

### 5.4 Frozen / CE / SupCon 首轮

本地 `all-MiniLM-L6-v2` 权重可用，已完成三个数据集 KIR50 seed42 的完整 train/val/test、1 epoch 正式单元。固定边界为 diagonal Mahalanobis、per-cluster covariance、mean+std radius、threshold=1。

| Dataset | Representation | K=1 OOS F1 | K=2 OOS F1 | selected-K OOS F1 |
|---|---|---:|---:|---:|
| CLINC150 | Frozen | 0.8816 | 0.8784 | — |
| CLINC150 | CE | 0.8510 | 0.8685 | — |
| CLINC150 | SupCon | 0.8991 | 0.8934 | — |
| BANKING77-OOS | Frozen | 0.8616 | 0.8463 | 0.9048 (K=5) |
| BANKING77-OOS | CE | 0.7868 | 0.7906 | 0.8779 (K=5) |
| BANKING77-OOS | SupCon | 0.8842 | 0.8693 | 0.9040 (K=5) |
| StackOverflow | Frozen | 0.8436 | 0.7941 | — |
| StackOverflow | CE | 0.9140 | 0.6077 | — |
| StackOverflow | SupCon | 0.9199 | 0.7292 | — |

这轮不是最终统计结论，但已经显示：表征适配可能改善 K=1 的 OOS 排序，却破坏类内多模态结构，使 K=2 多簇边界退化。因此 Frozen/CE/SupCon 必须使用 fixed-boundary 与 validation-tuned 两套结果共同解释。

## 6. 代码与工程进展

### 6.1 实验代码收拢

[cluster_separability vertical package](</home/bo/bo01/llmpy311/s2c/tools/experiments/cluster_separability/__main__.py>) 当前包含：

- `protocol.py`：统一 OOS 指标、gold sample grouping、validation-only 选择；
- `runner.py`：fixed/tuned K 网格；
- `baselines.py`：统一 MiniLM Gate-only Baseline；
- `analysis.py`：稳定性、near-OOS、overlap、hard-intent 和表征分析；
- `v20_analysis.py`：selected-K、near-OOS、效率聚合；
- `v20_random_partition.py`：随机分簇对照；
- `v20_adaptive_boundary.py`：自适应局部边界 Baseline；
- `v20_end_to_end.py`：端到端 preflight 和命令生成；
- `v21_semantic_probe.py`：MiniLM 语义空间探针；
- `v21_cluster_casebook.py`：子簇代表句与人工审计模板；
- `v21_representation_adaptation.py`：Frozen/CE/SupCon 表征适配。

所有新增实验均使用独立 v20/v21 根目录，避免覆盖历史 v19。

### 6.2 中文注释与协议审计

新增代码按 karpathy-guidelines 保持单一职责、最小修改和显式阻断：

- 解释 OOS 分数方向和阈值语义；
- 解释为什么 near-OOS bucket 必须加回全部 Known test；
- 解释 per-cluster covariance 的范围；
- 解释随机分簇为什么要匹配 KMeans 簇大小；
- 解释 automatic casebook 不能替代人工语义标注；
- 输入缺失或中间层未缓存时只生成 preflight，不伪造结果。

### 6.3 CLI

```bash
python -m tools.experiments.cluster_separability v20-analysis
python -m tools.experiments.cluster_separability v20-random
python -m tools.experiments.cluster_separability v20-adaptive
python -m tools.experiments.cluster_separability v20-end-to-end
python -m tools.experiments.cluster_separability v21-semantic-probe
python -m tools.experiments.cluster_separability v21-casebook
python -m tools.experiments.cluster_separability v21-adaptation --preflight
```

## 7. 测试与可复现性

当前从 s2c 工作目录运行：

```text
pytest -q tests
176 passed, 1 skipped
```

同时已验证：

- v20 targeted tests：13 passed；
- v21 targeted tests：8 passed；
- `py_compile`：通过；
- `git diff --check`：通过；
- v19 fixed/tuned 及 Baseline 缺失单元：0；
- v20/v21 结果均有 manifest、CSV/Parquet 输出和独立 provenance。

注意：从 workspace 根目录运行部分旧 CLI 测试会触发仓库已有的 `WorkspacePaths.discover()` 路径限制；规范验证目录是 `/home/bo/bo01/llmpy311/s2c`。

## 8. 当前未完成与风险

### 8.1 MiniLM 中间层

现有 v19 cache 只有最终 384 维 SentenceTransformer embedding，没有 layer2/4/6 hidden states。`layer_preflight.json` 已明确标记 blocked，不能用最终 embedding 冒充中间层结果。

### 8.2 人工语义审计

casebook 已准备，但 15 个代表 intent 的 3 人标注尚未执行。因此当前只能说“有可审计候选证据”，不能宣称 KMeans 子簇已被人工证实为语义子模式。

### 8.3 Frozen/CE/SupCon 的统计规模

Gate-only 的 Frozen/CE/SupCon/CE-Recon 结果已经覆盖正式三 seed；完整 Cascade 的 CE-Recon 也已经覆盖 KIR50 三 seed。论文仍应把 fixed/tuned 和 Gate-only/Cascade 分表，不能把不同协议混成一个平均数。

### 8.4 端到端传递

Gate-only 收益已经在完整 Gate→Router→Expert 上复验。当前结果支持“部分表示/边界变化能传递到系统”，但也显示 controlled baseline 在部分数据集更强，因此不能声称 CE-Recon 或多簇全面优胜。

### 8.5 语义扰动和层级/家族控制

同义改写、实体/动作最小变化、MiniLM layer2/4/6、MiniLM-L6/L12 对照尚未完成。这些应在核心 adaptation 多 seed 闭环后再执行。

## 9. 建议的下一步

1. 将 36 单元 Cascade 汇总、错误分解和权重 hash 登记进 `study_closeout`；
2. 完成 MOGB 官方协议的明确复现结论；无法公平复现时保留审计，不伪造数字；
3. 用现有结果重写论文 claims，明确多簇是条件性收益而非普遍提升；
4. 最后再做论文表图和 active/legacy 代码清理，不再创建新的版本号文档。

当前不建议继续增加更多 K、更多普通 Baseline、更多 TextOIR 方法或重复 UMAP；这些实验的信息增量低于上述未闭环问题。
