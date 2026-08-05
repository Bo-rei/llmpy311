# s2c 当前方法、实验进展与基线差异

更新时间：2026-08-05  
活动协议：`protocol_v2_textoir_v1`

本文是面向研究讨论的中文入口，回答四个问题：当前 s2c 到底实现了什么、已经做了哪些实验、
当前最好的结果是什么、为什么不能把这些结果直接写成“SOTA 排名”。原始逐样本结果、模型和
完整运行产物仍只保存在本地 artifact 中；本文只引用轻量统计和哈希证据。

## 1. 先给结论

当前 s2c 不是一个已经验证成功的“通用自适应多中心方法”。当前最可靠的自有结果是：

> **Known-only 训练的 Trainable MiniLM + 单中心 Gate（K=1）在 StackOverflow 上明显优于冻结
> MiniLM K=1；但在同一 Trainable MiniLM 表示上固定改成 K=2 后，Known Recall 上升，却产生大规模
> OOS false acceptance，OOS F1 下降。**

因此当前研究状态是：

- 表示适配（K=1）已经出现稳定正收益；
- 固定多中心的收益高度依赖数据集和意图，不能作为默认配置；
- StackOverflow 上的主要瓶颈是多球接受区域的过覆盖，而不是单纯的 MiniLM 没训练；
- 当前不能宣称超过 MOGB 论文或 DCLOOS，也不能宣称已达到 SOTA；
- MOGB、ADB、DA-ADB、DCLOOS 的现有结果中，有些是同协议组件比较，有些是不同 BERT/OOS 监督条件下的兼容性结果，必须分开报告。

## 2. 当前代码实现的方法是什么

### 2.0 一个必须先澄清的事实：当前还没有完成真正的自适应多中心

你的判断是正确的。当前代码中“多中心”和“自适应”是两个尚未同时成立的概念：

- **E2/E3 主方法是固定 K 后处理**：K 由配置文件给定，编码器先产生 embedding，随后每个 intent
  内部才执行 KMeans；KMeans 中心不进入 MiniLM 优化器，也没有梯度回传。
- **RACAL 阶段一是可训练表示，但只有 K=1**：它训练 MiniLM 最后两层和 residual projection，
  但没有训练多个中心，也没有学习中心数。
- **RACAL 阶段二只是固定 K=2 归因实验**：它复用阶段一 checkpoint，再对每个 intent 做 KMeans-2；
  没有重新训练表示，也没有根据 calibration 学习中心激活规则。
- **RC-AMBL 才是一次自适应结构原型**：它尝试根据 Known-only 风险和稳定性接受/拒绝分裂，
  但第一轮所有候选分裂都被安全门拒绝，最终所有 intent 都回退到 `K_y=1`，因此没有形成成功的
  adaptive-K 结果。

因此，当前最准确的方法命名是：

```text
主协议：fixed-K post-hoc multi-centroid Gate
当前最佳自有结果：Known-only trainable MiniLM + K=1 Gate
已尝试但未成功：RC-AMBL adaptive split pilot
尚未完成：joint representation–center training + calibration-selected K_y
```

真正的自适应多中心至少应同时具备三点：

1. 中心/子中心的产生或更新参与表示训练，或至少与表示学习交替更新；
2. 每个 intent 的 `K_y` 不是预先固定，而是由不使用测试 OOS 的规则决定；
3. 训练、结构选择和边界校准形成一个可复现的 Known-only 闭环。

目前三点尚未在 s2c 中同时实现并通过实验验证。

### 2.1 正式 protocol_v2 Gate

主要入口是：

- `src/protocol_v2/experiments/runner.py`
- `src/protocol_v2/gate/multi_sphere_oos_detector.py`
- `configs/methods/frozen_minilm_gate.yaml`

数据流如下：

```text
固定 canonical / registry / views
        ↓
本地 all-MiniLM-L6-v2 句向量
        ↓
L2 归一化
        ↓
每个 Known intent 内拟合 1 个或 K 个中心
        ↓
每个中心估计对角协方差与半径
        ↓
Gate score 与 Known/OOS 判定
```

代码中的 `MultiSphereOOSDetector` 有三个中心模式：

1. `class_centroid`：每个意图一个中心；
2. `class_centroid_mixture`：每个意图内部执行 KMeans，得到固定数量的子中心；
3. `kmeans`：全体样本做全局 KMeans，主要用于历史或诊断对照。

正式 protocol_v2 使用 `class_centroid_mixture`，并把 `subcenters_per_intent` 设为实验配置中的
`K`。因此当前正式多中心方法的本质是：

> **冻结或已给定的表示 + 每个 Known intent 内固定 K 的 KMeans + 每个子簇一个球形/对角马氏边界。**

### 2.2 距离、半径和判定语义

正式 Gate runner 在 `runner.py::_build_detector` 中固定了：

- `center_mode="class_centroid_mixture"`；
- `distance_metric` 为 `euclidean` 或 `mahalanobis_diag`；
- `radius_method="mean_std"`；
- `radius = mean(distance) + lambda * std(distance)`；
- `lambda=1.0`；
- `threshold=1.0`；
- `l2_normalize=True`；
- `random_state=42`。

对角马氏距离在每个簇内按维度估计方差：

```text
d(x,c) = sqrt(sum((x-c)^2 / (var + epsilon)))
r      = mean(d_train) + lambda * std(d_train)
```

默认 `acceptance_mode="nearest_sphere"`。它先找原始距离最近的球，再检查：

```text
d_nearest(x) <= r_nearest
```

代码还保留了显式的 `normalized_union` 语义，但它不是 E2/E3 的默认协议；该模式会按
`min_j d_j/r_j` 判断是否有任意球接收样本。两种语义不能混写，否则会改变历史 E2/E3 结果。

### 2.3 当前最强的自有方法：RACAL-v1 的 Trainable K=1

RACAL-v1 阶段一不是另一个完整 Cascade，而是一个 Gate 表示训练控制。主要代码是：

- `src/protocol_v2/experiments/racal_v1/representation.py`
- `src/protocol_v2/experiments/racal_v1/runner.py`
- `configs/experiments/protocol_v2_textoir_v1/racal_v1.yaml`

它的表示路径为：

```text
MiniLM AutoModel
    ↓ mean pooling
384D residual projection（384 → 256 → 384）
    ↓ L2 normalize
Gate detector
```

训练约束是：

- 只使用 Known train；
- 每轮刷新 Known 类中心；
- 使用基于中心距离的 CE；
- 加入类内紧致项和类间 margin 项；
- 先只训练 projection，再解冻 MiniLM 最后两层；
- checkpoint 只用 Known calibration 的 `F1-K + 0.05 × Known Recall` 选择；
- 不使用测试 OOS 选 epoch、半径、阈值或 K。

所以它目前准确的名称是：

> **Known-only residual-adapted MiniLM + single-centroid Gate**，而不是已完成的 adaptive-K 方法。

## 3. 已完成的自有实验和结论

| 阶段 | 做了什么 | 完成情况 | 当前结论 |
|---|---|---:|---|
| E0 | 三数据集 canonical、registry、views、exports、TEXTOIR runtime independence | complete | 数据和协议可以独立运行 |
| E1 | 三数据集 Gate smoke | 36/36 | 工程链路可运行 |
| E2 | 3 数据集 × 11 KIR × 5 seed × K=1..5 × 2 distance | 1650/1650 | 不存在跨数据集统一最优 K |
| E3 | KMeans/random-balanced、ARI、tiny-cluster、Known-only coverage | 720 cells + 180 诊断组 | 稳定聚类不等于有效 OOS 边界 |
| R1/M1 | CE-Recon、SupCon、Geometry 等表示探索 | 已完成但按 contract audit superseded | 几何指标改善不等于多中心恢复 |
| λ audit | lambda 网格、split 不重叠、Known-only 选择审计 | complete | 当前主协议不使用 test OOS 调 lambda |
| URCSG | Known-only leave-one-intent 风险选择 K | 6/6 | Banking77 条件性收益不足，StackOverflow 仍退化 |
| CCSG | 类级 mixture support、margin 和聚合消融 | 9/9 | 仅改变聚合不能稳定救活 K=2 |
| RC-AMBL | PCA split、父边界、收缩协方差、风险门 | 6/6 | 所有候选分裂被拒绝，未形成成功 adaptive-K |
| RACAL stage1 | Frozen K=1 回放 vs Trainable MiniLM K=1 | 3/3 + 3/3 | 当前最可靠的正收益来自 K=1 表示适配 |
| RACAL stage2 | 同一 Trainable checkpoint 下纯 K=1 vs 固定 K=2 | 3/3 | K=2 的 OOS 过接受仍然严重，停止 K=3--5 |

### 3.1 RACAL 阶段一：当前最好的自有结果

StackOverflow、KIR=0.50、seed 13/42/87：

| 方法 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|
| Frozen K=1 | 0.7729 ± 0.0417 | 0.7860 ± 0.0143 | 0.8371 ± 0.0013 | 0.2654 ± 0.0632 |
| Trainable K=1 | **0.8671 ± 0.0079** | **0.8565 ± 0.0030** | **0.8392 ± 0.0032** | **0.1114 ± 0.0165** |

相对 Frozen K=1：

- OOS F1 `+9.42pp`；
- F1-All `+7.06pp`；
- Known Recall `+0.21pp`；
- false acceptance `-15.40pp`；
- AUROC `+3.56pp`。

这说明表示训练确实有效，而且不是靠明显牺牲 Known Recall 换来的。

### 3.2 RACAL 阶段二：固定 K=2 的反证

同一批 Trainable MiniLM checkpoint、同一数据和边界，仅把每个 intent 从 K=1 改为 K=2：

| 方法 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|
| Trainable K=1 | 0.8671 ± 0.0079 | 0.8565 ± 0.0030 | 0.8392 ± 0.0032 | 0.1114 ± 0.0165 |
| Trainable fixed K=2 | 0.6765 ± 0.0615 | 0.7681 ± 0.0126 | 0.9362 ± 0.0029 | 0.4526 ± 0.0782 |
| K=2 − K=1 | **−19.06pp** | **−8.85pp** | **+9.70pp** | **+34.11pp** |

三个 seed 的 OOS F1 都下降；K=2 新增接受的 OOS 数量为 1169、753、1154，远多于恢复的
Known 样本 298、309、285。结论是：

> **当前主要问题不是“MiniLM 没训练好”，而是固定多球组合扩大了错误接受区域。**

## 4. 同协议下与 MOGB 风格组件的比较

下面的结果来自 `results/final_baselines/summary.csv`，覆盖相同的
`protocol_v2_textoir_v1`、KIR=0.50、Frozen MiniLM、5 个 seed。它们可以作为组件级公平证据，
但不是 MOGB 论文原始 BERT 结果。

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall |
|---|---|---:|---:|---:|
| CLINC150 | Single centroid | 88.94 | 80.27 | 78.76 |
| CLINC150 | Fixed K=2 | 89.20 | 80.09 | 75.14 |
| CLINC150 | Random partition | 89.07 | 80.75 | 79.48 |
| CLINC150 | MOGB-MiniLM | 81.32 | 44.95 | 31.57 |
| CLINC150 | MOGB partition + s2c boundary | 85.56 | 64.60 | 53.34 |
| Banking77 | Single centroid | 72.43 | 74.51 | 85.01 |
| Banking77 | Fixed K=2 | 75.46 | 76.16 | 82.55 |
| Banking77 | Random partition | 71.62 | 74.72 | 85.96 |
| Banking77 | MOGB-MiniLM | 74.99 | 48.60 | 33.41 |
| Banking77 | MOGB partition + s2c boundary | **79.40** | 64.81 | 52.18 |
| StackOverflow | Single centroid | 76.55 | 79.98 | 87.15 |
| StackOverflow | Fixed K=2 | 63.53 | 72.76 | 86.89 |
| StackOverflow | Random partition | 75.88 | 79.80 | 87.64 |
| StackOverflow | MOGB-MiniLM | 72.92 | 43.30 | 27.09 |
| StackOverflow | MOGB partition + s2c boundary | **79.25** | 63.34 | 50.39 |

解释：MOGB 风格粒球在 Banking77 和 StackOverflow 上的 OOS F1 比固定 K=2 高，但 Known Recall
和 F1-All 明显更低；这表明 OOS F1 的提升主要来自更激进的 Known 拒绝，不能直接称为综合性能提升。

## 5. 与 MOGB、ADB、DA-ADB、DCLOOS 的差异

### 5.1 MOGB

MOGB 完整逻辑不是“固定 MiniLM 后处理 KMeans”，而是：

```text
BERT 表示训练
→ 自适应 granular-ball 划分
→ 最近子中心损失
→ 重新优化表示
→ 再划分粒球
→ 粒球中心 + 平均半径边界
```

当前有两类 MOGB 证据：

1. **同协议 MOGB 风格组件比较**：使用 Frozen MiniLM，只替换粒球划分或边界；可以回答组件贡献，
   但不是完整 MOGB。
2. **官方 BERT 兼容运行**：StackOverflow KIR=.50 seed=0 和 Banking77 KIR=.75 seed=0
   的本地结果均明显低于论文参考值，状态为 `not_reproduced_strict`；原作者配套数据缺失，
   不能把本地数字当成对论文 MOGB 的公平否定。

因此当前不能写“s2c 超过 MOGB 论文”。只能写：

> 在相同 Frozen MiniLM 协议下，s2c 单中心的综合指标更稳；MOGB 风格组件在部分数据集提高
> OOS F1，但代价是 Known Recall/F1-All 下降。官方 MOGB 论文结果尚未被本地严格复现。

### 5.2 ADB 和 DA-ADB

现有 ADB/DA-ADB 是 BERT、单 cell 的现代兼容结果：

| 方法 | 数据/设置 | OOS F1 | F1-All | Accuracy | 状态 |
|---|---|---:|---:|---:|---|
| ADB | StackOverflow, KIR=.50, seed=0 | 89.47 | 87.63 | 88.53 | compatibility artifact |
| DA-ADB | StackOverflow, KIR=.50, seed=0 | **90.90** | **89.23** | **90.07** | compatibility artifact |

它们不是与 Frozen MiniLM s2c 的同协议五 seed 对比，不能直接作为严格排名；但它们说明当前
s2c 仍有明显的强基线差距，尤其是 F1-All 和 Accuracy。

### 5.3 DCLOOS

DCLOOS 是端到端方法，训练中使用特征级伪 OOS 和外部开放域 OOS；它与当前 Known-only Gate
的监督条件不同。

- 官方完整单元：超时，没有正式最终指标；
- reduced-budget 恢复单元：OOS F1 87.05、F1-All 90.26、Known Recall 92.14、Accuracy 88.68；
- 该结果是 KIR=.75、seed=888，并使用 pseudo-OOS + 外部 SQuAD OOS，不能和当前 Frozen
  MiniLM KIR=.50 结果直接排名。

所以 DCLOOS 当前只能说明：在更强 OOS 监督下，它的兼容运行结果高于当前 Frozen Gate；并不说明
在相同训练条件下已经公平超过 s2c。

## 6. 当前结果究竟说明了什么

### 已经有证据支持的结论

- Known-only 表示适配可以显著改善 K=1 Gate；
- 固定 K>1 没有跨数据集统一收益；
- Banking77 的多中心收益是条件性的；
- CLINC150 的多中心收益较弱；
- StackOverflow 的固定多中心主要出现 OOS 过接受；
- 聚类稳定、MOGB 风格粒球或改变聚合规则，并不能自动保证 OOS 边界有效。

### 目前不能声称的结论

- 不能声称当前 adaptive-K 已经成功；
- 不能声称 s2c 超过 MOGB 论文；
- 不能声称 s2c 超过 DCLOOS；
- 不能把 BERT 单 cell 的 ADB/DA-ADB 结果当成同协议五 seed 排名；
- 不能把历史 Cascade 结果与当前 Gate-only 结果混成一张表；
- 不能把 MOGB 官方负复现当作论文方法本身失败。

## 7. 当前唯一下一步

当前不应再跑 K=3--5，也不应继续增加新的损失项。若坚持验证自适应多中心，只允许登记一个
最小实验：

```text
StackOverflow / KIR=.50 / seed=13,42,87
Trainable MiniLM checkpoint 固定
每个 intent 只允许 K=1 或 K=2
激活规则只使用 train + Known calibration
其余 intent 保持 K=1
```

它只回答“Known-only 风险门能否识别少数安全的 K=2 intent”。不能使用阶段二 test OOS 结果调规则。
如果该实验仍然无法同时保持 OOS F1、F1-All 和 false acceptance，则应停止当前固定球形多中心路线，
把 Trainable K=1 作为当前自有最佳 Gate，并转向统一协议的强基线覆盖或新的边界几何设计。

## 8. 证据文件索引

| 内容 | 文件 |
|---|---|
| 当前总状态 | `docs/CURRENT_STATUS.md` |
| 实验总账 | `docs/EXPERIMENT_LEDGER.csv` |
| E2/E3 总体审计 | `docs/audits/protocol_v2_implementation/` |
| MOGB/DCLOOS 中文对比 | `docs/对比实验/MOGB_DCLOOS_对比结果报告.md` |
| RACAL 阶段一 | `docs/racal_v1/RACAL_V1_REPORT.md`、`RACAL_V1_CLOSEOUT.md` |
| RACAL 阶段二 | `docs/racal_v1/RACAL_V1_STAGE2_REPORT.md`、`RACAL_V1_STAGE2_CLOSEOUT.md` |
| 同协议基线 CSV | `results/final_baselines/summary.csv`、`results/mogb/fair_matrix.csv` |
| MOGB 复现审计 | `results/diagnostics/mogb_diff/`、`results/mogb_exact_reproduction*/` |
| DCLOOS 审计 | `docs/archive/external_baselines/dcloos/DCLOOS_REPRODUCTION_REPORT.md` |
| 阶段二完整 artifact | `../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/stage2_fixed_k2/` |
