# 为什么当前可训练 MiniLM 结果低于 `fulltex.tex` 的历史数字

## 结论先行

当前 Trainable MiniLM 的 StackOverflow/KIR=0.50、K=1 结果为
`OOS F1=0.8671±0.0079`。它低于论文表中的 `Ours=0.8971`，但这**不能解释为“训练 MiniLM 反而变差”**。两者不是同一个实验合同：

1. 论文表是历史系统级结果，当前 RACAL 是 Gate-only 结果；
2. 论文使用历史 StackOverflow 快照、旧 split 和旧下游配置，当前使用
   `protocol_v2_textoir_v1` 的固定数据/registry；
3. 论文表的历史 artifact 对 `oos_f1` 和 `overall_accuracy` 存在显式
   `main_table_ours` 覆盖，原始 `primary_metrics.oos_f1` 并不是 0.8971；
4. 论文方法使用冻结 MiniLM + 固定多中心，并配有历史语义 Gate、Router/Expert
   和历史阈值/配置；当前 Trainable MiniLM 使用 Known-only 训练、K=1、当前
   Gate 评价器，没有这些历史后处理。

因此，当前结果最可靠的解释是：**Trainable MiniLM 已经显著优于当前同协议 Frozen
K=1，但尚未复现历史系统级数字；历史数字本身也不能直接作为当前 Gate-only 的公平目标。**

## 1. 当前可比的结果

RACAL-v1 在同一 `protocol_v2_textoir_v1`、StackOverflow、KIR=0.50、3 个 seed 下，结果为：

| 方法 | OOS F1 | F1-All | Known Recall | False acceptance |
|---|---:|---:|---:|---:|
| Frozen K=1 | 0.7729±0.0417 | 0.7860±0.0143 | 0.8371±0.0013 | 0.2654±0.0632 |
| Trainable MiniLM K=1 | **0.8671±0.0079** | **0.8565±0.0030** | 0.8392±0.0032 | **0.1114±0.0165** |

训练相对冻结表示带来：

- OOS F1 `+9.42` 个百分点；
- F1-All `+7.06` 个百分点；
- Known Recall `+0.21` 个百分点；
- false acceptance `-15.40` 个百分点。

这说明当前训练契约对**单中心 Gate**是有效的。证据见
[`RACAL_V1_REPORT.md`](../racal_v1/RACAL_V1_REPORT.md:22) 和
[`RACAL_V1_STAGE1_MEAN_STD.csv`](../../results/diagnostics/racal_v1/RACAL_V1_STAGE1_MEAN_STD.csv)。

更重要的是，同一当前协议、同一 KIR 和同一 13/42/87 seed 的 E2 Gate-only K=1
精确参考为 `0.7729±0.0417`；另一个历史汇总 CSV 的 `0.7902` 使用了不同的聚合口径，
不应与本批训练结果混作逐 seed 对照。因此 Trainable K=1 相对严格配对 Frozen
提高约 `9.42` 个百分点。这个比较比直接对照论文主表更合理。

## 2. `fulltex.tex` 的历史数字是什么

论文正文明确写的是：MiniLM **不训练**，之后做 L2 归一化和 KMeans 多中心
（[`fulltex.tex`](../../fulltex.tex:229)；中心和边界见
[`fulltex.tex`](../../fulltex.tex:242)）。同时，历史 Cascade 还配置了
MiniLM Gate、SmolLM-135M Router/Expert、LoRA、`K_y=2` 和数据集相关的
`lambda`（[`fulltex.tex`](../../fulltex.tex:329)）。论文在表格前也明确把主表称为
“historical comparison protocol”，并把当前的 Gate-only 对照另列
（[`fulltex.tex`](../../fulltex.tex:362-364)）。

所以表中的 StackOverflow/KIR=0.50 `Ours=89.71` 是历史系统级数字，不是
“训练后的 MiniLM + K=1 Gate”数字。

## 3. 历史 `0.8971` 还存在指标覆盖问题

历史 StackOverflow artifact
`artifacts/s2c/outputs/paper_results/stackoverflow/kir50_seed42/full_anchor/eval_results.json`
中同时存在：

- `metrics.oos_f1 = 0.8971`；
- `metrics.primary_metrics.oos_f1 = 0.5910`；
- `metrics.primary_metrics.overall_accuracy = 0.6342`；
- `metrics.oos_by_source.heldout_oos.gate_false_accept_rate = 0.5332`。

对应的历史汇总还记录了：

```text
metric_override_source = main_table_ours
metric_override_fields = oos_f1,overall_accuracy
```

因此 `0.8971` 是论文主表兼容值/覆盖值，不能当作当前评价器重新计算出的原始
二分类 Gate F1。该 artifact 应作为历史参考保留，但不能用来证明当前 Trainable
MiniLM “差了 3 个百分点”。

## 4. 四个造成表面差距的具体原因

### 4.1 Gate-only 与完整 Cascade 不同

当前 RACAL 的训练和测试链路是：

```text
MiniLM → residual projection → K=1 Gate → OOS/ID
```

没有运行历史表中的 Router/Expert 和历史语义后处理。历史 artifact 的配置还包含
`semantic_gate_enabled=true`、prototype semantic gate、fusion 和固定语义阈值。
这些步骤会改变 OOS 接受/拒绝结果，因此不能把历史全链路输出与当前 Gate-only
输出直接比较。

### 4.2 数据和 split 不是同一个版本

当前 RACAL 使用 `protocol_v2_textoir_v1` 的 canonical、registry 和 views；每个
seed 的当前阶段报告为 train/calibration/test=`6000/1000/6000`，且 test Known/OOS
各约 3000。历史 StackOverflow artifact 使用的是
`stackoverflow20k_seeded_random_20260422` 路径；其记录的 test Known/OOS 为
`2993/2997`，sample ID、Known intent 列表和旧 registry 不能默认相同。

开放意图检测对已知意图列表、held-out intent 和样本划分非常敏感。即使总样本数都
是 20,000，也不能据此认为两个结果可直接比较。

### 4.3 训练目标改变了表示几何，但没有 OOS 负样本

当前 Trainable MiniLM 使用 mean pooling、384 维残差 projection、解冻最后两个
Transformer block，损失为 Known intent CE、类内紧致和异类 margin，checkpoint
仅由 Known calibration 选择（见
[`RACAL_V1_REPORT.md`](../racal_v1/RACAL_V1_REPORT.md:22-27)）。

它优化的是 Known 类别可分性与紧致性，不直接优化 OOS rejection。当前结果表明这对
K=1 有帮助，但固定 K=2 会把局部球并集扩大：

| 方法 | OOS F1 | Known Recall | False acceptance |
|---|---:|---:|---:|
| Trainable K=1 | 0.8671 | 0.8392 | 0.1114 |
| Trainable fixed K=2 | 0.6765 | 0.9362 | 0.4526 |

因此问题不是“MiniLM 没训练好”，而是**表示训练收益没有传递为安全的多球边界**。

### 4.4 历史表的多中心和当前 K=1 也不同

历史主表配置 `K_y=2`，当前 Trainable 结果是 K=1。当前同一 checkpoint 的 K=2
会提高 Known Recall，却让 OOS false acceptance 增加约 `34.11` 个百分点，OOS F1
下降约 `19.06` 个百分点（[`RACAL_V1_STAGE2_REPORT.md`](../racal_v1/RACAL_V1_STAGE2_REPORT.md:21-64)）。

所以不能期待“把当前 K=1 训练结果”自动达到历史 `K_y=2` 的系统数字；当前实验反而
揭示了历史配置中必须单独审计的多中心接受区域和后处理差异。

## 5. 目前应该如何解释这组结果

应采用下面的三层结论，而不是一句“训练 MiniLM 不如论文”：

1. **同协议表示结论：成立。** Trainable MiniLM 明显优于 Frozen K=1，且 Known Recall
   几乎不损失。
2. **多中心结论：尚未成立。** 在当前协议和当前训练表示上，固定 K=2 发生严重
   false acceptance，训练没有解决多球并集风险。
3. **历史论文复现结论：未完成且存在口径问题。** `fulltex.tex` 主表是历史系统级
   结果，且关键 OOS/Accuracy 字段有 override；它不能作为当前 Gate-only 训练结果的
   直接 benchmark。

## 6. 下一步最有价值的实验

不要先继续调更多 MiniLM loss。应先做一个**同一当前 protocol 的端到端闭环**：

1. 固定当前 Trainable K=1 checkpoint；
2. 使用当前 protocol 的 Router/Expert 或同等固定下游；
3. 分别接入 Frozen K=1、Trainable K=1 和当前最强安全 Gate；
4. 用统一的原始指标重新计算 OOS F1、F1-All、F1-K、Accuracy、Known Recall；
5. 同时保存 raw metric 和任何兼容性/论文表 override，绝不混写。

只有这一步完成后，才能回答“训练表示的收益是否传递到完整系统”。在此之前，
`0.8671` 与 `0.8971` 的差距主要是实验合同差异，不是已经证明的模型能力差距。

## 7. 新增的 λ/K 受控证据：不是 λ=1 单点造成的差距

为排除“Trainable MiniLM 只是使用了不合适的半径系数”这一解释，
`minilm_trainable_lambda_control_v1` 在完全相同的 checkpoint、split、距离和阈值下评价
λ={0.50,0.75,1.00,1.25,1.50,2.00}，并只用 Known calibration 选择每个
dataset×seed×K 的最小可行 λ。该阶段共 108/108 个评价单元，`test_used_for_selection=false`
且 `oos_used_for_training=false`。

Known-only 选择后的 K=2−K=1 差值为：

| 数据集 | OOS F1 Δ | F1-All Δ | Known Recall Δ | False acceptance Δ |
|---|---:|---:|---:|---:|
| CLINC150 | +0.79pp | +0.07pp | -1.88pp | -2.44pp |
| Banking77 | +2.08pp | +0.07pp | -1.21pp | -3.27pp |
| StackOverflow | -9.51pp | -4.61pp | +2.27pp | +11.83pp |

因此 StackOverflow 固定 K=2 的退化即使在 Known-only λ 校准后仍然存在，不能归因于
λ=1 的单点设置。更严格地说，StackOverflow K=1 的 calibration false-reject≤5% 约束在
λ≤2 的候选网格内也没有稳定可行值；λ=2 只是诊断上限，不是正式最优参数。

证据见 [`MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md`](MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md)、
[`k_delta_by_lambda.csv`](../../results/diagnostics/minilm_trainable_lambda_control_v1/k_delta_by_lambda.csv)
和 [`trainable_lambda_k_interaction.png`](../../figures/active_experiment_dashboard_v1/trainable_lambda_k_interaction.png)。

这组结果把当前解释进一步收窄为：Trainable MiniLM 的主要收益发生在单中心表示与 OOS 分数排序，
而历史 LaTeX 数字还包含不同历史数据/指标口径和完整 Cascade；它们不能被用来证明当前
Trainable K=1 应该直接达到 0.8971，也不能用调 λ 的方式自动恢复固定多中心。
