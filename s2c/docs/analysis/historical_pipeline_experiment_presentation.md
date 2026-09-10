# Gate → Router → Expert Pipeline 实验阶段汇报

更新时间：2026-09-01  
汇报对象：导师 / 课题组  
实验主线：archive v19 数据协议，`KIR=.50`、`seed=42`

> 本汇报回答两个问题：为什么当前 OOS F1 与论文结果有差距？以及，单纯 Gate 的 Trainable
> MiniLM 改善是否真正传递到了完整 pipeline？

## 一页结论

1. **就 OOS F1 而言，Gate replay 与 full pipeline 完全一致。** OOS 一旦被 Gate 拒绝，就不会
   进入 Router/Expert；原始和 tuned pipeline 的 OOS F1 与 Gate replay 最大绝对差异均为 `0`。
2. **参数确实影响结果。** 在验证集选择 `K`、`lambda`、threshold 和接受规则后，测试集确认：
   CLINC150 OOS F1 从 `89.56%` 提升到 `91.36%`，StackOverflow 从 `91.49%` 提升到
   `91.85%`，标准 Banking77 从 `83.50%` 提升到 `85.62%`。
3. **与论文的数值对比：** 当前 tuned candidate 在 StackOverflow 超过论文；CLINC150 只差
   `0.60 pp`；标准 Banking77 仍差 `2.61 pp`。这些是 descriptive reference，因为
   `fulltex.tex` 的 reported Banking77、archive `banking77` 和当前 H1 `banking77_oos`
   不是一条完整同合同链。
4. **多中心不是普遍有效。** CLINC150 和标准 Banking77 的特定 workpoint 有收益；StackOverflow
   的验证最优配置仍是 `K=1`，这与之前固定 K=2 的负结果一致。

最终建议：CLINC150 和 StackOverflow 采用 tuned Partial candidate 做下一轮多 seed 确认；
标准 Banking77 保留 Partial K=2 作为最高 OOS 候选，同时保留 LoRA K=4 作为更稳的 Known
coverage 备选；不要把这些 seed42 development candidate 直接写入论文主表。

## 1. Pipeline 与实验问题

完整链路为：

```text
输入文本
   │
   ▼
Gate：判断 Known / OOS
   ├── OOS → 直接输出 OOS
   └── Known → Router：预测 domain → Expert：预测 intent
```

因此，Router/Expert 只能改变 Known 样本的最终 intent，不会改变已经由 Gate 判为 OOS 的样本。
这也是为什么需要把 `OOS F1` 与 `full-pipeline macro F1` 分开：前者检查 Gate，后者还包含
下游 Known intent 分类。

### 实验分层

| 实验 | 数据 / 协议 | 目的 | 状态 |
|---|---|---|---|
| Archive baseline | archive `clinc150`、`stackoverflow`、标准 `banking77` | Frozen / Partial / LoRA 的 K=1 全链路对照 | 9 个单元完成 |
| OOS workpoint search | 同上；验证集含 OOS 标签 | 搜索 K、lambda、threshold、接受规则 | 完成 |
| Tuned full pipeline | 验证选择的候选接入下游 | 确认 OOS 改善是否传递 | 完成 |
| Historical H1 controlled reference | `banking77_oos` H1 | 当前 Trainable 主线；与 `fulltex.tex` Banking77 只作 descriptive 对照 | 已有 3 seed 参考 |

固定组件：对角 Mahalanobis、`mean + lambda × std`、mean pooling、L2 normalize、SmolLM
Router/Expert + LoRA。新 Router/Expert 训练均在 RTX 5070 CUDA 上完成。

## 2. 当前候选与论文结果

下表以论文 `fulltex.tex` 的 KIR=.50 `Ours` 行为参考；当前候选按 OOS F1 优先选择。

| 数据集 | 当前 tuned 配置 | 当前 Known F1 | 当前 OOS F1 | 当前 Accuracy | 论文历史 Known F1 | 论文历史 OOS F1 | 论文历史 Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | Partial K=3, λ=2, normalized union | 85.94% | 91.36% | 87.35% | 79.95% | 91.96% | 86.78% |
| StackOverflow | Partial K=1, λ=2, threshold=.85 | 86.36% | **91.85%** | 88.73% | 75.48% | 89.71% | 85.54% |
| 标准 Banking77 | Partial K=2, λ=1, threshold=.95 | 81.12% | 85.62% | 82.44% | 74.90% | 88.23% | 78.98% |

这里的“论文历史参考”不是同一任务上的重新测量：

- CLINC150 和 StackOverflow 的当前候选来自 archive seed42；
- 标准 Banking77 的当前候选与论文历史 `banking77_oos` 只做数值参考，不是同任务复现。

直接结论：

- CLINC150：当前结果距离论文 `0.60 pp`，Known F1 和 Accuracy 高于论文参考值；
- StackOverflow：当前 OOS F1 高于论文 `2.14 pp`；
- 标准 Banking77：当前 OOS F1 低于论文 `2.61 pp`，但任务并不相同，不能据此判定 pipeline 失败。

![Fig. 1：当前候选与论文 OOS F1 对比](../../figures/historical_archive_tuned_full_pipeline/pipeline_oos_f1_vs_paper.png)

图 1 只展示 OOS F1，柱顶是精确数值，灰色柱为论文参考。Banking77 面板应理解为
“标准 archive Banking77 与论文历史参考的数值并列”，不是严格 byte-identical 复现。

## 3. Fig. 2：Gate 改善是否传递到完整 pipeline？

![Fig. 2：Gate 与 full pipeline 的 OOS 一致性](../../figures/historical_archive_tuned_full_pipeline/pipeline_gate_full_oos_invariance.png)

图 2 的横轴分别是 Gate 和 Full pipeline；每条线两端重合，三个数据集的 OOS F1 标注均为
`Δ=0.00 pp`。这直接排除了“Router/Expert 把 OOS F1 降下来了”的解释。

原始 K=1 与 tuned candidate 的变化都发生在 Gate：

| 数据集 | 原始当前 OOS F1 | tuned OOS F1 | 提升 |
|---|---:|---:|---:|
| CLINC150 | 89.56% | 91.36% | +1.79 pp |
| StackOverflow | 91.49% | 91.85% | +0.36 pp |
| 标准 Banking77 | 83.50% | 85.62% | +2.12 pp |

full-pipeline macro F1 仍会受到下游分类影响，但这属于 Known intent 分类误差，不属于 OOS
拒识误差。比如 CLINC tuned 后 full macro F1 为 `81.67%`，StackOverflow 为 `85.78%`，
标准 Banking77 为 `78.70%`。

## 4. Fig. 3：OOS 提升付出了什么代价？

![Fig. 3：OOS error budget](../../figures/historical_archive_tuned_full_pipeline/pipeline_error_budget_oos_tradeoff.png)

图 3 把 tuned candidate 的代价拆成三个可读的错误来源：OOS 被错误接受、Known 被错误拒绝、
Expert 预测错误。

- **CLINC150**：tuned 后 Known rejection 从 `27.2%` 降到 `14.2%`，OOS acceptance 从
  `3.6%` 增到 `7.6%`，但整体 OOS F1 上升到 `91.36%`。这是一个 coverage–rejection
  workpoint 调整，而不是所有错误同时下降。
- **StackOverflow**：Known rejection 从 `16.0%` 降到 `14.5%`，OOS acceptance 仅从
  `2.2%` 增到 `2.7%`，因此 OOS F1 小幅上升，full macro F1 也略升。
- **标准 Banking77**：OOS acceptance 从 `16.8%` 降到 `6.3%`，但 Known rejection 从
  `16.5%` 增到 `25.9%`；它是更明显的 OOS 优先候选，Known coverage 代价需要在多 seed
  中继续确认。

如果标准 Banking77 更重视 Known 稳定性，可使用 [LoRA K=4 候选结果](../../../artifacts/s2c/runs/historical_archive_tuned_lora_extended_full_pipeline_seed42/banking77/kir50_seed42/per_method.csv)：
`OOS F1=85.10%`、`Known F1=83.99%`、`Accuracy=82.14%`。它比 Partial K=2 的 OOS F1 低
`0.52 pp`，但 Known coverage 更稳。

## 5. 为什么之前说多中心不好？

之前的负结论针对的是固定协议下的普遍性：

- Trainable 固定 K=2 控制中，StackOverflow OOS F1 比 K=1 低 `19.06 pp`，False Acceptance
  增加 `34.11 pp`；
- 冻结表示的历史 K 消融中，StackOverflow Diag-Mahalanobis K=2 为 `72.80%`，K=1 为
  `79.02%`；
- 因此不能把“增加中心”直接等同于“更好的 OOS 边界”。

本轮 CLINC 的 `K=3` 和标准 Banking77 的 `K=2` 属于不同条件：使用 Trainable 表示、不同
lambda/threshold、不同接受规则，并且候选由验证 OOS 工作点选择。StackOverflow 最优仍为 K=1，
所以本轮并没有推翻原结论，而是补充了“多中心收益具有数据集条件性”。

## 6. 为什么当前结果与论文有差距？

差距主要来自实验合同，而不是 Gate→pipeline 的实现断裂：

1. 当前 archive baseline 使用 `K=1`，论文主方法使用 `K_y=2`；
2. 当前 baseline 固定 `lambda=1`、threshold=1，论文历史实现允许更强的 OOS validation
   workpoint 信息；
3. 当前标准 `banking77` KIR=.50 使用 `38` 个 Known intents 和 `39` 个未知 intent；历史 Ours
   artifact 的实际数据键是 `banking77_oos`，KIR=.25 为 `12` Known、`66` OOS。论文表头写作
   `Banking77`，而 archive `banking77` 是另一条标准数据线；不能把两条线混合；
4. 论文主表是历史完整 Cascade；本轮 tuned 结果是 archive 数据上的 K=1 development
   candidate，不能直接回填论文表。

作为当前 H1 的 historical reference，`banking77_oos` H1 Trainable full-pipeline 三 seed OOS F1
均值为 `88.47%`，数值上略高于 `fulltex.tex` KIR=.50 的 `88.23%`；但 Gate、semantic calibration
和下游模型合同不同，不能把它写成同方法严格超过论文。标准 `banking77` 的 `85.62%` 仍属于
另一条 archive 数据线。

## 7. 最终建议与下一步

### 当前建议

- CLINC150：保留 `Partial K=3, λ=2, normalized_union`；
- StackOverflow：保留 `Partial K=1, λ=2, threshold=.85`；
- 标准 Banking77：OOS 优先用 `Partial K=2, λ=1, threshold=.95`，Known 优先可用 LoRA K=4
  候选；
- 这些配置先作为 seed42 development candidate，不宣布为跨 seed 稳定默认。

### 下一步

用相同候选配置在 `seed={13,42,87}` 上做确认，指标同时报告：OOS F1、False Acceptance、
Known F1、Known Recall、Accuracy 和 full macro F1。若 Banking77 的多中心候选在多 seed 中
持续提升 OOS 且 Known coverage 可接受，再决定是否作为正式 pipeline workpoint。

## 8. 可复核入口

- 论文对比表：[comparison_to_paper.csv](../../results/analysis/historical_archive_tuned_full_pipeline/comparison_to_paper.csv)
- tuned full-pipeline 结果：[comparison_to_paper.csv](../../results/analysis/historical_archive_tuned_full_pipeline/comparison_to_paper.csv)，包含 full macro F1 和错误预算字段
- 图清单：[FIGURE_MANIFEST.json](../../results/analysis/historical_archive_tuned_full_pipeline/FIGURE_MANIFEST.json)
- 图源：[build_historical_pipeline_experiment_figures.py](../../tools/analysis/build_historical_pipeline_experiment_figures.py)
- workpoint 搜索：[search_historical_archive_gate_workpoints.py](../../tools/analysis/search_historical_archive_gate_workpoints.py)
- 详细实验报告：[HISTORICAL_ARCHIVE_FULL_PIPELINE_REPORT.md](HISTORICAL_ARCHIVE_FULL_PIPELINE_REPORT.md)

所有图均由 Python/Matplotlib 从聚合结果生成；PNG 为汇报预览，SVG/PDF 保留可编辑矢量版本。
测试 OOS 只用于候选确认，不参与搜索选择；本轮未修改 `fulltex.tex`，未覆盖历史 checkpoint，
未提交 raw prediction 或 embedding。
