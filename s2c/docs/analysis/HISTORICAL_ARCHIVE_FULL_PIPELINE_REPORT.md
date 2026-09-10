# Archive 数据协议下的 Trainable MiniLM Gate 全链路结果

更新时间：2026-09-01

## 结论先行

在旧 archive 数据协议、`KIR=.50`、`seed=42`、`K=1`、对角 Mahalanobis、固定
`mean + 1×std` 边界下，Trainable MiniLM 确实比 Frozen MiniLM 更适合做 Gate，且这个
收益能够传递到完整的 `Gate → Router → Expert` pipeline。

但 LoRA 版本没有在三个数据集上统一超过当前 partial fine-tuning：

- CLINC150：Partial `89.56%`，LoRA `88.76%`；LoRA 低 `0.80 pp`。
- StackOverflow：Partial `91.49%`，LoRA `90.91%`；LoRA 低 `0.58 pp`。
- 标准 Banking77：Partial `82.99%`，LoRA `83.50%`；LoRA 高 `0.51 pp`。

因此本轮结论是：继续使用训练后的 MiniLM 做 Gate；当前默认选 Partial，而不是把
LoRA 作为三个数据集的统一默认。LoRA 只在标准 Banking77 的这个 seed 上显示出小幅优势，
未满足扩展到 `seed={13,42,87}` 的预设条件。

## 1. 实验合同

本轮使用的是 `archives/submissions/s2c-submission/data` 中的旧数据快照，三个任务为
`clinc150`、`stackoverflow` 和标准 `banking77`。这不是当前 `banking77_oos` 任务，也
不是论文主表的严格 H0 `Ours` 复现。

| 项目 | 固定设置 |
|---|---|
| 数据 | archive `gate/router/experts` 数据 |
| KIR | `.50` |
| seed | `42` |
| Gate | `K=1`、对角 Mahalanobis、`mean + 1×std`、`score=1` |
| semantic gate | 关闭；与 archive submission 代码一致 |
| 训练/选择 | Known train；checkpoint 只用 Known calibration 选择 |
| 测试 OOS | 不参与训练、checkpoint、阈值或结构选择 |
| 下游 | Router/Expert 使用 SmolLM + LoRA；本轮按 archive 数据重新训练 |

三套完整链路分别有 `3` 个 Gate 方法，共 `9` 个 full-pipeline 单元。每个单元的
full-pipeline 指标与 Gate replay 的最大绝对差异为 `0`。

## 2. Full pipeline 指标

百分比均为测试集指标；`FA` 是 OOS false acceptance，`Known Recall` 是已知样本被
Gate 接受的比例。

| 数据集 | Gate | OOS F1 | FA | Known Recall | full macro F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| CLINC150 | Frozen K=1 | 88.16% | 6.49% | 73.11% | 75.18% | 83.33% |
| CLINC150 | Partial K=1 | **89.56%** | **3.63%** | 72.80% | **77.32%** | **85.18%** |
| CLINC150 | LoRA K=1 | 88.76% | 4.89% | 72.27% | 76.05% | 84.16% |
| StackOverflow | Frozen K=1 | 84.36% | 14.61% | 82.93% | 78.44% | 81.07% |
| StackOverflow | Partial K=1 | **91.49%** | **2.20%** | **84.00%** | **85.41%** | **88.40%** |
| StackOverflow | LoRA K=1 | 90.91% | 3.20% | 83.83% | 84.54% | 87.65% |
| 标准 Banking77 | Frozen K=1 | 78.98% | 25.58% | **85.59%** | 77.24% | 77.27% |
| 标准 Banking77 | Partial K=1 | 82.99% | 17.12% | 82.70% | 79.16% | 80.58% |
| 标准 Banking77 | LoRA K=1 | **83.50%** | **16.79%** | 83.49% | **79.54%** | **81.04%** |

相对 Frozen，Partial 的 OOS F1 分别提升 `+1.40/+7.14/+4.01 pp`；这确认了“训练后的
MiniLM 做 Gate”这个方向在旧协议下是成立的。LoRA 相对 Partial 的变化为：

| 数据集 | Δ OOS F1 | Δ FA | Δ Known Recall | Δ full macro F1 | Δ Accuracy |
|---|---:|---:|---:|---:|---:|
| CLINC150 | −0.80 pp | +1.26 pp | −0.53 pp | −1.27 pp | −1.02 pp |
| StackOverflow | −0.58 pp | +1.00 pp | −0.17 pp | −0.88 pp | −0.75 pp |
| 标准 Banking77 | **+0.51 pp** | **−0.32 pp** | **+0.79 pp** | **+0.38 pp** | **+0.45 pp** |

## 3. Full pipeline 中到底发生了什么

OOS F1 与 Gate-only 完全一致是预期的：OOS 一旦被 Gate 拒绝，Router/Expert 不再处理；
下游只能影响被 Gate 接受的 Known 样本。因此，本轮 full pipeline 的新增价值主要体现在
`full macro F1`、整体 Accuracy 和 Known intent 的最终分类。

以 Partial 相对 Frozen 为例：

- CLINC150：OOS false acceptance 从 `6.49%` 降到 `3.63%`，同时 Expert error 从 `3.34%`
  降到 `2.81%`，最终 macro F1 提升 `2.14 pp`。
- StackOverflow：OOS false acceptance 从 `14.61%` 降到 `2.20%`，是本轮最明显的 Gate
  收益；Expert error 从 `7.45%` 降到 `5.97%`，最终 macro F1 提升 `6.97 pp`。
- 标准 Banking77：OOS false acceptance 从 `25.58%` 降到 `17.12%`，Expert error 从
  `6.30%` 降到 `5.41%`，最终 macro F1 提升 `1.92 pp`。

StackOverflow 和 Banking77 是单域任务，pipeline 对 1-class Router 使用常量路由；因此
它们的 `router_error_rate=0` 是结构性结果，不应解读为多域 Router 的泛化能力。

## 4. LoRA MiniLM 是怎么微调的

LoRA 版本严格按本轮要求实现：

- 基座：`all-MiniLM-L6-v2`，384 维输出；
- 原始 MiniLM 参数全部冻结；
- PEFT LoRA：`target_modules="all-linear"`、`r=8`、`lora_alpha=16`、
  `lora_dropout=0.1`、`bias="none"`、`task_type=FEATURE_EXTRACTION`；
- 保留 `384 → 256 → 384` residual projection，并与 LoRA 一起训练；
- mean pooling、L2 normalize、K=1、Mahalanobis、边界、loss 和 Known-only calibration
  选择规则保持不变。

| 项目 | 数量 |
|---|---:|
| 原始 MiniLM 参数 | 22,713,216 |
| 原始 MiniLM 可训练参数 | 0 |
| LoRA 参数 | 337,920 |
| residual projection 参数 | 198,016 |
| 总可训练参数 | **535,936** |
| 当前 Partial 总可训练参数 | 3,746,944 |

LoRA+projection 的可训练参数量约为 Partial 的 `14.30%`。三套 LoRA Gate manifest 均记录
`device=cuda`、`requested_device=cuda`，且 `base_trainable_parameter_count=0`。

## 5. GPU 证据

本轮新训练没有回落到 CPU：

- 三套 LoRA MiniLM Gate 的 `run_manifest.json` 明确记录 `device=cuda`；
- 标准 Banking77 Partial MiniLM 也在本轮以 `device=cuda` 重新训练；
- CLINC150 的 Router、10 个 Expert，StackOverflow 的 Router、6 个 Expert，以及标准
  Banking77 的 Router、38 类 Expert 的训练日志均记录 `Device: cuda`；
- full pipeline 三个数据集的 manifest 均记录 `device=cuda`。

需要区分一个历史事实：CLINC150/StackOverflow 的 Partial Gate 是已存在且与 archive 数据
字节一致的旧 checkpoint，本轮没有按用户要求重新训练那 9 个历史 Partial checkpoint；它们
的旧 manifest 没有记录实际训练 device。因此本轮可以完全确认的是：新 LoRA/Banking Partial
训练和全部新下游训练使用 CUDA；不能把旧 Partial checkpoint 的历史训练硬说成已被本轮 GPU
重新验证。

## 6. 与论文 `fulltex.tex` 的关系

`fulltex.tex` 的 KIR=.50 `Ours` 记录为 CLINC150/StackOverflow/Banking77 的 OOS F1
`91.96%/89.71%/88.23%`，但论文主表使用 `K_y=2`、数据与语义 Gate 的历史配置；本轮是
archive 数据上的 `K=1`、固定 `lambda=1`、关闭 semantic gate 的控制实验。因此这里的
`89.56%/91.49%/82.99%` 不能直接替换论文主表，也不能称为严格 H0 Cascade 复现。它回答
的是更窄且可复核的问题：在旧 archive 数据协议下，训练后的 MiniLM 能否作为 Gate，并且
把 Gate 的改善传递到完整 pipeline。

## 7. 可复核入口

- 轻量汇总：[comparison.csv](../../results/analysis/historical_archive_full_pipeline/comparison.csv)
- 参数与设备合同：[parameter_contract.csv](../../results/analysis/historical_archive_full_pipeline/parameter_contract.csv)
- 汇总 manifest：[MANIFEST.json](../../results/analysis/historical_archive_full_pipeline/MANIFEST.json)
- 三个 full-pipeline artifact 根：`../artifacts/s2c/runs/historical_archive_full_pipeline_seed42/`
- 训练配置：[historical_archive_full_pipeline.yaml](../../configs/experiments/historical_archive_full_pipeline.yaml)
- 运行入口：[run_historical_archive_full_pipeline.py](../../tools/eval/run_historical_archive_full_pipeline.py)
- MiniLM LoRA 实现：[representation.py](../../src/protocol_v2/experiments/racal_v1/representation.py)

本轮未修改 `fulltex.tex`，未覆盖历史 artifact，未自动 commit/push，也未把 raw prediction
或 checkpoint 写入 Git 轻量结果目录。

## 8. OOS 优先的边界调参：与论文结果直接对照

为回答“是不是参数不合适”，在验证集（含 OOS 标签）上搜索了 `K=1…3`，并对 Banking77
扩展到 `K=1…5`，同时搜索 `lambda`、score threshold 和接受规则，再把验证选择的候选接入
full pipeline；测试集只做最终确认。
当前候选与论文 KIR=.50 `Ours` 的数值对比如下：

| 数据集 | 当前候选配置 | 当前 Known F1 | 当前 OOS F1 | 当前 Acc | 论文 Known F1 | 论文 OOS F1 | 论文 Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | Partial K=3, λ=2, normalized union | 85.94% | 91.36% | 87.35% | 79.95% | 91.96% | 86.78% |
| StackOverflow | Partial K=1, λ=2, threshold=.85 | 86.36% | **91.85%** | 88.73% | 75.48% | 89.71% | 85.54% |
| 标准 Banking77 | Partial K=2, λ=1, threshold=.95 | 81.12% | 85.62% | 82.44% | 74.90% | 88.23% | 78.98% |

这组结果说明：

1. CLINC150 的 OOS F1 已距论文只差 `0.60 pp`，Known F1 和 Acc 还高于论文参考值；
2. StackOverflow 的 OOS F1 已超过论文 `2.14 pp`，Known F1 和 Acc 没有下降；
3. 标准 Banking77 仍比论文低 `2.61 pp`，但它不是论文实际使用的 `banking77_oos` 任务。
   在匹配历史 `banking77_oos` 任务的已有 H1 三 seed full-pipeline 结果中，Trainable
   OOS F1 均值为 `88.47%`，已经略高于论文的 `88.23%`。

因此，当前观察到的“pipeline 变差”不是 Gate 输出被 Router/Expert 吞掉：原始链路和调参链路
的 OOS F1 都与 Gate replay 严格一致。真正的差距来自 Gate 工作点和论文数据/结构配置，尤其是
`K=1` 对比论文 `K_y=2`，以及标准 `banking77` 对比历史 `banking77_oos`。

调参诊断结果保存在 `../artifacts/s2c/runs/historical_archive_gate_search_seed42/`；
候选全链路结果位于 `../artifacts/s2c/runs/historical_archive_tuned_full_pipeline_seed42/`，
轻量论文对比表位于
`results/analysis/historical_archive_tuned_full_pipeline/comparison_to_paper.csv`。
