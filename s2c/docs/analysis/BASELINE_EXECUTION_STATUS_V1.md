# 外部基线同协议运行状态 V1

> 这是执行状态和合同审计，不是新的 SOTA 排名。旧兼容单格数字与 protocol_v2 结果严格分层。

统一的逐样本合同与阻塞项登记见 [`UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`](UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md) 和 `results/analysis/unified_prediction_contract_v1/blocked_methods.csv`。

## 最新状态（2026-08-11；覆盖下方早期 blocker 记录）

本文下方的 runtime follow-up 保留为历史审计轨迹，不能覆盖以下最新状态：

- ADB 已完成 3 数据集 × 3 KIR × 5 seed，共 `45/45` 个有效外部 BERT/TextOIR 单元；它们与当前协议数据、Known-list 和 sample-id 可对齐，但不属于 MiniLM 同骨干 fair 排名。
- DA-ADB 已完成 StackOverflow/KIR=.50、seed=`42,87,100` 的 `3` 个有效外部单元；汇总结果为 OOS F1 `72.48±6.24%`、F1-All `74.02±3.13%`，状态是 `valid_external_cell_pending_replication`，不是 invalid，也不是已收敛的同骨干公平结果。
- TextOIR 中的 MSP、SEG、OpenMax、LOF、DOC、DeepUnk、`(K+1)-way`、MDF、ARPL、KNNCL-last/all、ADB、DA-ADB、DA-ADB-llama 等均属于应登记的 baseline；当前只有已完成最终指标且合同闭合的路线才可进入统一行级结果，其余保留 published/legacy 或 blocked 状态。
- StackOverflow、Banking77 和当前 OOS 数据与 TextOIR 对应源文件同源；尚未闭合的是训练/评估合同，不是数据内容。

## 已完成

- StackOverflow/KIR=0.50 的 protocol_v2 ADB 导出已经从 `data/exports/protocol_v2_textoir_v1/adb/` 物化到独立 artifact 数据根。
- seed=42、87、100 的 train/dev/test 和 Known labels 均复制前后 SHA256 一致；Known 列表与 registry 完全一致。
- `run_external_textoir.py` 新增显式 `--data-root` 和 `--known-labels-file`，外部 runner 不再必须读取 `textoir/data`。

## ADB/DA-ADB 当前状态

| 方法 | 同协议状态 | 说明 |
|---|---|---|
| ADB | `valid_same_protocol_external_matrix` | 3 个数据集 × 3 个 KIR × 5 个 seed，共 45/45 个逐样本单元已审计；这是 BERT/TextOIR 外部兼容合同，不是 MiniLM fair 行。 |
| DA-ADB | `valid_external_cell_pending_replication` | StackOverflow/KIR=.50 的 seed=42、87、100 已产生有限、非塌缩预测并完成汇总；仍需与 TextOIR published/legacy 合同逐项复核。 |
| ADB/DA-ADB 旧兼容单格 | `legacy_compatibility_only` | 历史 seed=0 TSV 继续保留，不与当前 protocol_v2 三 seed ADB 或 MiniLM fair 行混合。 |

## 旧兼容数字的边界

旧 ADB 单格为 OOS F1 89.47、F1-All 87.63；旧 DA-ADB 单格为 OOS F1 90.90、F1-All 89.23。它们使用独立 TextOIR snapshot、BERT 和单 seed，不能与当前 Trainable/Frozen 五 seed fair rows 合并排名，也不能被称为当前 protocol_v2 的正式 SOTA 对比。

## 当前实验结论

ADB 已有 45 个同源数据、同 Known-list 合同的有效外部兼容单元；DA-ADB 已有 StackOverflow/KIR=.50 的三个当前协议外部单元。二者由于使用 BERT/TextOIR 训练合同，仍只能作为外部参照，不能与 MiniLM fair 行合并成同监督排名，也不能仅凭当前适配结果宣称超过或落后于 TextOIR published baseline。

## 下一步

1. 不重复已审计的 ADB 45/45 单元；它们已作为 BERT/TextOIR 外部参照进入比较总览；
2. DA-ADB 不扩展新的 KIR/数据集矩阵，先完成当前三个单元与 published/legacy 合同的逐项复核；
3. 只有在合同差异、Known-list、seed 和评估器对齐后，才允许把 DA-ADB 提升为更强的外部比较层；
4. 在外部基线未闭合前，继续使用当前 fair 矩阵和逐样本错误/工作点图做实验分析，不重复 E2/E3/K 网格。

详细执行记录：`results/analysis/archive/analysis/baseline_execution_status_v1/attempt_status.csv`、`protocol_data_hashes.csv`、`MANIFEST.json`。

## Runtime follow-up（2026-08-08）

为避免把单一解释器故障误判为 ADB 算法失败，另外检查了当前可见的多个 Python runtime（项目默认环境、`bo`、`implicit_intent` 和 `textoir-py39`）以及 CPU-only 环境变量组合。多个 runtime 在执行最小 `import torch` 或 CUDA 状态检查时均在 12–15 秒超时；此前同一环境还出现过原生 signal 6 和 D-state 进程。`nvidia-smi` 能看到 RTX 5070，但没有稳定的可用计算进程。

因此当前阻断被细化为：

```text
protocol data contract: verified
TextOIR source checkout: clean/read-only
model files: converted isolated pytorch_model.bin available
torch runtime import/CUDA probe: not reproducible
same-protocol ADB/DA-ADB metrics: unavailable
```

这不是 ADB/DA-ADB 的性能结论，也不是 StackOverflow 数据错误。下一次尝试必须使用经过最小 torch import、BERT forward 和 CUDA/CPU smoke 验证的独立环境；在该条件满足前，不再启动外部训练进程。

## Runtime follow-up（2026-08-09）

再次执行了不启动训练的最小探针：

- RTX 5070 路径在 `/home/bo/anaconda3/bin/python3` 下无稳定 stdout，超时后被安全终止；
- `CUDA_VISIBLE_DEVICES=''` 的 CPU-only `import torch` 触发 `free(): double free detected in tcache 2`，返回码 134；
- `nvidia-smi` 仍可见 GPU，但这不能证明 Python/PyTorch forward 可用。
- 同一环境下 `pytest tests/unit -q` 也无法在合理时间内完成，进程进入 D 状态后被安全终止；这被视为环境验证失败，不是单元测试通过或算法失败。

结论保持为 `runtime_blocked_no_metrics`：没有启动 ADB、DA-ADB 或其他外部训练，没有新增或覆盖任何结果。下一次必须更换或修复隔离 Python/PyTorch 环境，并先通过 torch import、BERT forward、单批 CUDA/CPU smoke，才允许启动一个 StackOverflow/KIR=.50/seed=42 单格。

## Runtime follow-up（补充本地环境探针）

为避免遗漏可复用的本地环境，又检查了 `minimind`、`dreamer` 和 `dreamer_tf`：

- `minimind` 环境的 `import torch` 在 8 秒内超时；
- `dreamer` 与 `dreamer_tf` 没有安装 PyTorch；
- 未启动任何 BERT forward、ADB 或 DA-ADB 训练进程。

因此当前可见环境仍没有满足“torch import → BERT forward → CPU/GPU smoke”的候选运行时。
该检查只加强 runtime blocker 证据，不构成 ADB/DA-ADB 性能结论，也不改变已有数据、registry 或实验结果。

## Runtime follow-up（2026-08-09：协议数据根与 ADB 单格复核）

- 已使用 `tools/compat/textoir/build_protocol_data_root.py` 将
  `data/exports/protocol_v2_textoir_v1/adb/stackoverflow/seed_42/kir_0.50/`
  复制到独立 artifact 数据根；train/dev/test/known_labels 的复制前后 SHA256 一致。
- ADB dry-run 已通过，manifest 明确记录 TEXTOIR commit、Known labels、split SHA256 和完整命令；
  该过程不读取 `textoir/data`，也没有启动训练。
- 实际 ADB 尝试在训练前环境探针停止：默认 `/home/bo/anaconda3/bin/python` 缺少
  `easydict`；带有 `easydict` 的 `textoir-py39` 环境在 `torch.cuda.is_available()` 探针超时。
- 因此 ADB/DA-ADB 仍保持 `runtime_blocked_no_metrics`，没有新增同协议性能数字；既有
  `legacy_compatibility_only` 单格不能升级为 fair 主表。

证据 artifact：
`../artifacts/s2c/external/adb_protocol_v2_probe_data_v1/PROTOCOL_DATA_ROOT_MANIFEST.json`、
`../artifacts/s2c/external/adb_protocol_v2_probe/stackoverflow/ADB/kir50/seed42/run_manifest.json`。

## DCLOOS 端到端基线（2026-08-10）

- 官方仓库和外部 SQuAD 负样本来源已闭合；负样本快照 SHA256 为
  `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`。
- 当前 StackOverflow/KIR=.50/seed=42 的 protocol registry 有 10 个 Known intent；DCLOOS 在同一
  `train.tsv` 上按自身 `known_cls_ratio=.50` 再随机抽 5 个 Known intent，列表不一致。因此它不能标成
  `same_protocol`，只能标成 `dcloos_adapted_protocol_migration`。
- 隔离 smoke 已完整通过到测试/指标导出：1 epoch、BERT、pseudo-OOS+外部 SQuAD，OOS F1=0、F1-All=8.31%。
  该数字只说明链路完整，预算远低于收敛训练，不进入任何性能排名。
- 期间定位并在运行时 overlay 记录了三个上游兼容问题：Python 标量 `.item()`、epoch 从 1 开始导致
  `annealing_kl` 越界、旧 Transformers tokenizer/AdamW/本地 BERT 路径；没有修改第三方源码。
- 详细合同说明、5 个隔离 artifact 及“不得混排”规则见 `docs/archive/analysis/DCLOOS_CONTRACT_STATUS_V1.md`。

因此 DCLOOS 当前仍为 `adapted_smoke_complete_but_not_comparable`；要得到可比较数字，下一步必须
先实现固定 `known_labels_file` adapter，再运行一个收敛单格，之后才考虑多 seed。

## MSP 外部兼容探测（2026-08-10）

- 已完成 StackOverflow/KIR=.50/seed=42 的 dry-run。该探测复用了当前 protocol 的 10 个 Known labels、
  train/dev/test SHA256 和独立数据根；TEXTOIR upstream commit 为
  `dffe2b1b848a069a6808f8089b4cb9bd16e2062b`，工作树干净。
- 实际训练没有开始：外部 runtime shim 在 `probe_python_environment` 阶段 90 秒超时，未取得
  `torch/transformers` 环境信息，也没有生成预测、指标或 checkpoint。原始运行清单仍保留
  `status=dry_run`，不能将该单元写成 MSP 性能结果。
- 当前判定：`runtime_blocked_no_metrics`。这不是 MSP 算法失败，也不是 StackOverflow 数据失败；
  需要先有能通过最小 torch import、BERT forward 和单批 GPU/CPU smoke 的隔离 runtime，才可重试一个
  收敛单格。
- 证据 artifact：`../artifacts/s2c/external/msp_protocol_v2_stackoverflow_kir50_seed42_v1/`。

该探测保持与 ADB/DA-ADB、DCLOOS 相同的合同分层：没有可审计指标的外部方法不进入比较排名。

### Runtime follow-up（2026-08-10）

再次只执行最小探针，没有启动训练。`/home/bo/anaconda3/bin/python` 在 `import torch` 阶段进入
D-state，20 秒 timeout 无法回收；同机没有稳定的 CUDA compute process。该结果进一步确认当前阻断是
Python/PyTorch runtime，而不是 MSP 的语义指标或 StackOverflow 数据问题。后续不再重复外部训练，
除非先更换隔离环境并通过 `import torch`、BERT forward 和单批 CPU/GPU smoke。

## Runtime follow-up（2026-08-09：兼容运行已获得可审计单元）

前述 `runtime_blocked_no_metrics` 已被更具体的结果覆盖，旧条目保留作历史审计：

| 方法 | 单元 | 当前判定 | 统一指标来源 |
|---|---|---|---|
| ADB | StackOverflow/KIR=0.50/seed=42 | `valid_same_protocol_external_cell` | `y_true.npy`/`y_pred.npy` |
| ADB | StackOverflow/KIR=0.50/seed=87 | `valid_same_protocol_external_cell` | `y_true.npy`/`y_pred.npy` |
| ADB | StackOverflow/KIR=0.50/seed=100 | `valid_same_protocol_external_cell` | `y_true.npy`/`y_pred.npy` |
| DA-ADB | StackOverflow/KIR=0.50/seed=42 原始兼容 | `invalid_metrics_nan_all_class` | 逐样本预测审计 |
| DA-ADB | StackOverflow/KIR=0.50/seed=42 clamp30 适配 | `invalid_metrics_all_class_prediction` | 逐样本预测审计 |

ADB seed=42/87/100 的统一重算结果已经写入
`results/analysis/archive/analysis/comparison_atlas_v1/stackoverflow_kir50_external_and_fair_cells.csv`。
ADB 三个有效单元的 OOS F1 分别为 86.30%、86.99% 和 89.13%，F1-All 分别为
84.30%、85.32% 和 87.40%；这些数字仍然是 BERT/TextOIR 兼容合同，不得与 MiniLM
fair Gate 行伪装成同一训练条件。

DA-ADB 的进程返回码为 0 不能视为成功：训练日志出现 NaN，最终 `y_pred` 全部为
类别 0，统一 OOS F1=0、F1-All≈0.87%。官方 `results.csv` 与逐样本预测不一致，
因此该单元保留为 invalid，不进入有效排名。

最新机器可读收口：
`results/analysis/archive/analysis/comparison_atlas_v1/STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md`、
`external_same_protocol_cells.csv`、`external_invalid_semantic_runs.csv`。

## Runtime follow-up（2026-08-10：本机解释器穷举探针）

为判断外部基线是否只是某一个环境的问题，又在本机可见解释器上执行了不启动训练的
`import torch` 探针：`bo`、`textoir-py39`、`implicit_intent` 均在 8 秒内超时；`dreamer`
没有安装 PyTorch；系统 Python 也没有 PyTorch。没有产生 BERT forward、训练、预测或新指标。

因此当前外部 baseline 的可审计状态保持不变：ADB 的 3 个已有兼容 cell 可以作为 BERT/TextOIR
外部参照，DA-ADB 仍 invalid，DCLOOS 仍是 reduced/adapted 证据；不得因为解释器探针失败而把
任何方法写成算法失败。只有建立独立、可回收的 PyTorch runtime 并通过 BERT forward 与单批 CPU/GPU
smoke 后，才允许恢复外部单格实验。

## Runtime follow-up（2026-08-10：独立 CUDA runtime 三 seed 已完成）

已建立并通过最小 CUDA smoke 的隔离环境 `/tmp/s2c_gpu_runtime_20260810`：
PyTorch `2.9.1+cu128`、Transformers `4.46.3`、RTX 5070。随后在新的独立 artifact root
完成 ADB StackOverflow/KIR=.50/seed=`42,87,100` 三个单元，三个 `run_manifest.json` 均为
`status=complete`，逐样本 `y_true.npy/y_pred.npy` SHA256 审计通过。

这组三 seed 的 y_true/y_pred 重算均值为：OOS F1=`87.36±1.61%`、F1-All=`85.66±1.59%`、
Known Recall=`80.78±1.44%`、false acceptance=`7.52±1.97%`。完整报告见
`docs/archive/analysis/ADB_GPU_RUNTIME_THREE_SEED_V1.md`，机器可读审计见
`results/analysis/archive/analysis/external_gpu_runtime_comparison_v1/`。

该结果替代此前同 seed 的旧兼容运行数值作为最新 ADB 外部参照，但仍保持
`BERT/TextOIR external contract` 标记；不能与 MiniLM fair Gate 合并排名，也不能推断
ADB 的独立算法增益。DA-ADB 仍 invalid，DCLOOS 仍为额外 pseudo/external-OOS 监督合同。

## Runtime follow-up（2026-08-10：DA-ADB 隔离 CUDA 单格已获得有效预测）

为确认 DA-ADB 的 NaN/全类预测是否只是运行时故障，使用同一隔离 CUDA 环境
`/tmp/s2c_gpu_runtime_20260810`，在 StackOverflow/KIR=`0.50`/seed=`42` 上重新运行。该单格：

- `run_manifest.json` 为 `status=complete`，返回码为 0；
- 训练日志全程为有限值，逐样本 `y_true.npy/y_pred.npy` 形状一致、无 NaN/Inf，预测包含全部 11 个标签；
- 数据、Known labels、split SHA256 与 protocol_v2 外部适配根一致，未读取 `textoir/data`；
- OOS F1=`70.82%`，F1-All=`72.03%`，F1-Known=`72.15%`，Accuracy=`70.15%`，Known Recall=`72.93%`，
  false acceptance=`30.33%`，false rejection=`27.07%`。

因此 DA-ADB 当前状态从“无有效预测”细化为：

```text
valid_external_cell_pending_replication
```

它证明了兼容层可以产生可审计 DA-ADB 输出，但不证明当前实现达到论文结果。新单格的 OOS F1 明显低于旧 seed=0 兼容单格 `90.90%`，差异可能来自 Known list、seed、数据快照、环境或兼容配置；在完成逐项对齐前不能挑选高值，也不能把它纳入 MiniLM fair 排名。机器可读证据：
`results/analysis/archive/analysis/da_adb_gpu_runtime_v1/`；详细报告：
`docs/archive/analysis/DA_ADB_GPU_RUNTIME_SINGLE_CELL_V1.md`。
