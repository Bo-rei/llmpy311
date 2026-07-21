# s2c 实验目录与读法

本文件只保留“实验问题 → 入口 → 结果位置”的映射。历史轮次名称仍会出现在目录和 manifest 中，但不再作为新的文档体系。

> 本文件是结果阅读入口，不是实验计划。当前能否运行某个命令，以 `main` checkout 的源码和
> 对应 manifest 为准；artifact 目录存在并不代表生成它的脚本仍在当前分支。

## 先分两类结果

### Gate-only

只评价 OOS Gate，不经过 Router/Expert。适合回答：

- 多簇是否改善 Known/OOS 可分性；
- near-OOS 是表示碰撞还是边界过覆盖；
- K、距离、MiniLM 表示和 Baseline 的几何差异。

主要指标：`oos_f1`、`auroc`、`aupr_oos`、`fpr95`、`id_recall`、near-OOS 指标。

### 完整 Cascade

评价 `Gate → Router → Expert`。适合回答：

- Gate 改善是否传到系统；
- Known macro-F1 和 overall accuracy 是否保持；
- 错误究竟发生在 Gate、Router 还是 Expert。

主要指标：`oos_f1`、`known_macro_f1`、`overall_accuracy`、`error_stage`。

两类结果不能合并成一张“总 F1”表。

## 目录地图

| 目录 | 内容 | 论文用途 |
| --- | --- | --- |
| `cluster_separability_v19/` | KIR × K × distance、基础 Gate Baseline | Gate 主表/附录 |
| `cluster_separability_v20/` | 随机分簇、selected-K、near/far、效率 | 机制附录 |
| `cluster_separability_v21/` | MiniLM 邻域、语义间隔、表示对照 | MiniLM 机制主表/附录 |
| `minilm_representation_analysis/` | CE-Recon、碰撞/过覆盖、边界形状 | 表示机制分析 |
| `external_validation/hard_negative_oos/` | 外部 hard-negative zero-adjustment | 外部有效性 |
| `cascade_repair/gpu_kir50_seed42/` | 本轮修复后的下游模型和 12 个完整 Cascade | 系统验证 |
| `cascade_full/gpu_kir50/` | KIR50 × 三 seed × 四 Gate 的完整 Cascade，36/36 已完成 | 系统稳定性主表 |
| `study_closeout/` | 结果清单、hash、论文来源、已知缺口 | provenance |

## 读取一个实验单元

按这个顺序，不要先打开逐样本大文件：

```text
1. matrix_manifest.json 或 run_manifest.json
2. eval_results.json
3. 汇总 CSV（如果有）
4. predictions.json / scores.parquet（只用于错误案例和复核）
5. training_manifest.json、checkpoint_selection.json
```

重点检查：

- `dataset / kir / data_seed` 是否与比较对象相同；
- OOS 是否为正类、分数方向是否一致；
- 阈值和 K 是否只在 validation 选择；
- CE-Recon 是否使用了对应的 encoder checkpoint；
- Cascade 是否使用本轮修复后的 Expert，而不是旧 smoke checkpoint。

## 当前代表性 Cascade 命令

先做预检（不会加载模型，也不会改结果）：

```bash
python tools/eval/run_cascade_repair.py
```

确认三数据集都是 `preflight_only` 且 `missing` 为空后，再执行：

```bash
source /home/bo/anaconda3/etc/profile.d/conda.sh
conda activate bo
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparselt/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib"
PYTHONPATH="$PWD" python tools/eval/run_cascade_repair.py --execute
```

如果只想重跑一个数据集：

```bash
python tools/eval/run_cascade_repair.py --dataset banking77_oos --execute
```

完整 KIR50 矩阵先准备下游组件和 Gate 适配器（这不是 12 单元代表性结果）：

```bash
python tools/train/run_cascade_components.py
python tools/eval/prepare_cascade_gates.py
```

确认 `missing` 为空后，GPU 训练缺失组件并运行完整矩阵；若目录已有单元，脚本应复用而不是覆盖：

```bash
python tools/train/run_cascade_components.py --execute
python tools/eval/prepare_cascade_gates.py --execute
python tools/eval/run_cascade_matrix.py --execute
```

该矩阵固定 `KIR=50`，因为 CE-Recon encoder 当前只覆盖 KIR50；不要把它
误写成 KIR25/75 的完整四 Gate 结果。

代表性 seed42 修复结果的默认输出：

```text
../artifacts/s2c/outputs/experiments/cascade_repair/gpu_kir50_seed42/evaluations/
```

完整三 seed 矩阵使用独立输出根目录：

```text
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/evaluations/
```

当前 `matrix_manifest.json` 已记录 `status=complete`、`completed_unit_count=36`。再次执行矩阵命令只会复用已有 `eval_results.json`，不会重新训练或覆盖单元。完整矩阵汇总为：

```text
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/cascade_summary.csv
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/cascade_error_decomposition.csv
```

完成后用聚合器生成论文可读表；对哪个根目录运行，汇总就写回哪个根目录：

```bash
python tools/analysis/export_cascade_repair_summary.py \
  --input-root ../artifacts/s2c/outputs/experiments/cascade_repair/gpu_kir50_seed42
```

它会生成 `cascade_summary.csv`、`cascade_error_decomposition.csv` 和 `repair_provenance.json`；模型 SHA256 与 GPU/PyTorch 版本记录在同目录的 `repair_manifest.json`。

脚本检测到 `eval_results.json` 时会复用该单元，不会覆盖已有结果；需要新的协议时应使用新的显式输出目录。

## 如何比较四个 Gate

```text
frozen_k1                 Frozen MiniLM + K=1
frozen_selected_k         Frozen MiniLM + validation-selected K
ce_recon_selected_k       CE-Recon encoder + 对应多簇 detector
best_controlled_baseline  validation 选出的线性 Gate（如 MSP/Entropy/Energy）
```

四个 Gate 在同一个 dataset/KIR/seed 下必须共用 Router 和 Expert。否则结果差异不能归因于 Gate。

## 不要误读的结果

- `selected K=1` 是有效结论，不是实验失败；它表示 validation 没有支持多簇。
- StackOverflow 的 `K>1` 退化不能用更多 K 扫描“修掉”，应作为支持结构不匹配的负结果。
- 低 Known macro-F1 先查 Router、Expert 单点和 label map，再讨论 Gate。
- 外部 hard-negative 只用于测试，不能拿它重新调阈值或选 K。
- MOGB 审计文件不是 MOGB 性能表。
