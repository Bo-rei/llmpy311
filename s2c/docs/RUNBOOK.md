# s2c 运行手册

所有命令从 `s2c/` 项目根目录执行。当前交付分支是 `main`；不要切换、提交或推送 `autoresearch`。

## 环境

CPU 预检和单元测试可使用系统 Python；模型训练/评价使用已验证的 `bo` 环境：

```bash
source /home/bo/anaconda3/etc/profile.d/conda.sh
conda activate bo
python -V
```

当前 baseline 序列化审计使用 sklearn `1.8.0`，已加载 9 个 baseline 文件且 warning 数为 0；没有需要因兼容性 warning 重训的模型。不要把 `environment.yml` 中的历史版本字符串当作实际运行版本。

## 最小预检

```bash
git status --short
python -m py_compile src/pipeline/system_pipeline.py \
  tools/eval/eval_system_pipeline_v19.py \
  tools/eval/run_cascade_repair.py \
  tools/analysis/export_cascade_repair_summary.py \
  tools/analysis/audit_experiment_registry.py
pytest tests/unit -q
git diff --check
```

## 完整 Cascade 预检

这些命令只读 manifest，不加载 GPU 模型，不覆盖结果：

```bash
/home/bo/anaconda3/envs/bo/bin/python tools/train/run_cascade_components.py
/home/bo/anaconda3/envs/bo/bin/python tools/eval/prepare_cascade_gates.py
/home/bo/anaconda3/envs/bo/bin/python tools/eval/run_cascade_matrix.py
```

应看到：9 个组件 ready、9 个 Gate adapter ready、matrix `36/36 complete`。若不满足，先检查对应 manifest 的 `missing`，不要直接扩大实验矩阵。

## 复现/继续运行

只在预检完整通过且确实需要重跑时执行：

```bash
/home/bo/anaconda3/envs/bo/bin/python tools/train/run_cascade_components.py --execute
/home/bo/anaconda3/envs/bo/bin/python tools/eval/prepare_cascade_gates.py --execute
/home/bo/anaconda3/envs/bo/bin/python tools/eval/run_cascade_matrix.py --execute
```

脚本会复用已有 `eval_results.json`；新的协议必须指定新的输出根目录，禁止覆盖旧 artifact。

代表性 seed42 入口：

```bash
/home/bo/anaconda3/envs/bo/bin/python tools/eval/run_cascade_repair.py
/home/bo/anaconda3/envs/bo/bin/python tools/eval/run_cascade_repair.py --execute
```

输出在 `../artifacts/s2c/outputs/experiments/cascade_repair/gpu_kir50_seed42/`，完整三 seed 输出在 `cascade_full/gpu_kir50/`。

## 汇总和冻结

```bash
/home/bo/anaconda3/envs/bo/bin/python tools/analysis/export_cascade_repair_summary.py \
  --input-root ../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50 \
  --protocol cascade_full_kir50_three_seed_four_gate

/home/bo/anaconda3/envs/bo/bin/python tools/analysis/audit_experiment_registry.py
```

汇总器只聚合 `eval_results.json`，不会重新选择 K/阈值。登记审计会检查 unit count，并把清单 hash 写入 artifact 下的 `study_closeout/pipeline_freeze_manifest.json`。

## 结果读取顺序

1. `matrix_manifest.json`：完成性、split、unit 数量。
2. `cascade_gate_summary.csv`：三 seed 均值/标准差和相对 Frozen K=1 的配对差。
3. `cascade_banking_tradeoff.csv`：Banking77 的 OOS/ID 取舍。
4. `cascade_error_decomposition_summary.csv`：Gate、Router、Expert 阶段错误。
5. 单元 `eval_results.json`、`predictions.json`：只用于复核和案例。

## 常见误读和排查

### 把旧 smoke 当成当前下游结果

先查看 `cascade_full/gpu_kir50/component_plan.json` 和 Expert `results.json`。Banking77/StackOverflow 当前使用 constant router + manifest domain + 修复后的 Expert；旧低预算 smoke 不进入主表。

### 看到 Known macro-F1 下降就认为 Expert 坏了

先看同一行 `id_recall`、`known_false_reject_rate` 和 `oos_false_accept_rate`。若 Gate false reject 上升而 Router/Expert error 没有同步上升，这是 operating-point trade-off。

### 误把 Gate-only OOS F1 当成端到端分类结果

Gate-only 不会产生具体 intent；完整系统必须读取 `Known macro-F1`、`overall_accuracy` 和错误阶段。

### registry audit 失败

检查 manifest/summary 路径是否相对 `artifact_root`，以及 CSV 行数是否与登记协议一致。不要通过修改 expected count 掩盖缺失单元。

### 需要清理脚本

先读 `configs/active_entrypoints.json` 和 `configs/unreferenced_entrypoints_report.json`。历史脚本若仍被测试、旧配置或 manifest 引用，保持兼容；未引用脚本只报告，不批量删除。
