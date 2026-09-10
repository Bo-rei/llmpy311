# RC-AMBL 实验复现说明

## 固定范围

本阶段只允许运行 StackOverflow、KIR=`0.50`、seed=`13/42/87`。不得把该命令改成其他数据集、KIR 或 seed 来替代本阶段结果；不得覆盖 `contract_repair1`--`contract_repair3` 历史 artifact。

## 运行命令

```bash
cd /home/bo/bo01/llmpy311/s2c

# 无写入协议检查
python scripts/experiments/adaptive_v1/run_adaptive_v1.py \
  --dataset stackoverflow --kir 0.50 --seed 42 --dry-run

# 三个正式 pilot cell
python scripts/experiments/adaptive_v1/run_adaptive_v1.py \
  --dataset stackoverflow --kir 0.50 --seed 13
python scripts/experiments/adaptive_v1/run_adaptive_v1.py \
  --dataset stackoverflow --kir 0.50 --seed 42
python scripts/experiments/adaptive_v1/run_adaptive_v1.py \
  --dataset stackoverflow --kir 0.50 --seed 87

# 校验、汇总、诊断和图表
python scripts/experiments/adaptive_v1/verify_adaptive_v1.py
python scripts/experiments/adaptive_v1/summarize_adaptive_v1.py
python scripts/experiments/adaptive_v1/diagnose_adaptive_v1.py
MPLBACKEND=Agg python scripts/experiments/adaptive_v1/plot_adaptive_v1.py
```

脚本默认写入：

```text
../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/
```

轻量 CSV 和 PNG 写入：

```text
results/diagnostics/adaptive_v1/
```

## 输入和环境

- canonical、registry、views、exports：`data/protocol_v2_textoir_v1/`；
- 冻结 MiniLM cache：由当前协议的 canonical embedding cache 提供，运行时不下载、不重新编码；
- 所有 registry/view/export 的 SHA 在每个 seed 的 `provenance.json` 中记录；
- Python、NumPy 和输入 hash 记录在每个 seed 的 `provenance.json`；
- `ADAPTIVE_V1_RESULT_MANIFEST.json` 保存 contract_repair5 根目录内文件的 SHA256。

## 结果读取

- 每个 seed 的两种方法指标：`runs/stackoverflow/seed_*/metrics.csv`；
- 训练/校准选择审计：`*_selection_audit.json`；
- 阈值：`*_thresholds.json`；
- 候选中心和操作：`*_centers.json`、`*_operations.csv`；
- 测试只保留哈希后的 sample id：`*_predictions.csv`；
- 主表：`results/diagnostics/adaptive_v1/main_results.csv`；
- 分裂与 K_y 诊断：`diagnostics/ky_distribution.csv`、`center_operations.csv`、`intent_split_summary.csv`。

## 复现限制

本阶段的 RC-AMBL evidence 阈值是 Known calibration 合同，并非 E2 `nearest_sphere` 的逐值复现；E2 K=1、K=2 和 E3 random-balanced 行在主表中明确标记为 `validated_reused`。MOGB 行是冻结 MiniLM 组件参考，不是官方 BERT 严格复现。DCLOOS 使用额外 pseudo/external OOS 监督，本轮不纳入主表。

StackOverflow 完整文本、embedding、checkpoint 和逐样本 score 不进入公开结果目录；公开目录只保留轻量统计、配置和图表。

